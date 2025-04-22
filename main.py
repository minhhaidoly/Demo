import os
import io
import base64
import numpy as np
import cv2
import torch
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from PIL import Image
from sklearn.metrics import accuracy_score, recall_score, precision_score
import time
import uvicorn
from functools import lru_cache

# Import model from model.py (assuming model.py from Question 1)
from model import RetinalVesselSegmentationCNN, create_cnn_model, preprocess_image, segment_image

app = FastAPI(title="RetinalScan API", 
              description="API for retinal vessel segmentation to aid in detecting diabetic retinopathy")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL_PATH = "models/"  # Directory containing model weights
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Model variants
MODEL_VARIANTS = {
    "3x3": "cnn_3x3.pth",
    "5x5": "cnn_5x5.pth",
    "hybrid": "cnn_hybrid.pth"
}

# Cache for loaded models to avoid reloading
models_cache = {}

class SegmentationResponse(BaseModel):
    segmented_image: str  # Base64 encoded image

class EvaluationResponse(BaseModel):
    segmented_image: str  # Base64 encoded image
    metrics: Dict[str, float]

class ModelInfo(BaseModel):
    id: str
    name: str
    description: str

@lru_cache(maxsize=3)
def load_model(variant: str):
    """Load model from disk with caching"""
    if variant not in MODEL_VARIANTS:
        raise HTTPException(status_code=400, detail=f"Model variant {variant} not found")
    
    model_path = os.path.join(MODEL_PATH, MODEL_VARIANTS[variant])
    
    # Check if model exists
    if not os.path.exists(model_path):
        # For demo purposes, we'll create a dummy model if the file doesn't exist
        model = create_cnn_model(variant)
        print(f"Warning: Model file {model_path} not found. Using untrained model.")
    else:
        # Load model
        model = create_cnn_model(variant)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Loaded model from {model_path}")
    
    model.to(DEVICE)
    model.eval()
    return model

def process_uploaded_image(file):
    """Process the uploaded image file"""
    try:
        # Read image file
        contents = file.file.read()
        image = np.array(Image.open(io.BytesIO(contents)))
        
        # Ensure image is RGB (3 channels)
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
    finally:
        file.file.close()

def image_to_base64(image):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.png', image)
    return base64.b64encode(buffer).decode('utf-8')

def calculate_metrics(segmented_mask, ground_truth):
    """Calculate evaluation metrics"""
    # Ensure binary masks
    segmented_binary = (segmented_mask > 127).astype(np.uint8).flatten()
    ground_truth_binary = (ground_truth > 127).astype(np.uint8).flatten()
    
    accuracy = accuracy_score(ground_truth_binary, segmented_binary)
    sensitivity = recall_score(ground_truth_binary, segmented_binary, zero_division=0)
    # Specificity = true_negatives / (true_negatives + false_positives)
    # Equivalent to recall_score for the negative class (0)
    tn = np.sum((ground_truth_binary == 0) & (segmented_binary == 0))
    fp = np.sum((ground_truth_binary == 0) & (segmented_binary == 1))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity)
    }

def batch_segment_image(model, processed_image, patch_size=27, batch_size=64):
    """
    Segment a full retinal image using batch processing for efficiency
    """
    # Create padded image
    pad_size = patch_size // 2
    h, w = processed_image.shape
    padded = cv2.copyMakeBorder(processed_image, pad_size, pad_size, pad_size, pad_size, 
                               cv2.BORDER_REFLECT)
    
    # Create output segmentation mask
    segmentation = np.zeros_like(processed_image)
    
    # Prepare batch coordinates
    coords = []
    for i in range(pad_size, padded.shape[0] - pad_size):
        for j in range(pad_size, padded.shape[1] - pad_size):
            if i - pad_size < h and j - pad_size < w:  # Ensure within original image bounds
                coords.append((i, j))
    
    # Process in batches
    with torch.no_grad():
        for batch_start in range(0, len(coords), batch_size):
            batch_coords = coords[batch_start:batch_start + batch_size]
            
            # Extract patches
            batch_patches = np.zeros((len(batch_coords), 1, patch_size, patch_size), dtype=np.float32)
            for idx, (i, j) in enumerate(batch_coords):
                patch = padded[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
                batch_patches[idx, 0] = patch / 255.0  # Normalize to [0,1]
            
            # Convert to tensor and predict
            batch_tensor = torch.from_numpy(batch_patches).float().to(DEVICE)
            outputs = model(batch_tensor)
            _, predictions = torch.max(outputs, 1)
            
            # Update segmentation mask
            for idx, (i, j) in enumerate(batch_coords):
                if predictions[idx].item() > 0:  # If vessel (class 1 or 2)
                    segmentation[i - pad_size, j - pad_size] = 255
    
    return segmentation

@app.get("/models", response_model=List[ModelInfo])
async def get_models():
    """Get available model variants"""
    return [
        ModelInfo(
            id="3x3", 
            name="3×3 Kernel CNN", 
            description="Uses 3×3 kernels for all convolutional layers, best for detecting fine details."
        ),
        ModelInfo(
            id="5x5", 
            name="5×5 Kernel CNN", 
            description="Uses 5×5 kernels for all layers, better at capturing larger vessel structures."
        ),
        ModelInfo(
            id="hybrid", 
            name="Hybrid (5×5 → 3×3) CNN", 
            description="Uses 5×5 kernel for first layer and 3×3 for subsequent layers, balancing context and detail."
        )
    ]

@app.post("/upload", response_model=SegmentationResponse)
async def upload_image(file: UploadFile = File(...), model_variant: str = Form("3x3")):
    """
    Upload an image for vessel segmentation
    """
    start_time = time.time()
    
    # Load model
    try:
        model = load_model(model_variant)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
    
    # Process uploaded image
    image = process_uploaded_image(file)
    
    # Preprocess image (green channel extraction, CLAHE, top-hat filtering)
    preprocessed = preprocess_image(image)
    
    # Segment image using batch processing
    segmented = batch_segment_image(model, preprocessed)
    
    # Convert segmented image to base64
    segmented_b64 = image_to_base64(segmented)
    
    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.2f} seconds")
    
    return SegmentationResponse(segmented_image=segmented_b64)

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_segmentation(
    image_file: UploadFile = File(...),
    ground_truth_file: UploadFile = File(...),
    model_variant: str = Form("3x3")
):
    """
    Upload an image and ground truth mask for evaluation
    """
    # Load model
    model = load_model(model_variant)
    
    # Process uploaded image and ground truth
    image = process_uploaded_image(image_file)
    ground_truth = process_uploaded_image(ground_truth_file)
    
    # Convert ground truth to grayscale if needed
    if len(ground_truth.shape) == 3:
        ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_RGB2GRAY)
    
    # Preprocess and segment image
    preprocessed = preprocess_image(image)
    segmented = batch_segment_image(model, preprocessed)
    
    # Calculate metrics
    metrics = calculate_metrics(segmented, ground_truth)
    
    # Convert segmented image to base64
    segmented_b64 = image_to_base64(segmented)
    
    return EvaluationResponse(
        segmented_image=segmented_b64,
        metrics=metrics
    )

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs(MODEL_PATH, exist_ok=True)
    
    # Run FastAPI with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)