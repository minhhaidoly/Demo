import torch
import torch.nn as nn
import numpy as np
import cv2

class RetinalVesselSegmentationCNN(nn.Module):
    def __init__(self, kernel_size=(3, 3)):
        super(RetinalVesselSegmentationCNN, self).__init__()
        # First convolutional layer with 20 feature maps
        self.conv1 = nn.Conv2d(1, 20, kernel_size=kernel_size, padding='same')
        self.sigmoid1 = nn.Sigmoid()
        
        # Second convolutional layer with 20 feature maps
        self.conv2 = nn.Conv2d(20, 20, kernel_size=kernel_size, padding='same')
        self.sigmoid2 = nn.Sigmoid()
        
        # Third convolutional layer with 20 feature maps
        self.conv3 = nn.Conv2d(20, 20, kernel_size=kernel_size, padding='same')
        self.sigmoid3 = nn.Sigmoid()
        
        # Average pooling layer (subsample only once after the third conv layer)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Calculate the size of flattened features after conv and pooling layers
        # For a 27×27 input patch, after three conv layers and one pooling, we get:
        # 27×27 → conv1 → 27×27 → conv2 → 27×27 → conv3 → 27×27 → pool → 13×13
        # So the flattened size would be 20×13×13 = 3380
        flattened_size = 20 * 27 // 2 * 27 // 2
        
        # Fully connected layer (with 3 outputs for 3 classes: background, large vessels, small vessels)
        self.fc = nn.Linear(flattened_size, 3)
        self.sigmoid_out = nn.Sigmoid()
        
    def forward(self, x):
        # First convolutional layer
        x = self.conv1(x)
        x = self.sigmoid1(x)
        
        # Second convolutional layer
        x = self.conv2(x)
        x = self.sigmoid2(x)
        
        # Third convolutional layer
        x = self.conv3(x)
        x = self.sigmoid3(x)
        
        # Pooling layer (only one after the third conv layer)
        x = self.avgpool(x)
        
        # Flatten the output
        x = torch.flatten(x, 1)
        
        # Fully connected layer
        x = self.fc(x)
        x = self.sigmoid_out(x)
        
        return x

def create_cnn_model(kernel_variant):
    """Create a CNN model based on the specified kernel variant."""
    if kernel_variant == "3x3":
        # CNN with 3×3 kernels for all layers
        return RetinalVesselSegmentationCNN(kernel_size=(3, 3))
    elif kernel_variant == "5x5":
        # CNN with 5×5 kernels for all layers
        return RetinalVesselSegmentationCNN(kernel_size=(5, 5))
    elif kernel_variant == "hybrid":
        # Custom CNN with 5×5 kernels for first layer and 3×3 for subsequent layers
        model = RetinalVesselSegmentationCNN(kernel_size=(3, 3))
        model.conv1 = nn.Conv2d(1, 20, kernel_size=(5, 5), padding='same')
        return model
    else:
        raise ValueError(f"Unknown kernel variant: {kernel_variant}")

def preprocess_image(image):
    """
    Preprocess the image as described in the paper:
    1. Extract green channel
    2. Apply adaptive histogram equalization
    3. Apply white top-hat filtering
    
    Args:
        image: RGB fundus image (numpy array)
    
    Returns:
        Preprocessed image (numpy array)
    """
    # Extract green channel (most informative for vessel segmentation)
    green_channel = image[:, :, 1]
    
    # Apply adaptive histogram equalization CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(green_channel)
    
    # Apply white top-hat filtering to enhance vessels and remove bright structures
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(255 - equalized, cv2.MORPH_TOPHAT, kernel)
    
    return tophat

def segment_image(model, image, device="cpu"):
    """
    Segment a full retinal image using the patch-based approach.
    
    Args:
        model: Trained RetinalVesselSegmentationCNN model
        image: RGB fundus image (numpy array)
        device: Device to run the model on ("cpu" or "cuda")
    
    Returns:
        Segmentation mask (numpy array)
    """
    model.to(device)
    model.eval()
    
    # Preprocess the image
    processed = preprocess_image(image)
    
    # Pad the image to handle boundary conditions
    patch_size = 27
    pad_size = patch_size // 2
    padded = cv2.copyMakeBorder(processed, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
    
    # Create output segmentation mask
    segmentation = np.zeros_like(processed)
    
    # Slide a window over the image and classify each patch
    with torch.no_grad():
        for i in range(pad_size, padded.shape[0] - pad_size):
            for j in range(pad_size, padded.shape[1] - pad_size):
                # Extract patch
                patch = padded[i - pad_size:i + pad_size + 1, j - pad_size:j + pad_size + 1]
                
                # Normalize and reshape patch
                patch = patch.astype(np.float32) / 255.0
                patch = patch.reshape(1, 1, patch_size, patch_size)
                
                # Convert to tensor
                patch_tensor = torch.from_numpy(patch).float().to(device)
                
                # Get prediction
                output = model(patch_tensor)
                _, predicted = torch.max(output, 1)
                
                # If predicted as vessel (class 1 or 2), set pixel in segmentation
                if predicted.item() > 0:
                    segmentation[i - pad_size, j - pad_size] = 255
    
    return segmentation

def generate_large_vessel_mask(mask):
    """
    Generate mask for large vessels by erosion of the original mask.
    
    Args:
        mask: Binary vessel mask
        
    Returns:
        Tuple of (full mask, large vessels mask, small vessels mask)
    """
    # Threshold to binary
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Erode to get large vessels as described in the paper
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    large_vessels = cv2.erode(mask, kernel)
    
    # Small vessels are the vessels in the original mask that are not in the large vessel mask
    small_vessels = mask - large_vessels
    
    return mask, large_vessels, small_vessels