import torch

image_tensor = torch.rand((1, 1, 100, 100)) 

def detect_points(image_tensor, threshold=0.9):
    # Example implementation for detecting points in an image using thresholding
    # Assuming image_tensor is of shape (1, 1, H, W) for grayscale images
    
    # Convert to binary based on threshold
    binary_tensor = (image_tensor > threshold).float()
    
    # Find coordinates of points
    points = torch.nonzero(binary_tensor, as_tuple=False)
    
    return points
