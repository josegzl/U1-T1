import torch
import torch.nn.functional as F

def detect_edges(image_tensor):
    # Example implementation for detecting edges using Sobel operator
    
    # Define Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
    
    # Apply Sobel filters
    gradient_x = F.conv2d(image_tensor, sobel_x, padding=1)
    gradient_y = F.conv2d(image_tensor, sobel_y, padding=1)
    
    # Calculate gradient magnitude
    edge_magnitude = torch.sqrt(gradient_x**2 + gradient_y**2)
    
    # Normalize edge magnitude
    edges = edge_magnitude / edge_magnitude.max()
    
    return edges
