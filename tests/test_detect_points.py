import torch

import sys
import os


# Add the src directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from detect_points import detect_points
def test_detect_points():
    # Sample test for point detection
    image_tensor = torch.rand((1, 1, 100, 100))  # Random tensor simulating an image
    result = detect_points(image_tensor)
    assert result is not None, "Point detection failed."