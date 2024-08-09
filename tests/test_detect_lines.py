import torch

import sys
import os


# Add the src directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from detect_lines import detect_lines
def test_detect_lines():
    # Sample test for line detection
    image_tensor = torch.rand((1, 1, 100, 100))  # Random tensor simulating an image
    result = detect_lines(image_tensor)
    assert result is not None, "Line detection failed."