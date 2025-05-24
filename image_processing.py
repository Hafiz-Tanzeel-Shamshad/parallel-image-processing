import cv2
import numpy as np
from PIL import Image
import time

def pil_to_cv2(pil_image):
    """Convert PIL Image to OpenCV format."""
    # Convert to RGB if has alpha channel
    if pil_image.mode == 'RGBA':
        pil_image = pil_image.convert('RGB')
    
    # Convert PIL Image to numpy array
    img_array = np.array(pil_image)
    
    # Convert RGB to BGR (OpenCV format)
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    return img_array

def cv2_to_pil(cv2_image):
    """Convert OpenCV image to PIL format."""
    # Convert BGR to RGB if it's a color image
    if len(cv2_image.shape) == 3 and cv2_image.shape[2] == 3:
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    return Image.fromarray(cv2_image)

def apply_edge_detection(image_path, params):
    """
    Apply Canny edge detection to an image.
    
    Args:
        image_path: Path to the image file
        params: Dictionary containing thresh1 and thresh2 parameters
        
    Returns:
        PIL Image with edge detection applied
    """
    # Start timing
    start_time = time.time()
    
    # Load the image
    if isinstance(image_path, str):
        # Load from file path
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    else:
        # Convert PIL Image to cv2 format
        img = pil_to_cv2(image_path)
    
    # Convert to grayscale if it's a color image
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    thresh1 = params.get('thresh1', 100)
    thresh2 = params.get('thresh2', 200)
    edges = cv2.Canny(blurred, thresh1, thresh2)
    
    # Convert back to PIL Image
    result = cv2_to_pil(edges)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    return result, processing_time

def apply_gaussian_blur(image_path, params):
    """
    Apply Gaussian blur to an image.
    
    Args:
        image_path: Path to the image file
        params: Dictionary containing kernel_size and sigma parameters
        
    Returns:
        PIL Image with Gaussian blur applied
    """
    # Start timing
    start_time = time.time()
    
    # Load the image
    if isinstance(image_path, str):
        # Load from file path
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    else:
        # Convert PIL Image to cv2 format
        img = pil_to_cv2(image_path)
    
    # Get parameters
    kernel_size = params.get('kernel_size', 5)
    sigma = params.get('sigma', 0)
    
    # Make sure kernel size is odd
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
    
    # Convert back to PIL Image
    result = cv2_to_pil(blurred)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    return result, processing_time

def apply_histogram_equalization(image_path, params):
    """
    Apply histogram equalization to an image.
    
    Args:
        image_path: Path to the image file
        params: Dictionary containing clip_limit for CLAHE
        
    Returns:
        PIL Image with histogram equalization applied
    """
    # Start timing
    start_time = time.time()
    
    # Load the image
    if isinstance(image_path, str):
        # Load from file path
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    else:
        # Convert PIL Image to cv2 format
        img = pil_to_cv2(image_path)
    
    # Get parameters
    clip_limit = params.get('clip_limit', 2.0)
    
    # Check if the image is grayscale or color
    if len(img.shape) == 3:
        # Split the image into channels
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge the CLAHE enhanced L-channel with the a and b channel
        limg = cv2.merge((cl, a, b))
        
        # Convert back to BGR
        result_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        # Apply CLAHE directly to grayscale image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        result_img = clahe.apply(img)
    
    # Convert back to PIL Image
    result = cv2_to_pil(result_img)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    return result, processing_time
