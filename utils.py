# Handles utility functions like getting sample images and download links.
import os
import base64
from PIL import Image


def get_sample_images():
    """
    Creates and returns sample images for testing.
    
    Returns:
        Tuple of (list of PIL Images, list of temporary file paths)
    """
    images = []
    image_paths = []
    
    # Create temp directory if it doesn't exist
    if not os.path.exists("temp"):
        os.makedirs("temp")
    
    # Create and save sample images with different patterns
    image_size = (800, 600)
    
    # Sample 1: Gradient pattern
    img1 = Image.new('RGB', image_size)
    pixels = img1.load()
    for i in range(img1.width):
        for j in range(img1.height):
            r = int(255 * i / img1.width)
            g = int(255 * j / img1.height)
            b = int(255 * (i + j) / (img1.width + img1.height))
            pixels[i, j] = (r, g, b)
    img_path1 = "temp/sample_image_0.jpg"
    img1.save(img_path1)
    images.append(img1)
    image_paths.append(img_path1)
    
    # Sample 2: Checkerboard pattern
    img2 = Image.new('RGB', image_size)
    pixels = img2.load()
    square_size = 50
    for i in range(img2.width):
        for j in range(img2.height):
            if ((i // square_size) % 2 == 0 and (j // square_size) % 2 == 0) or \
               ((i // square_size) % 2 == 1 and (j // square_size) % 2 == 1):
                pixels[i, j] = (255, 255, 255)
            else:
                pixels[i, j] = (0, 0, 0)
    img_path2 = "temp/sample_image_1.jpg"
    img2.save(img_path2)
    images.append(img2)
    image_paths.append(img_path2)
    
    # Sample 3: Radial gradient
    img3 = Image.new('RGB', image_size)
    pixels = img3.load()
    center_x, center_y = img3.width // 2, img3.height // 2
    max_dist = ((img3.width // 2) ** 2 + (img3.height // 2) ** 2) ** 0.5
    for i in range(img3.width):
        for j in range(img3.height):
            dist = ((i - center_x) ** 2 + (j - center_y) ** 2) ** 0.5
            intensity = int(255 * (1 - dist / max_dist))
            pixels[i, j] = (intensity, intensity, intensity)
    img_path3 = "temp/sample_image_2.jpg"
    img3.save(img_path3)
    images.append(img3)
    image_paths.append(img_path3)
    
    # Sample 4: Stripes pattern
    img4 = Image.new('RGB', image_size)
    pixels = img4.load()
    stripe_width = 40
    for i in range(img4.width):
        for j in range(img4.height):
            if (i // stripe_width) % 2 == 0:
                pixels[i, j] = (200, 50, 50)
            else:
                pixels[i, j] = (50, 50, 200)
    img_path4 = "temp/sample_image_3.jpg"
    img4.save(img_path4)
    images.append(img4)
    image_paths.append(img_path4)
    
    return images, image_paths

def create_download_link(file_bytes, filename, link_text):
    """
    Creates an HTML download link for a file.
    
    Args:
        file_bytes: Bytes of the file to download
        filename: Name to save the file as
        link_text: Text to display for the link
        
    Returns:
        HTML markup for a download link
    """
    b64 = base64.b64encode(file_bytes).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{link_text}</a>'
    return href
