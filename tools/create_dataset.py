import os
import numpy as np
from PIL import Image, ImageDraw
import random

# Define the folder structure
output_folder = "./data/raw"
shapes = ['circle', 'rectangle', 'triangle']
num_images = 100  # Adjust the number of images per shape

# Ensure the directory exists
os.makedirs(output_folder, exist_ok=True)

# Function to generate a random image of a shape (outline only)
def create_shape_image(shape, size=(128, 128)):
    img = Image.new('RGB', size, (255, 255, 255))  # White background
    draw = ImageDraw.Draw(img)
    w, h = size
    
    if shape == 'circle':
        radius = min(w, h) // 4
        x0 = random.randint(radius, w - radius)
        y0 = random.randint(radius, h - radius)
        draw.ellipse((x0-radius, y0-radius, x0+radius, y0+radius), outline=(0, 0, 0), width=2)  # Outline only
    
    elif shape == 'rectangle':
        x0 = random.randint(0, w // 2)
        y0 = random.randint(0, h // 2)
        x1 = random.randint(w // 2, w)
        y1 = random.randint(h // 2, h)
        draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 0), width=2)  # Outline only
    
    elif shape == 'triangle':
        x1 = random.randint(0, w)
        y1 = random.randint(0, h)
        x2 = random.randint(0, w)
        y2 = random.randint(0, h)
        x3 = random.randint(0, w)
        y3 = random.randint(0, h)
        draw.polygon([(x1, y1), (x2, y2), (x3, y3)], outline=(0, 0, 0), width=2)  # Outline only
    
    return img

# Function to generate and save dataset
def generate_dataset():
    for shape in shapes:
        for i in range(num_images):  # Generate images from 0 to 9999
            img = create_shape_image(shape)
            img_path = os.path.join(output_folder, f"{shape}{str(i).zfill(4)}.png")  # Format number with leading zeros
            img.save(img_path)
            print(f"Saved {img_path}")

# Generate the dataset
generate_dataset()