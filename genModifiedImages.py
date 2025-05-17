import os
import sys
import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFilter

def add_noise(image):
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    return cv2.add(image, noise)

def rotate_image(image):
    angle = random.uniform(-10, 10)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

def add_lines(image):
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    for _ in range(random.randint(1, 5)):
        x1 = random.randint(0, image.shape[1])
        y1 = random.randint(0, image.shape[0])
        x2 = random.randint(0, image.shape[1])
        y2 = random.randint(0, image.shape[0])
        draw.line((x1, y1, x2, y2), fill=(random.randint(100, 200)), width=1)
    return np.array(pil_img)

def add_dots(image):
    pil_img = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_img)
    for _ in range(random.randint(10, 30)):
        x = random.randint(0, image.shape[1])
        y = random.randint(0, image.shape[0])
        draw.ellipse((x, y, x+2, y+2), fill=(random.randint(100, 200)))
    return np.array(pil_img)

def deform_image(image):
    pts1 = np.float32([[5,5], [image.shape[1]-5, 5], [5, image.shape[0]-5]])
    dx = random.randint(-10, 10)
    dy = random.randint(-10, 10)
    pts2 = np.float32([[5+dx,5+dy], [image.shape[1]-5, 5+dy], [5+dx, image.shape[0]-5]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT)

def apply_random_transform(image):
    funcs = [add_noise, rotate_image, add_lines, add_dots, deform_image]
    random.shuffle(funcs)
    transformed = image.copy()
    for func in funcs[:random.randint(2, 4)]:
        transformed = func(transformed)
    return transformed

def main(input_path, digit_label, num_images):
    os.makedirs(f"./Dataset/{digit_label}", exist_ok=True)
    
    # Load and convert to grayscale if needed
    orig_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if orig_img is None:
        print("Failed to load image. Please check the path.")
        return

    # Save original image
    base_name = os.path.basename(input_path)
    save_path = f"./Dataset/{digit_label}/{base_name}"
    cv2.imwrite(save_path, orig_img)

    # Generate modified images
    for i in range(1, num_images + 1):
        transformed = apply_random_transform(orig_img)
        out_name = f"./Dataset/{digit_label}/{digit_label}_{i}.jpg"
        cv2.imwrite(out_name, transformed)

    print(f"Generated {num_images} images in ./Dataset/{digit_label}/")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 genModifiedImages.py <image_path> <digit_label> <num_images>")
        sys.exit(1)
    
    input_image = sys.argv[1]
    digit = sys.argv[2]
    count = int(sys.argv[3])
    main(input_image, digit, count)
