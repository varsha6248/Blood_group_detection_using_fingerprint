import os
import numpy as np
from PIL import Image
from config import IMG_HEIGHT, IMG_WIDTH, DATASET_PATH, CLASS_LABELS

def load_image_bmp(image_path, target_size):
    try:
        img = Image.open(image_path).convert("RGB")  # Ensure 3 channels
        img = img.resize(target_size)
        return np.array(img) / 255.0  # Normalize to [0, 1]
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def load_data(dataset_path, class_labels, img_height, img_width):
    images, labels = [], []
    for class_index, class_label in enumerate(class_labels):
        class_dir = os.path.join(dataset_path, class_label)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist. Skipping.")
            continue

        for image_file in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_file)
            if os.path.isfile(image_path) and image_file.lower().endswith('.bmp'):
                image = load_image_bmp(image_path, (img_height, img_width))
                if image is not None:
                    images.append(image)
                    labels.append(class_index)

    if not images or not labels:
        raise ValueError("No valid images found in the dataset.")

    return np.array(images), np.array(labels)