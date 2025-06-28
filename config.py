import os

# Dataset Path
DATASET_PATH = r'C:\Users\varsha\OneDrive\Desktop\project\dataset_blood_group'

# Blood Group Labels
CLASS_LABELS = ['A-', 'A+', 'AB-', 'AB+', 'B-', 'B+', 'O-', 'O+']

# Image Parameters
IMG_HEIGHT, IMG_WIDTH = 64, 64  # Smaller image size for faster processing

# Training Parameters
BATCH_SIZE = 32
NUM_CLASSES = len(CLASS_LABELS)
EPOCHS = 20# Use early stopping to prevent over-training
LEARNING_RATE = 0.001

# Model Directory
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# TensorFlow Settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info logs