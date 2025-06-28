import os
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
from config import DATASET_PATH, CLASS_LABELS, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, NUM_CLASSES, EPOCHS, LEARNING_RATE
from data_loader import load_data
from model_builder import build_cnn
from utils import predict_blood_group

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        logging.error(f"Dataset path {DATASET_PATH} does not exist. Please check the path.")
        exit(1)

    logging.info("Loading dataset...")
    images, labels = load_data(DATASET_PATH, CLASS_LABELS, IMG_HEIGHT, IMG_WIDTH)

    # Ensure dataset is not empty
    if len(images) == 0 or len(labels) == 0:
        logging.error("Dataset is empty. Ensure the dataset contains valid images and labels.")
        exit(1)

    logging.info("Normalizing image data...")
    images = images / 255.0  # Normalize images to [0, 1] range

    logging.info("Splitting dataset into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels)

    y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val = to_categorical(y_val, num_classes=NUM_CLASSES)

    # Compute class weights to handle class imbalance
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weights = dict(enumerate(class_weights))

    logging.info("Building CNN model...")
    cnn_model = build_cnn((IMG_HEIGHT, IMG_WIDTH, 3), NUM_CLASSES)

    cnn_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    logging.info("Training the model...")
    history = cnn_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=2
    )

    # Save training history
    metrics_dir = 'metrics'
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, 'training_metrics.npz')
    np.savez(metrics_path, history=history.history)
    logging.info(f"Training metrics saved at {metrics_path}")

    # Evaluate model on validation set
    logging.info("Evaluating the model...")
    val_predictions = np.argmax(cnn_model.predict(X_val), axis=1)
    val_true = np.argmax(y_val, axis=1)
    #logging.info("\n" + classification_report(val_true, val_predictions, target_names=CLASS_LABELS))
    logging.info("\n" + classification_report(val_true, val_predictions, target_names=CLASS_LABELS, zero_division=0))

    logging.info("\nConfusion Matrix:\n" + str(confusion_matrix(val_true, val_predictions)))

    # Save the model
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'blood_group_cnn_model.keras')  # Use .keras format
    cnn_model.save(model_path)
    logging.info(f"CNN model saved successfully at {model_path}!")

    # Test prediction
    test_image_path = r'C:\Users\varsha\OneDrive\Desktop\project\samples\sai.bmp'
    if os.path.exists(test_image_path):
        logging.info("Predicting blood group for the test image...")
        try:
            predicted_blood_group = predict_blood_group(
                cnn_model, test_image_path, CLASS_LABELS, (IMG_HEIGHT, IMG_WIDTH)
            )
            logging.info(f"The predicted blood group is: {predicted_blood_group}")
        except Exception as e:
            logging.error(f"An error occurred during prediction: {e}")
    else:
        logging.warning("Test image not found! Please provide a valid path.")