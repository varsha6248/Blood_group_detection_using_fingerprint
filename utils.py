
from tensorflow.keras.preprocessing import image
import numpy as np

def predict_blood_group(model, img_path, class_labels, img_size):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    # Optionally print probabilities for debugging
    print("Prediction probabilities:", predictions)
    return class_labels[predicted_class_idx]