import cv2
import numpy as np

def normalize_fingerprint(image_path, output_path):
    # Step 1: Read the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return
    
    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 3: Enhance contrast using histogram equalization
    equalized = cv2.equalizeHist(gray)
    
    # Step 4: Denoise using Gaussian Blur
    denoised = cv2.GaussianBlur(equalized, (5, 5), 0)
    
    # Step 5: Apply Otsu's Binarization
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Step 6: Skeletonize the binary image
    skeleton = np.zeros_like(binary)
    temp_binary = binary.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    while True:
        eroded = cv2.erode(temp_binary, kernel)
        temp = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(temp_binary, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        temp_binary = eroded.copy()
        if cv2.countNonZero(temp_binary) == 0:
            break
    
    # Step 7: Invert the skeleton to get a white background with black fingerprint
    inverted_skeleton = cv2.bitwise_not(skeleton)  # Inverts black/white
    
    # Step 8: Save the processed image
    cv2.imwrite(output_path, inverted_skeleton)
    print(f"Normalized fingerprint saved at {output_path}")

# Example usage:
normalize_fingerprint(
    r'C:\Users\varsha\OneDrive\Desktop\project\samples\sai.bmp', 
    r'C:\Users\varsha\OneDrive\Desktop\project\samples\normalized_sai.bmp'
)