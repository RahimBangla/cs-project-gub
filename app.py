import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage import filters
from scipy.fftpack import fft2, fftshift

# Load Pretrained Model (ResNet-18 for simplicity)
model = resnet18(pretrained=True)
model.eval()

# Grad-CAM Implementation
def grad_cam(model, img_tensor, target_layer):
    gradients = []
    activations = []
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])
    
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)
    
    output = model(img_tensor.unsqueeze(0))
    target_class = output.argmax()
    model.zero_grad()
    output[0, target_class].backward()
    
    gradients = gradients[0].detach().numpy()[0]
    activations = activations[0].detach().numpy()[0]
    
    weights = np.mean(gradients, axis=(1, 2))
    cam = np.sum(weights[:, None, None] * activations, axis=0)
    cam = np.maximum(cam, 0)  # ReLU
    cam = cv2.resize(cam, (img_tensor.shape[1], img_tensor.shape[2]))
    cam = (cam - cam.min()) / (cam.max() - cam.min())  # Normalize
    return cam

# Edge Detection
def apply_edge_detection(img, method="Sobel"):
    if method == "Sobel":
        return filters.sobel(img)
    elif method == "Canny":
        return cv2.Canny((img * 255).astype(np.uint8), 100, 200)
    elif method == "MarrHildreth":
        return filters.gaussian(img, sigma=2) - filters.gaussian(img, sigma=1)
    else:
        raise ValueError("Unknown method")

# Fourier Frequency Analysis
def frequency_analysis(img):
    f_transform = fftshift(fft2(img))
    magnitude_spectrum = 20 * np.log(np.abs(f_transform) + 1)
    return magnitude_spectrum

# Visualization
def visualize_results(img, cam, edge_img, frequency_spectrum):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    
    plt.subplot(2, 2, 2)
    plt.title("Grad-CAM Heatmap")
    plt.imshow(cam, cmap='jet')
    plt.colorbar()
    
    plt.subplot(2, 2, 3)
    plt.title("Edge Detection")
    plt.imshow(edge_img, cmap='gray')
    
    plt.subplot(2, 2, 4)
    plt.title("Frequency Spectrum")
    plt.imshow(frequency_spectrum, cmap='gray')
    
    plt.tight_layout()
    plt.show()

# Main Function
if __name__ == "__main__":
    # Initialize camera
    cap = cv2.VideoCapture(0)  # 0 for default camera, or provide a camera index
    
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        exit()
    
    print("Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break
        
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.resize(gray_frame, (224, 224)) / 255.0  # Normalize to [0, 1]
        gray_frame = np.stack((gray_frame,) * 3, axis=-1)  # Convert to 3 channels
        img_tensor = torch.tensor(gray_frame).float().permute(2, 0, 1)  # Convert to tensor and permute dimensions
        
        # Grad-CAM
        target_layer = model.layer4[0].conv2
        cam = grad_cam(model, img_tensor, target_layer)
        
        # Edge Detection
        edge_img = apply_edge_detection(gray_frame[:, :, 0], method="Sobel")  # Use only one channel for edge detection
        
        # Frequency Analysis
        frequency_spectrum = frequency_analysis(gray_frame[:, :, 0])  # Use only one channel for frequency analysis
        
        # Display results
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Edge Detection", edge_img)
        cv2.imshow("Frequency Spectrum", (frequency_spectrum / frequency_spectrum.max() * 255).astype(np.uint8))
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Visualize detailed results (only for manual inspection)
        visualize_results(gray_frame[:, :, 0], cam, edge_img, frequency_spectrum)
    
    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()
