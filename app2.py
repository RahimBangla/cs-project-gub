import os
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Sobel Edge Extraction
def sobel_edge_extraction(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edge = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
    return edge

# Function to capture an image from the camera
def capture_image(image_path):
    cap = cv2.VideoCapture(0)  # Open the camera
    if not cap.isOpened():
        raise Exception("Could not open camera.")
    
    print("Press 'c' to capture a new image or 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image.")
            break
        
        cv2.imshow("Capture Image", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):  # Capture the image
            cv2.imwrite(image_path, frame)
            print(f"Image saved to {image_path}.")
            break
        elif key == ord('q'):  # Quit the capture
            print("Image capture canceled.")
            break

    cap.release()
    cv2.destroyAllWindows()

# Custom Dataset
class GenFaceDataset(Dataset):
    def __init__(self, image_dir, labels, transform=None):
        self.image_dir = image_dir
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name, label = self.labels[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Image not found: {img_path}. Capturing new image.")
            capture_image(img_path)
            time.sleep(1)  # Wait for 1 second to ensure the image is saved
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Image not found after capture: {img_path}")
        
        if self.transform:
            image = self.transform(image)
        
        edge_image = sobel_edge_extraction(image)
        return image, edge_image, label

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return x

# CAEL Model
class CAEL(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers):
        super().__init__()
        self.appearance_encoder = TransformerEncoder(input_dim, embed_dim, num_heads, num_layers)
        self.edge_encoder = TransformerEncoder(input_dim, embed_dim, num_heads, num_layers)
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.fc = nn.Linear(embed_dim, 2)  # Binary classification: Real or Fake

    def forward(self, appearance, edge):
        appearance_features = self.appearance_encoder(appearance)
        edge_features = self.edge_encoder(edge)
        combined_features, _ = self.cross_attention(appearance_features, edge_features, edge_features)
        output = self.fc(combined_features.mean(dim=0))
        return output

# Training Function
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for appearance, edge, labels in dataloader:
        appearance, edge, labels = appearance.to(device), edge.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(appearance, edge)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# Evaluation Function
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for appearance, edge, labels in dataloader:
            appearance, edge, labels = appearance.to(device), edge.to(device), labels.to(device)
            outputs = model(appearance, edge)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total

# Main Function
def main():
    # Hyperparameters
    input_dim = 224 * 224  # Assuming flattened 224x224 images
    embed_dim = 512
    num_heads = 8
    num_layers = 6
    batch_size = 16
    num_epochs = 10
    learning_rate = 0.001

    # Directories and Labels
    image_dir = "path_to_images"
    labels = [("image1.jpg", 0), ("image2.jpg", 1)]  # Replace with actual labels

    # Check if images exist, if not capture them
    for img_name, _ in labels:
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}. Capturing new image.")
            capture_image(img_path)
            time.sleep(1)  # Wait for 1 second to ensure the image is saved

    # Data Transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),  # Convert to PIL Image
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Dataset and DataLoader
    dataset = GenFaceDataset(image_dir, labels, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CAEL(input_dim, embed_dim, num_heads, num_layers).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training and Evaluation
    for epoch in range(num_epochs):
        try:
            train_loss = train(model, dataloader, optimizer, criterion, device)
            accuracy = evaluate(model, dataloader, device)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}, Accuracy: {accuracy:.4f}")
        except FileNotFoundError as e:
            print(e)

if __name__ == "__main__":
    main()
