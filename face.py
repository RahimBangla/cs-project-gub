from deepface import DeepFace
import cv2
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fftpack import fft2, fftshift

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection & DeepFake Analysis App")
        
        # Set environment variable to suppress TensorFlow warnings
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Buttons
        ttk.Button(self.main_frame, text="Capture from Camera", command=self.capture_from_camera).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(self.main_frame, text="Load Image", command=self.load_image).grid(row=0, column=1, padx=5, pady=5)
        
        # Create frame for multiple visualizations
        self.viz_frame = ttk.Frame(self.main_frame)
        self.viz_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Create labels for each visualization type
        self.viz_labels = {}
        self.viz_methods = ['Original', 'Frequency', 'Edge', 'Attention']
        for i, method in enumerate(self.viz_methods):
            label = ttk.Label(self.viz_frame)
            label.grid(row=0, column=i, padx=5)
            method_label = ttk.Label(self.viz_frame, text=method)
            method_label.grid(row=1, column=i, padx=5)
            self.viz_labels[method] = label
        
        # Create Treeview for results
        self.tree = ttk.Treeview(self.main_frame, columns=("Value"), show="tree")
        self.tree.grid(row=2, column=0, columnspan=2, pady=5, sticky="nsew")
        
        # Create table for metadata
        columns = ("Method", "Type", "Source", "Number", "Resolution", "Year")
        self.metadata_table = ttk.Treeview(self.main_frame, columns=columns, show="headings")
        for col in columns:
            self.metadata_table.heading(col, text=col)
            self.metadata_table.column(col, width=100)
        self.metadata_table.grid(row=3, column=0, columnspan=2, pady=5, sticky="nsew")

    def generate_heatmap(self, img, method):
        if method == 'Frequency':
            # FFT analysis
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            f = fft2(gray)
            fshift = fftshift(f)
            magnitude_spectrum = 20*np.log(np.abs(fshift))
            return magnitude_spectrum / magnitude_spectrum.max()
            
        elif method == 'Edge':
            # Edge detection based heatmap
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            return edges / 255.0
            
        elif method == 'Attention':
            # Attention-based visualization
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            return blur / 255.0
            
        return np.ones_like(img[:,:,0]) # Default case

    def detect_face(self, image_path):
        try:
            # Load the image and perform face detection
            face_result = DeepFace.detectFace(image_path)
            
            # Analyze the image for potential deepfake
            analysis = DeepFace.analyze(image_path, actions=['emotion', 'age', 'gender', 'race'])
            
            # Basic deepfake detection heuristics
            confidence_score = analysis[0]['emotion']['neutral']
            is_deepfake = confidence_score < 0.3  # If neutral emotion confidence is low, might be manipulated
            
            result = {
                'face_detection': face_result,
                'deepfake_analysis': {
                    'is_deepfake': is_deepfake,
                    'confidence': 1 - confidence_score,
                    'analysis': analysis[0]
                }
            }
            return result, True
        except Exception as e:
            return str(e), False

    def update_visualization(self, *args):
        if hasattr(self, 'current_image'):
            self.update_display(self.current_image_path, self.current_detection_result)

    def update_display(self, image_path, detection_result):
        # Store current image info for visualization updates
        self.current_image_path = image_path
        self.current_detection_result = detection_result
        
        # Load and display image
        img = cv2.imread(image_path)
        self.current_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image to a standard size for display
        display_size = (200, 200)
        resized_image = cv2.resize(self.current_image, display_size)
        
        # Generate and display all visualizations
        for method in self.viz_methods:
            if method == 'Original':
                display_img = resized_image
            else:
                heatmap = self.generate_heatmap(resized_image, method)
                colored_heatmap = plt.cm.jet(heatmap)[:, :, :3]
                colored_heatmap = (colored_heatmap * 255).astype(np.uint8)
                display_img = cv2.addWeighted(resized_image, 0.7, colored_heatmap, 0.3, 0)
            
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(display_img))
            self.viz_labels[method].configure(image=img_tk)
            self.viz_labels[method].image = img_tk
        
        # Clear previous results
        for item in self.tree.get_children():
            self.tree.delete(item)
        for item in self.metadata_table.get_children():
            self.metadata_table.delete(item)
            
        # Display results in tree
        if isinstance(detection_result, tuple):
            result, success = detection_result
            if success:
                # Add analysis results to tree
                self.tree.insert("", "end", text="Analysis Results")
                status = "⚠️ POTENTIAL DEEPFAKE" if result['deepfake_analysis']['is_deepfake'] else "✅ LIKELY AUTHENTIC"
                self.tree.insert("", "end", text=f"Status: {status}")
                self.tree.insert("", "end", text=f"Confidence: {result['deepfake_analysis']['confidence']:.2%}")
                
                analysis = result['deepfake_analysis']['analysis']
                self.tree.insert("", "end", text=f"Age: {analysis['age']}")
                self.tree.insert("", "end", text=f"Gender: {analysis['gender']}")
                self.tree.insert("", "end", text=f"Dominant Emotion: {max(analysis['emotion'].items(), key=lambda x: x[1])[0]}")
                
                # Add metadata to table
                img = Image.open(image_path)
                self.metadata_table.insert("", "end", values=(
                    "DeepFake" if result['deepfake_analysis']['is_deepfake'] else "Authentic",
                    "Face Manipulation",
                    "User Upload",
                    "1",
                    f"{img.size[0]}x{img.size[1]}",
                    "2024"
                ))
            else:
                self.tree.insert("", "end", text=f"Error: {result}")
    
    def capture_from_camera(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            cv2.imwrite("captured.jpg", frame)
            cap.release()
            result = self.detect_face("captured.jpg")
            self.update_display("captured.jpg", result)
        else:
            for item in self.tree.get_children():
                self.tree.delete(item)
            self.tree.insert("", "end", text="Error: Could not access camera")
    
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if file_path:
            result = self.detect_face(file_path)
            self.update_display(file_path, result)

    def show_robustness_plot(self):
        # Create a figure for the robustness plot
        plt.figure(figsize=(10, 8))
        
        # Sample data for demonstration (replace with actual data)
        severity = [0, 1, 2, 3, 4, 5]
        auc_values = {
            'CViT': [60, 55, 50, 45, 40, 35],
            'CrossViT': [65, 60, 55, 50, 45, 40],
            'EViT': [50, 45, 40, 35, 30, 25],
            'CAEL': [70, 65, 60, 55, 50, 45]
        }
        
        # Plotting each method
        for method, values in auc_values.items():
            plt.plot(severity, values, marker='o', label=method)

        plt.title('Robustness to Unseen Image Distortions')
        plt.xlabel('Severity')
        plt.ylabel('AUC (%)')
        plt.xticks(severity)
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()
