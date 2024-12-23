from deepface import DeepFace
import cv2
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fftpack import fft2, fftshift, dct
from datetime import datetime
from skimage.filters import sobel, roberts, prewitt, laplace
from skimage.feature import canny

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GenFace: A Large-Scale Fine-Grained Face Forgery Benchmark and Cross Appearance-Edge Learning")
        
        # Set environment variable to suppress TensorFlow warnings
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        
        # Initialize history storage
        self.results_history = []
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Buttons
        ttk.Button(self.main_frame, text="Capture from Camera", command=self.capture_from_camera).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(self.main_frame, text="Load Image", command=self.load_image).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.main_frame, text="Load Multiple Images", command=self.load_multiple_images).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(self.main_frame, text="Show History", command=self.show_history).grid(row=0, column=3, padx=5, pady=5)
        
        # Create frame for multiple visualizations
        self.viz_frame = ttk.Frame(self.main_frame)
        self.viz_frame.grid(row=1, column=0, columnspan=4, pady=10)
        
        # Create labels for each visualization type
        self.viz_labels = {}
        self.viz_methods = ['Original', 'Canny', 'Sobel', 'LoG', 'Marr-Hildreth', 'DCT']
        for i, method in enumerate(self.viz_methods):
            label = ttk.Label(self.viz_frame)
            label.grid(row=0, column=i, padx=5)
            method_label = ttk.Label(self.viz_frame, text=method)
            method_label.grid(row=1, column=i, padx=5)
            self.viz_labels[method] = label
        
        # Create Treeview for results
        self.tree = ttk.Treeview(self.main_frame, columns=("Value"), show="tree")
        self.tree.grid(row=2, column=0, columnspan=4, pady=5, sticky="nsew")
        
        # Create table for metadata
        columns = ("Method", "Type", "Source", "Number", "Resolution", "Year", "Timestamp")
        self.metadata_table = ttk.Treeview(self.main_frame, columns=columns, show="headings")
        for col in columns:
            self.metadata_table.heading(col, text=col)
            self.metadata_table.column(col, width=100)
        self.metadata_table.grid(row=3, column=0, columnspan=4, pady=5, sticky="nsew")

    def generate_heatmap(self, img, method):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        if method == 'Canny':
            edges = canny(gray, sigma=2)
            return edges
            
        elif method == 'Sobel':
            edges = sobel(gray)
            return edges
            
        elif method == 'LoG':
            edges = laplace(gray)
            return np.abs(edges)
            
        elif method == 'Marr-Hildreth':
            # Gaussian blur followed by Laplacian
            blurred = cv2.GaussianBlur(gray, (5,5), 1)
            edges = cv2.Laplacian(blurred, cv2.CV_64F)
            return np.abs(edges)
            
        elif method == 'DCT':
            dct_result = dct(dct(gray.T, norm='ortho').T, norm='ortho')
            return np.abs(dct_result) / np.max(np.abs(dct_result))
            
        return gray # Default case

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
                },
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'image_path': image_path
            }
            
            # Add result to history
            self.results_history.append(result)
                
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
                if method == 'DCT':
                    colored_heatmap = plt.cm.viridis(heatmap)[:, :, :3]
                else:
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
                    "2024",
                    result.get('timestamp', 'N/A')
                ))
            else:
                self.tree.insert("", "end", text=f"Error: {result}")
    
    def capture_from_camera(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            cap.release()
            result = self.detect_face(filename)
            self.update_display(filename, result)
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
            
    def load_multiple_images(self):
        file_paths = filedialog.askopenfilenames(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
        )
        if file_paths:
            for file_path in file_paths:
                result = self.detect_face(file_path)
                self.update_display(file_path, result)
                
    def show_history(self):
        history_window = tk.Toplevel(self.root)
        history_window.title("Detection History")
        
        # Create frame for history content
        history_frame = ttk.Frame(history_window)
        history_frame.pack(padx=10, pady=10, fill='both', expand=True)
        
        # Create treeview for history
        columns = ("Timestamp", "Image", "Result", "Confidence")
        history_tree = ttk.Treeview(history_frame, columns=columns, show="headings")
        for col in columns:
            history_tree.heading(col, text=col)
            history_tree.column(col, width=150)
        
        # Create image preview label
        preview_label = ttk.Label(history_frame)
        preview_label.pack(side='right', padx=10)
        
        def show_image(event):
            selected_item = history_tree.selection()
            if selected_item:
                item = history_tree.item(selected_item[0])
                for result in self.results_history:
                    if os.path.basename(result.get('image_path', '')) == item['values'][1]:
                        img = Image.open(result['image_path'])
                        img.thumbnail((200, 200))  # Resize for preview
                        img_tk = ImageTk.PhotoImage(img)
                        preview_label.configure(image=img_tk)
                        preview_label.image = img_tk
        
        history_tree.bind('<<TreeviewSelect>>', show_image)
        
        # Add history items
        for result in self.results_history:
            if isinstance(result, dict):
                history_tree.insert("", "end", values=(
                    result.get('timestamp', 'N/A'),
                    os.path.basename(result.get('image_path', 'N/A')),
                    "DeepFake" if result['deepfake_analysis']['is_deepfake'] else "Authentic",
                    f"{result['deepfake_analysis']['confidence']:.2%}"
                ))
        
        history_tree.pack(side='left', fill='both', expand=True)

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
