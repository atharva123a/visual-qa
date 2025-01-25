import os
from kagglehub import dataset_download # type: ignore
from PIL import Image # type: ignore
import gradio as gr # type: ignore
import json
import cv2 # type: ignore
import random
import shutil
from sklearn.model_selection import train_test_split # type: ignore
from ultralytics import YOLO # type: ignore
from collections import Counter
import numpy as np

# Step 1: Download Dataset Function
def download_dataset(username, key, dataset_name, download_path="./datasets"):
    
    if os.path.exists(download_path) and os.listdir(download_path):
        return f"Dataset already exists in {download_path}. Skipping download."
    
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key

    try:
        dataset_path = dataset_download(dataset_name)
        os.makedirs(download_path, exist_ok=True)

        # Move dataset to the specified directory
        for item in os.listdir(dataset_path):
            source = os.path.join(dataset_path, item)
            target = os.path.join(download_path, item)
            if os.path.isdir(source):
                os.rename(source, target)
            else:
                os.replace(source, target)
        return f"Dataset downloaded successfully to {download_path}."
    except Exception as e:
        return f"Error downloading dataset: {e}"

def list_dataset_files(source_dir):
    try:
        files = os.listdir(source_dir)
        return "\n".join(files)  # Format the list as a string with one file per line
    except Exception as e:
        return f"Error listing files: {e}"

# Step 2: Map and Resize Classes Function
def map_and_resize_classes(source_dir, target_dir, class_mapping_str, new_width, new_height):
    """
    Maps and resizes images with random augmentation.
    """
    try:
        # Parse class mapping
        class_mapping = json.loads(class_mapping_str)
        
        # Create target directory
        os.makedirs(target_dir, exist_ok=True)
        
        processed_count = 0
        augmented_count = 0
        
        # Process each subdirectory
        for class_dir in os.listdir(source_dir):
            if class_dir in class_mapping:
                class_path = os.path.join(source_dir, class_dir)
                if os.path.isdir(class_path):
                    # Process each image in the class directory
                    for img_file in os.listdir(class_path):
                        if img_file.endswith(('.jpg', '.jpeg', '.png')):
                            # Read and resize image
                            img_path = os.path.join(class_path, img_file)
                            img = cv2.imread(img_path)
                            resized_img = cv2.resize(img, (new_width, new_height))
                            
                            # Save original image
                            base_name = f"{class_mapping[class_dir]}_{processed_count}"
                            save_path = os.path.join(target_dir, f"{base_name}.jpg")
                            cv2.imwrite(save_path, resized_img)
                            processed_count += 1
                            
                            # Random augmentation (40% chance)
                            if random.random() < 0.4:
                                augmented_img = resized_img.copy()
                                
                                # Higher probability for mosaic and mixup
                                aug_type = random.choices(
                                    ['mosaic', 'mixup', 'rotate', 'flip'],
                                    weights=[0.4, 0.3, 0.2, 0.1]
                                )[0]
                                
                                if aug_type == 'mosaic':
                                    # Create mosaic from 4 random images in same class
                                    class_images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                                    if len(class_images) >= 4:
                                        mosaic_size = new_height * 2
                                        mosaic = np.zeros((mosaic_size, mosaic_size, 3), dtype=np.uint8)
                                        
                                        for idx, m_img_file in enumerate(random.sample(class_images, 4)):
                                            m_img = cv2.imread(os.path.join(class_path, m_img_file))
                                            m_img = cv2.resize(m_img, (new_width, new_height))
                                            x = (idx % 2) * new_width
                                            y = (idx // 2) * new_height
                                            mosaic[y:y+new_height, x:x+new_width] = m_img
                                        
                                        augmented_img = cv2.resize(mosaic, (new_width, new_height))
                                
                                elif aug_type == 'mixup':
                                    # Mix with another random image from same class
                                    class_images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                                    if len(class_images) >= 2:
                                        mix_img_file = random.choice(class_images)
                                        mix_img = cv2.imread(os.path.join(class_path, mix_img_file))
                                        mix_img = cv2.resize(mix_img, (new_width, new_height))
                                        alpha = random.uniform(0.3, 0.7)
                                        augmented_img = cv2.addWeighted(augmented_img, alpha, mix_img, 1-alpha, 0)
                                
                                elif aug_type == 'rotate':
                                    # Random rotation (-15 to 15 degrees)
                                    angle = random.uniform(-15, 15)
                                    center = (new_width/2, new_height/2)
                                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                                    augmented_img = cv2.warpAffine(augmented_img, rotation_matrix, (new_width, new_height))
                                
                                elif aug_type == 'flip':
                                    # Horizontal flip
                                    augmented_img = cv2.flip(augmented_img, 1)
                                
                                # Save augmented image
                                aug_save_path = os.path.join(target_dir, f"{base_name}_aug_{aug_type}.jpg")
                                cv2.imwrite(aug_save_path, augmented_img)
                                augmented_count += 1
        
        return f"Processing complete:\n{processed_count} images processed\n{augmented_count} augmented images created"
                            
    except Exception as e:
        return f"Error during processing: {e}"

def generate_yolo_labels(image_dir, label_dir, class_mapping_str, object_size_fraction):
    """
    Generates YOLO bounding box annotations for resized images via Gradio.
    Args:
        image_dir (str): Path to the images (e.g., filtered_fruits/Training/images).
        label_dir (str): Path to save YOLO labels.
        class_mapping_str (str): JSON string mapping class names to IDs (e.g., '{"lemon": 0, "apple": 1}').
        object_size_fraction (float): Fraction of the image occupied by the object (0-1).
    Returns:
        str: Status message.
    """
    try:
        # Parse class mapping from JSON string
        class_mapping = json.loads(class_mapping_str)

        # Ensure label directory exists
        os.makedirs(label_dir, exist_ok=True)

        processed_files = []
        for img_file in os.listdir(image_dir):
            if not img_file.endswith(".jpg"):
                continue

            # Extract class name from the file prefix
            class_name = img_file.split("_")[0]
            class_id = class_mapping.get(class_name)
            if class_id is None:
                print(f"Class '{class_name}' not in class_mapping. Skipping {img_file}.")
                continue

            img_path = os.path.join(image_dir, img_file)
            # Read the image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read {img_path}. Skipping.")
                continue

            h, w, _ = img.shape  # Image dimensions (should be 224x224)
            if h != 224 or w != 224:
                print(f"Image {img_file} is not 224x224. Skipping.")
                continue

            # Calculate bounding box dimensions
            box_size = object_size_fraction  # Fraction of the image size occupied by the object
            box_w = box_size  # Width of bounding box as a fraction of the image width
            box_h = box_size  # Height of bounding box as a fraction of the image height
            x_center = 0.5  # Center horizontally
            y_center = 0.5  # Center vertically

            # Save label in YOLO format
            label_path = os.path.join(label_dir, img_file.replace(".jpg", ".txt"))
            try:
                with open(label_path, "w") as f:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")
                processed_files.append(f"Generated label for {img_file}: {label_path}")
            except Exception as e:
                return f"Error writing label for {img_file}: {e}"

        return f"Successfully generated YOLO labels. Processed files:\n" + "\n".join(processed_files)

    except Exception as e:
        return f"An error occurred: {e}"

def visualize_random_label(image_dir, label_dir, class_mapping_str):
    """
    Visualizes a random image with its YOLO format bounding box.
    
    Args:
        image_dir (str): Directory containing images
        label_dir (str): Directory containing labels
        class_mapping_str (str): JSON string of class mapping
    """
    try:
        # Parse class mapping and create class names list
        class_mapping = json.loads(class_mapping_str)
        max_class_id = max(class_mapping.values())
        class_names = [''] * (max_class_id + 1)
        for name, idx in class_mapping.items():
            class_names[idx] = name

        # Get list of image files
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        if not image_files:
            return None, "No images found in directory"

        # Select random image
        random_image = random.choice(image_files)
        image_path = os.path.join(image_dir, random_image)
        label_path = os.path.join(label_dir, random_image.replace('.jpg', '.txt'))

        if not os.path.exists(label_path):
            return None, f"Label file not found for {random_image}"

        # Load and process image
        image = cv2.imread(image_path)
        h, w, _ = image.shape

        # Read and draw bounding boxes
        with open(label_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            data = line.strip().split()
            class_id = int(data[0])
            x_center, y_center, box_width, box_height = map(float, data[1:])

            # Denormalize coordinates
            x_center *= w
            y_center *= h
            box_width *= w
            box_height *= h

            # Calculate box coordinates
            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)
            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            # Draw box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, class_names[class_id], (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert BGR to RGB for Gradio
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, f"Visualizing {random_image} with its bounding box"

    except Exception as e:
        return None, f"Error during visualization: {e}"

def split_train_val(image_dir, label_dir, output_base_dir, test_size):
    """
    Splits the dataset into training and validation sets.
    """
    try:
        # Define output directories
        train_image_dir = os.path.join(output_base_dir, "train/images")
        train_label_dir = os.path.join(output_base_dir, "train/labels")
        val_image_dir = os.path.join(output_base_dir, "val/images")
        val_label_dir = os.path.join(output_base_dir, "val/labels")

        # Create output directories
        os.makedirs(train_image_dir, exist_ok=True)
        os.makedirs(train_label_dir, exist_ok=True)
        os.makedirs(val_image_dir, exist_ok=True)
        os.makedirs(val_label_dir, exist_ok=True)

        # Get all images
        images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
        if not images:
            return "No images found in the source directory."

        # Split into train and validation sets
        train_images, val_images = train_test_split(images, test_size=float(test_size), random_state=42)

        # Move training images and labels
        for img_file in train_images:
            # Image file paths
            src_img = os.path.join(image_dir, img_file)
            dst_img = os.path.join(train_image_dir, img_file)
            shutil.copy(src_img, dst_img)

            # Label file paths
            label_file = img_file.replace(".jpg", ".txt")
            src_label = os.path.join(label_dir, label_file)
            dst_label = os.path.join(train_label_dir, label_file)
            if os.path.exists(src_label):
                shutil.copy(src_label, dst_label)

        # Move validation images and labels
        for img_file in val_images:
            # Image file paths
            src_img = os.path.join(image_dir, img_file)
            dst_img = os.path.join(val_image_dir, img_file)
            shutil.copy(src_img, dst_img)

            # Label file paths
            label_file = img_file.replace(".jpg", ".txt")
            src_label = os.path.join(label_dir, label_file)
            dst_label = os.path.join(val_label_dir, label_file)
            if os.path.exists(src_label):
                shutil.copy(src_label, dst_label)

        return f"Dataset split complete:\n" \
               f"Training images: {len(train_images)}\n" \
               f"Validation images: {len(val_images)}\n" \
               f"Files saved to: {output_base_dir}"

    except Exception as e:
        return f"Error during dataset split: {e}"

def create_data_yaml(base_path, class_mapping_str, yaml_path="data.yaml"):
    """
    Creates a YAML file for YOLO training configuration with absolute paths
    """
    try:
        # Convert base_path to absolute path
        abs_base_path = os.path.abspath(base_path)
        
        class_mapping = json.loads(class_mapping_str)
        class_names = [k for k, v in sorted(class_mapping.items(), key=lambda x: x[1])]
        
        yaml_content = f"""
path: {abs_base_path}  # Base path to the dataset
train: train/images  # Path to training images (relative to path)
val: val/images      # Path to validation images (relative to path)

nc: {len(class_names)}  # Number of classes
names: {class_names}  # Class names
"""
        with open(yaml_path, 'w') as f:
            f.write(yaml_content.strip())
        
        # Verify that directories exist
        train_path = os.path.join(abs_base_path, "train/images")
        val_path = os.path.join(abs_base_path, "val/images")
        
        if not os.path.exists(train_path):
            return f"Warning: Training images directory not found at {train_path}"
        if not os.path.exists(val_path):
            return f"Warning: Validation images directory not found at {val_path}"
            
        return f"Created YAML configuration at {yaml_path}\nBase path: {abs_base_path}\nVerified directories exist."
    except Exception as e:
        return f"Error creating YAML file: {e}"

def evaluate_model(model_path, image_path, confidence_threshold=0.7, imgsz=0):
    """
    Evaluates a YOLO model on a single image and returns the annotated image with detections.
    """
    try:
        # Load the model
        model = YOLO(model_path)
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            return None, "Error: Could not read image"
        
        # Run inference
        if(imgsz > 0):
            results = model(image, imgsz=imgsz, conf=confidence_threshold)
        else:
            results = model(image, conf=confidence_threshold)
        
        # Process results
        detections = []
        hash_map = {}
        
        for result in results:
            for box in result.boxes:
                coordinates = box.xyxy.tolist()[0]
                confidence = box.conf[0].item()
                class_id = int(box.cls[0].item())
                label = model.names[class_id].lower()
                
                x1, y1, x2, y2 = map(int, coordinates)
                
                # Draw bounding boxes and labels
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"{label} ({confidence:.2f})", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
                
                # Store detections
                existing_coordinates = hash_map.get(label, [])
                existing_coordinates.append([x1, y1, x2, y2])
                hash_map[label] = existing_coordinates
                detections.append((label, coordinates, confidence))
        
        # Convert BGR to RGB for Gradio
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create detection summary
        detection_summary = "\nDetection Summary:\n"
        for label, coords_list in hash_map.items():
            detection_summary += f"{label}: {len(coords_list)} instances\n"
        
        return image, detection_summary
        
    except Exception as e:
        return None, f"Error during evaluation: {e}"

def train_yolo_model(model_type, data_yaml_path, epochs, imgsz, batch_size, patience, lr0, lrf, momentum, project_name, exp_name, freeze_layers):
    """
    Trains a YOLO model with specified parameters.
    """
    try:
        # Verify data.yaml exists and is readable
        if not os.path.exists(data_yaml_path):
            return f"Error: data.yaml not found at {data_yaml_path}"
            
        # Load and verify data.yaml content
        with open(data_yaml_path, 'r') as f:
            yaml_content = f.read()
            print(f"Data YAML content:\n{yaml_content}")  # Debug print
        
        # Load the model
        model = YOLO(model_type)
        
        if freeze_layers:
            # Convert string input like "backbone", "neck" to list
            layers_to_freeze = [layer.strip() for layer in freeze_layers.split(",")]
            for layer in layers_to_freeze:
                if layer in ["backbone", "neck", "head"]:
                    model.model.requires_grad_(True)  # First unfreeze all
                    if layer == "backbone":
                        model.model.model[0:9].requires_grad_(False)  # Freeze backbone
                    elif layer == "neck":
                        model.model.model[9:15].requires_grad_(False)  # Freeze neck
                    elif layer == "head":
                        model.model.model[15:].requires_grad_(False)

        # Train the model with absolute path to data.yaml
        abs_yaml_path = os.path.abspath(data_yaml_path)
        if(lr0 and lrf):
            results = model.train(
                data=abs_yaml_path,
                epochs=int(epochs),
                imgsz=int(imgsz),
                batch=int(batch_size),
                patience=int(patience),  # Reduced patience
                lr0=float(lr0),  # Initial learning rate
                lrf=float(lrf),  # Final learning rate factor
                project=project_name,
                # momentum= int(momentum),
                name=exp_name,
            )
        else:
            results = model.train(
                data=abs_yaml_path,
                epochs=int(epochs),
                imgsz=int(imgsz),
                batch=int(batch_size),
                patience=int(patience),  # Reduced patience
                project=project_name,
                name=exp_name,
            )
      
        return f"Training completed!\nResults saved in {project_name}/{exp_name}"
    except Exception as e:
        return f"Error during training: {e}"

def check_class_balance(labels_dir, class_mapping_str):
    """
    Checks the distribution of classes in the labels directory.
    
    Args:
        labels_dir (str): Path to directory containing label files
        class_mapping_str (str): JSON string of class mapping
    Returns:
        str: Formatted string showing class distribution
    """
    try:
        # Parse class mapping
        class_mapping = json.loads(class_mapping_str)
        # Create reverse mapping (id -> name)
        id_to_name = {str(v): k for k, v in class_mapping.items()}
        
        # Count instances of each class
        class_counts = Counter()
        
        # Check if directory exists
        if not os.path.exists(labels_dir):
            return f"Error: Labels directory not found at {labels_dir}"
            
        # Count classes in all label files
        for label_file in os.listdir(labels_dir):
            if not label_file.endswith('.txt'):
                continue
                
            with open(os.path.join(labels_dir, label_file), 'r') as f:
                for line in f:
                    class_id = line.split()[0]
                    class_counts[class_id] += 1
        
        # Format results
        result = "Class Distribution:\n"
        total_instances = sum(class_counts.values())
        
        for class_id, count in sorted(class_counts.items()):
            class_name = id_to_name.get(class_id, f"Unknown class {class_id}")
            percentage = (count / total_instances) * 100 if total_instances > 0 else 0
            result += f"{class_name}: {count} instances ({percentage:.1f}%)\n"
            
        result += f"\nTotal instances: {total_instances}"
        return result
        
    except Exception as e:
        return f"Error checking class balance: {e}"

# Gradio Interface
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Fine tuning YOLOv8")

        gr.Markdown("# Dataset Downloader and Image Resizing Pipeline")
        # Inputs for dataset download
        username = gr.Textbox(label="Kaggle Username", placeholder="Enter your Kaggle username (e.g. atharvarsalokhe)", value="atharvarsalokhe")
        key = gr.Textbox(label="Kaggle API Key", placeholder="Enter your Kaggle API key", type="password",)
        dataset_name = gr.Textbox(label="Dataset Name", placeholder="Enter Kaggle dataset name (e.g., moltean/fruits)", value="moltean/fruits")
        download_path = gr.Textbox(label="Download Path", placeholder="Enter the path to save the dataset", value="./datasets")

        # Outputs for dataset download
        download_output = gr.Textbox(label="Dataset Download Status", lines=5, interactive=False)

        # Button to download dataset
        download_button = gr.Button("Download Dataset")

        # Input and button for listing files
        list_files_path = gr.Textbox(
            label="Directory to List", 
            placeholder="Enter directory path to list files from",
            value="./datasets/fruits-360_dataset_100x100/fruits-360/Training"  # Default value matching download_path
        )
        list_button = gr.Button("List Dataset Files")
        list_output = gr.Textbox(label="Dataset Files", lines=5, interactive=False)
        
        # Connect the button to the function and output
        list_button.click(
            fn=list_dataset_files,
            inputs=[list_files_path],
            outputs=[list_output]
        )

        # Inputs for class mapping and resizing
        source_dir = gr.Textbox(label="Source Directory", placeholder="Path to the source dataset", value="./datasets/fruits-360_dataset_100x100/fruits-360/Training")
        target_dir = gr.Textbox(label="Target Directory", placeholder="Path to save resized images", value="./filtered_fruits/training/images")
        class_mapping = gr.Textbox(
            label="Class Mapping (JSON Format)",
            placeholder='{"Lemon 1": "lemon", "Strawberry 1": "strawberry", "Kiwi 1": "kiwi"}',
            value='{"Lemon 1": "lemon", "Strawberry 1": "strawberry", "Kiwi 1": "kiwi"}'
        )
        new_width = gr.Number(label="Image Width", value=224)
        new_height = gr.Number(label="Image Height", value=224)

        # Outputs for resizing
        resize_output = gr.Textbox(label="Resizing and Mapping Status", lines=5, interactive=False)

        # Buttons
        run_resize = gr.Button("Run Mapping and Resizing")

        # Bind dataset download function
        download_button.click(
            fn=download_dataset,
            inputs=[username, key, dataset_name, download_path],
            outputs=download_output
        )

        # Bind resizing function
        run_resize.click(
            fn=map_and_resize_classes,
            inputs=[source_dir, target_dir, class_mapping, new_width, new_height],
            outputs=resize_output
        )

        gr.Markdown("# YOLO Label Generator")

        # Inputs for image directory, label directory, class mapping, and object size fraction
        image_dir = gr.Textbox(label="Image Directory", placeholder="Path to images (e.g., filtered_fruits/Training/images)", value="./filtered_fruits/training/images")
        label_dir = gr.Textbox(label="Label Directory", placeholder="Path to save YOLO labels (e.g., filtered_fruits/Training/labels)", value="./filtered_fruits/training/labels")
        class_mapping = gr.Textbox(
            label="Class Mapping (JSON Format)",
            placeholder='{"eggplant": 0, "kiwi": 1, "lemon": 2, "strawberry": 3, "tomato": 4}',
            value='{"eggplant": 0, "kiwi": 1, "lemon": 2, "strawberry": 3, "tomato": 4}'
        )
        object_size_fraction = gr.Number(label="Object Size Fraction", value=0.8)

        # Output
        output = gr.Textbox(label="Status Output", lines=10, interactive=False)

        # Button to trigger YOLO label generation
        generate_button = gr.Button("Generate YOLO Labels")

        # Bind the function to the button
        generate_button.click(
            fn=generate_yolo_labels,
            inputs=[image_dir, label_dir, class_mapping, object_size_fraction],
            outputs=output
        )

        gr.Markdown("# Label Visualization")
        
        # Add visualization section
        with gr.Row():
            vis_image = gr.Image(label="Visualization Output")
            vis_status = gr.Textbox(label="Visualization Status", lines=2)

        # Button to trigger visualization
        visualize_button = gr.Button("Visualize Random Label")

        # Bind the visualization function
        visualize_button.click(
            fn=visualize_random_label,
            inputs=[image_dir, label_dir, class_mapping],
            outputs=[vis_image, vis_status]
        )

        gr.Markdown("# Class Distribution Check")
    
        with gr.Row():
            check_labels_dir = gr.Textbox(
                label="Labels Directory",
                value="./filtered_fruits/training/labels",
                placeholder="Path to directory containing labels"
            )
            
        # Output for class distribution
        class_distribution = gr.Textbox(
            label="Class Distribution",
            lines=8,
            interactive=False
        )
        
        # Button to check class balance
        check_balance_button = gr.Button("Check Class Distribution")
        
        # Bind the check balance function
        check_balance_button.click(
            fn=check_class_balance,
            inputs=[check_labels_dir, class_mapping],
            outputs=class_distribution
        )

        gr.Markdown("# Dataset Split")
        
        # Inputs for dataset splitting
        with gr.Row():
            split_image_dir = gr.Textbox(
                label="Source Image Directory",
                value="./filtered_fruits/training/images",
                placeholder="Path to source images"
            )
            split_label_dir = gr.Textbox(
                label="Source Label Directory",
                value="./filtered_fruits/training/labels",
                placeholder="Path to source labels"
            )
            
        with gr.Row():
            output_base_dir = gr.Textbox(
                label="Output Base Directory",
                value="./training_dataset",
                placeholder="Base directory for train/val splits"
            )
            test_size = gr.Slider(
                label="Validation Set Size",
                minimum=0.1,
                maximum=0.5,
                value=0.2,
                step=0.1
            )

        # Output for split status
        split_status = gr.Textbox(label="Split Status", lines=4)

        # Button to trigger split
        split_button = gr.Button("Split Dataset")

        # Bind the split function
        split_button.click(
            fn=split_train_val,
            inputs=[split_image_dir, split_label_dir, output_base_dir, test_size],
            outputs=split_status
        )

        gr.Markdown("# YOLO Training Configuration")
        
        with gr.Row():
            model_type = gr.Textbox(
                label="YOLO Model Type",
                value="yolov8n.pt",
                placeholder="Enter YOLO model (e.g., yolov8n.pt, yolov8s.pt)",
            )
            yaml_path = gr.Textbox(
                label="Data YAML Path",
                value="./data.yaml",
                placeholder="Path to save data.yaml",
            )
            
        # Button to create YAML file
        yaml_status = gr.Textbox(label="YAML Creation Status", lines=2)
        create_yaml_button = gr.Button("Create Data YAML")

        # Bind the YAML creation function
        create_yaml_button.click(
            fn=create_data_yaml,
            inputs=[output_base_dir, class_mapping, yaml_path],
            outputs=yaml_status
        )

        gr.Markdown("# Model Evaluation")
        
        with gr.Row():
            eval_model_path = gr.Textbox(
                label="Model Path",
                value="yolov8n.pt",
                placeholder="Path to YOLO model (e.g., yolov8n.pt, runs/train/exp/weights/best.pt)"
            )
            eval_image_path = gr.Textbox(
                label="Test Image Path",
                placeholder="Path to image for testing",
                value="./fruits.png"
            )
            confidence_threshold = gr.Slider(
                label="Confidence Threshold",
                minimum=0.1,
                maximum=1.0,
                value=0.7,
                step=0.1
            )
        
        with gr.Row():
            eval_image_output = gr.Image(label="Detection Results")
            eval_text_output = gr.Textbox(label="Detection Summary", lines=5)

        with gr.Row():
            imgsz = gr.Number(
                label="Image Size",
                value=0,
                # minimum=128,
                # maximum=1536
            )
        # Button to trigger evaluation
        eval_button = gr.Button("Evaluate Model")
        
        # Bind the evaluation function
        eval_button.click(
            fn=evaluate_model,
            inputs=[eval_model_path, eval_image_path, confidence_threshold, imgsz],
            outputs=[eval_image_output, eval_text_output]
        )

        gr.Markdown("# YOLO Model Training")
        
        with gr.Row():
            train_model_type = gr.Textbox(
                label="Model Type",
                value="yolov8n.pt",
                placeholder="Enter YOLO model type (e.g., yolov8n.pt)"
            )
            train_data_yaml = gr.Textbox(
                label="Data YAML Path",
                value="./data.yaml",
                placeholder="Path to data.yaml file"
            )
        
        with gr.Row():
            train_epochs = gr.Number(
                label="Number of Epochs",
                value=50,  # Updated default
                minimum=1,
                maximum=300
            )
            train_batch_size = gr.Number(
                label="Batch Size",
                value=32,
                minimum=1,
                maximum=128
            )
        
        with gr.Row():
            train_lr0 = gr.Number(
                label="Initial Learning Rate",
                value=0.01,
                # minimum=0.0001,
                # maximum=0.1
            )
            train_lrf = gr.Number(
                label="Final Learning Rate Factor",
                value=0.1,
                # minimum=0.01,
                # maximum=1.0
            )
        
        with gr.Row():
            train_project = gr.Textbox(
                label="Project Name",
                value="fruit_detection",
                placeholder="Project folder name"
            )
            train_name = gr.Textbox(
                label="Experiment Name",
                value="yolo8n_baseline",
                placeholder="Experiment name"
            )
            freeze_layers = gr.Textbox(
                label="Freeze Layers",
                value="backbone,neck",  # Default to freezing backbone and neck
                placeholder="Enter layers to freeze (backbone,neck,head)"
            )
        
        # Add explanation for freeze layers
        gr.Markdown("""
        ### Layer Freezing Guide:
        - 0-2: Backbone layers
        - 3-5: Neck layers
        - 6+: Head layers
        Freezing backbone and neck layers (0-5) is recommended for fine-tuning.
        """)
        
        # Training status output
        train_status = gr.Textbox(
            label="Training Status",
            lines=3,
            interactive=False
        )
        
        with gr.Row():
            train_imgsz = gr.Number(
                label="Image Size",
                value=224,
                # minimum=128,
                # maximum=1536
            )

        with gr.Row():
            train_patience = gr.Number(
                label="Patience",
                value=5,
                # minimum=1,
                # maximum=10
            )
        
        with gr.Row():
            train_momentum = gr.Number(
                label="Momentum",
                value=0.9,
                # minimum=0.1,
                # maximum=1.0
            )

        # with gr.Row():
        #     train_freeze = gr.Number(
        #         label="Freeze",
        #         value=5,
        #         # minimum=0,
        #         # maximum=10
        #     )
        # Button to start training
        train_button = gr.Button("Start Training")
        
        # Bind the training function
        train_button.click(
            fn=train_yolo_model,
            inputs=[
                train_model_type,
                train_data_yaml,
                train_epochs,
                train_imgsz,
                train_batch_size,
                train_patience,
                train_lr0,
                train_lrf,
                train_momentum,
                train_project,
                train_name,
                freeze_layers
            ],
            outputs=train_status
        )

       

    demo.launch()

if __name__ == "__main__":
    gradio_interface()
