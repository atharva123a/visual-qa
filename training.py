import os
import shutil
import cv2
from sklearn.model_selection import train_test_split

def map_and_reorganize(source_dir, target_dir, class_mapping):
    print(f"Current working directory: {os.getcwd()}")
    print(f"Checking if source directory exists: {os.path.exists(source_dir)}")
    print(f"Full absolute path to source: {os.path.abspath(source_dir)}")

    """
    Maps specific class names to generic categories and reorganizes the dataset.
    Args:
        source_dir (str): Path to the original dataset (e.g., Fruits360/Training).
        target_dir (str): Path to the new dataset directory.
        class_mapping (dict): Mapping of specific class names to generic categories.
    """
    os.makedirs(target_dir, exist_ok=True)

    for specific_class, generic_class in class_mapping.items():
        source_class_dir = os.path.join(source_dir, specific_class)
        target_class_dir = os.path.join(target_dir, generic_class)
        os.makedirs(target_class_dir, exist_ok=True)

        if os.path.exists(source_class_dir):
            for img_file in os.listdir(source_class_dir):
                source_file = os.path.join(source_class_dir, img_file)
                target_file = os.path.join(target_class_dir, img_file)
                shutil.copy(source_file, target_file)
                print(f"Copied {source_file} to {target_file}")
        else:
            print(f"Class {specific_class} not found in {source_dir}")

# Example Usage
source_dir = "./fruits-360_dataset_100x100/fruits-360/Training"
target_dir = "./filtered_fruits/Training"
class_mapping = {
    "Apple 6": "apple",
    "Apple Braeburn 1": "apple",
    "Apple Red 1": "apple",
    "Apple Red 2": "apple",
    "Apple Red 3": "apple",
    "Banana 1": "banana",
    "Banana Red 1": "banana",
    "Lemon 1": "lemon",
    'Strawberry 1': 'strawberry',
}


def generate_yolo_labels(image_dir, label_dir, class_mapping, padding=0.2):
    """
    Generates YOLO bounding box annotations for centered objects.
    Args:
        image_dir (str): Path to the images (e.g., FilteredFruits/Training).
        label_dir (str): Path to save YOLO labels.
        class_mapping (dict): Mapping of class names to IDs (e.g., {"Apple": 0, "Banana": 1}).
        padding (float): Padding around the object as a fraction of image size.
    """
    os.makedirs(label_dir, exist_ok=True)

    for class_name, class_id in class_mapping.items():
        class_image_dir = os.path.join(image_dir, class_name)
        if not os.path.exists(class_image_dir):
            print(f"Skipping {class_name}, directory not found.")
            continue

        for img_file in os.listdir(class_image_dir):
            if not img_file.endswith(".jpg"):
                continue

            img_path = os.path.join(class_image_dir, img_file)
            # Get image dimensions
            img = cv2.imread(img_path)
            h, w, _ = img.shape

            # Calculate bounding box dimensions
            box_w = 1.0 - padding  # Full width minus padding
            box_h = 1.0 - padding  # Full height minus padding
            x_center = 0.5         # Centered horizontally
            y_center = 0.5         # Centered vertically

            # Normalize the bounding box
            label_path = os.path.join(label_dir, img_file.replace(".jpg", ".txt"))
            with open(label_path, "w") as f:
                f.write(f"{class_id} {x_center} {y_center} {box_w} {box_h}\n")
                print(f"Generated label for {img_file}: {label_path}")

# Example Usage
image_dir = "filtered_fruits/Training"
label_dir = "filtered_fruits/Training/labels"
class_mapping = {"apple": 0, "banana": 1, "lemon": 2, "strawberry": 3}  # Map class names to IDs

# generate_yolo_labels(image_dir, label_dir, class_mapping)

def visualize_bounding_boxes(image_path, label_path, class_names, output_path=None):
    """
    Visualizes bounding boxes on an image using YOLO format annotations.
    
    Args:
        image_path (str): Path to the image file.
        label_path (str): Path to the corresponding label file (.txt).
        class_names (list): List of class names (e.g., ['Apple', 'Banana']).
        output_path (str): Optional path to save the visualized image.
    """
    # Load the image
    image = cv2.imread(image_path)
    h, w, _ = image.shape

    # Check if the label file exists
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return

    # Read the label file
    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        # Parse the bounding box information
        data = line.strip().split()
        class_id = int(data[0])
        x_center, y_center, box_width, box_height = map(float, data[1:])

        # Denormalize the coordinates
        x_center *= w
        y_center *= h
        box_width *= w
        box_height *= h

        # Calculate the top-left and bottom-right coordinates
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        # Add the class name
        cv2.putText(image, class_names[class_id], (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the image
    cv2.imshow("Image with Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the image with bounding boxes (optional)
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Saved visualized image to: {output_path}")


image_path = "filtered_fruits/Training/apple/0_100.jpg"
label_path = "filtered_fruits/Training/labels/0_100.txt"
class_names = ['apple', 'banana', 'lemon', 'strawberry']  # Class names in the order of their IDs

# visualize_bounding_boxes(image_path, label_path, class_names)
def restructure_dataset(image_root, label_root, output_image_dir):
    """
    Restructures the dataset for YOLO by consolidating images and labels into single folders.
    Args:
        image_root (str): Path to the root directory containing class-specific image folders.
        label_root (str): Path to the directory containing label files.
        output_image_dir (str): Path to the output images folder.
        output_label_dir (str): Path to the output labels folder.
    """
    os.makedirs(output_image_dir, exist_ok=True)
    # os.makedirs(output_label_dir, exist_ok=True)

    for class_dir in os.listdir(image_root):
        class_path = os.path.join(image_root, class_dir)
        if not os.path.isdir(class_path):
            continue

        for img_file in os.listdir(class_path):
            if img_file.endswith(".jpg"):
                # Source image file
                src_img = os.path.join(class_path, img_file)

                # Corresponding label file
                # label_file = img_file.replace(".jpg", ".txt")
                # src_label = os.path.join(label_root, label_file)

                # Destination paths
                dst_img = os.path.join(output_image_dir, img_file)
                # dst_label = os.path.join(output_label_dir, label_file)

                # Copy files
                if(src_img != dst_img):
                    shutil.copy(src_img, dst_img)
                # if os.path.exists(src_label):
                #     # shutil.copy(src_label, dst_label)
                #     print(f"Copied {src_img} and {src_label} to {output_image_dir} and {output_label_dir}")
                # else:
                #     print(f"Label file not found for {img_file}, skipping.")

# Example Usage
image_root = "filtered_fruits/Training"
label_root = "filtered_fruits/Training/labels"
output_image_dir = "filtered_fruits/Training/images"
# output_label_dir = "filtered_fruits/Training/labels"

restructure_dataset(image_root, label_root, output_image_dir)

def rename_images_by_class(dataset_dir):
    """
    Renames image files in class-specific folders to include the class name as a prefix.
    Args:
        dataset_dir (str): Path to the dataset directory (e.g., filtered_fruits/Training).
    """
    for class_name in os.listdir(dataset_dir):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for img_file in os.listdir(class_dir):
            if img_file.endswith(".jpg"):  # Process only image files
                old_path = os.path.join(class_dir, img_file)
                new_filename = f"{class_name}_{img_file}"  # Add class name as a prefix
                new_path = os.path.join(class_dir, new_filename)

                os.rename(old_path, new_path)
                print(f"Renamed {old_path} to {new_path}")


def split_train_val(image_dir, label_dir, train_image_dir, train_label_dir, val_image_dir, val_label_dir, test_size=0.2):
    """
    Splits the dataset into training and validation sets.
    Args:
        image_dir (str): Directory containing all images.
        label_dir (str): Directory containing all labels.
        train_image_dir (str): Output directory for training images.
        train_label_dir (str): Output directory for training labels.
        val_image_dir (str): Output directory for validation images.
        val_label_dir (str): Output directory for validation labels.
        test_size (float): Fraction of data to use for validation.
    """
    # Ensure output directories exist
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    # Get all images
    images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    # Split into train and validation sets
    train_images, val_images = train_test_split(images, test_size=test_size, random_state=42)

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

    print(f"Dataset split complete: {len(train_images)} training images, {len(val_images)} validation images.")

# Example Usage
image_dir = "filtered_fruits/Training/all_images"
label_dir = "filtered_fruits/Training/all_labels"
train_image_dir = "datasets/train/images"
train_label_dir = "datasets/train/labels"
val_image_dir = "datasets/val/images"
val_label_dir = "datasets/val/labels"

split_train_val(image_dir, label_dir, train_image_dir, train_label_dir, val_image_dir, val_label_dir, test_size=0.2)