
from functools import wraps
from datetime import datetime
import time
from tqdm import tqdm

from functools import wraps
import time

def log_time(description="Activity"):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"\n{'=' * 55}\nStarted: {description}")
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            print(f"Completed: {description}. Duration: {end_time - start_time:.2f}s\n{'=' * 55}\n")
            return result
        return wrapper
    return decorator



def draw_boxes(image, bboxes, labels):
    for bbox, label in zip(bboxes, labels):
        # bbox is expected to be a list or a tuple like [x_min, y_min, width, height]

        x_min, y_min, width, height = bbox
        x_max, y_max = x_min + width, y_min + height
        
        print(label)
        print(x_min,x_max,y_min,y_max)
        # Draw rectangle
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)

        # Draw label
        #label_name = category_map.get(label, 'Unknown')
        #cv2.putText(image, label_name, (int(x_min), int(y_min - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image

    def plot_image_with_bboxes(image_tensor, bboxes, labels, save_path="sample_image_with_bboxes.png"):
    """
    Plots an image with its bounding boxes and labels.
    Args:
    - image_tensor (torch.Tensor): The image tensor.
    - bboxes (torch.Tensor): The bounding boxes.
    - labels (torch.Tensor): The labels for the bounding boxes.
    - save_path (str): Path to save the plotted image.
    """
    # Convert the image tensor to PIL for display
    image = transforms.ToPILImage()(image_tensor)

    # Create a matplotlib figure
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Add bounding boxes and labels to the image
    for bbox, label in zip(bboxes, labels):
        # Extract coordinates
        x, y, w, h = bbox
        # Create a rectangle patch
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        # Add the rectangle to the Axes
        ax.add_patch(rect)
        # Add label
        ax.text(x, y, str(label), color='white', fontsize=12, backgroundcolor='red')
        print(label)

    # Save the image with bounding boxes
    plt.savefig(save_path)

# Use this function with a sample from your train_dataset
# sample = train_dataset[512]  # Replace '4' with any valid index of your dataset
# image_tensor, bboxes, labels = sample
# plot_image_with_bboxes(image_tensor, bboxes, labels)
# print(f"1: {sample[0]} - shape {sample[0].shape}, 2: {sample[1]} - shape {sample[1].shape}, 3: {sample[2]} - shape {sample[2].shape}, {len(sample)}")

def plot_image_with_bboxes(image_tensor, bboxes, labels, save_path="sample_image_with_bboxes.png"):
    image = transforms.ToPILImage()(image_tensor)

    fig, ax = plt.subplots(1)
    ax.imshow(image)

    for bbox, label in zip(bboxes, labels):
        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, str(label), color='white', fontsize=12, backgroundcolor='red')
        print(label)

    plt.savefig(save_path)