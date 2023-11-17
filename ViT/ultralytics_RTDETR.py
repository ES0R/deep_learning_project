from ultralytics import RTDETR
import pandas as pd

def save_results_to_csv(results, csv_filename):
    # Prepare data for DataFrame
    data = {}
    for attr in dir(results):
        if not attr.startswith('__') and not callable(getattr(results, attr)):
            data[attr] = getattr(results, attr)

    # Convert to DataFrame
    results_df = pd.DataFrame([data])

    # Save the DataFrame to a CSV file
    results_df.to_csv(csv_filename, index=False)
    print(f"Data converted to DataFrame and saved as '{csv_filename}'.")

def print_attribute_from_csv(csv_filename, attribute):
    # Read the CSV file into a DataFrame
    results_df = pd.read_csv(csv_filename)

    # Print the specified attribute
    if attribute in results_df.columns:
        print(f"{attribute}:")
        print(results_df[attribute].iloc[0])
    else:
        print(f"Attribute '{attribute}' not found in the CSV file.")

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR('rtdetr-l.pt')

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset
results = model.train(data='coco8.yaml', epochs=1, imgsz=640, iou=0.9)

# Save results to CSV
save_results_to_csv(results, "results_data.csv")

# Extract the actual data from the confusion matrix object
conf_matrix_obj = results.confusion_matrix
conf_matrix_data = conf_matrix_obj.matrix

# Convert to DataFrame and save to CSV
conf_matrix_df = pd.DataFrame(conf_matrix_data)
conf_matrix_df.to_csv("conf_matrix.csv", index=False)

# Reading and printing the confusion matrix from CSV
print_attribute_from_csv("conf_matrix.csv", "matrix")
