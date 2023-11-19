import os

# Function to generate a dynamic name based on the model and a sequence number
def generate_dynamic_name(model_path, base_dir='runs/detect'):
    model_name = os.path.basename(model_path).split('.')[0]

    # Create base_dir if it does not exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    existing_dirs = [d for d in os.listdir(base_dir) if d.startswith(model_name + '_')]
    
    sequence_numbers = [int(d.split('_')[-1]) for d in existing_dirs if d.split('_')[-1].isdigit()]
    next_number = max(sequence_numbers) + 1 if sequence_numbers else 1

    return f"{model_name}_{next_number}"
