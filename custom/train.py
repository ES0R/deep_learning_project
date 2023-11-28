import fiftyone as fo

# Define the target directory where you want to store the COCO dataset
target_dir = "/dtu/blackhole/19/147257/coco-2017/"  # Replace with your desired target directory

# Download the COCO dataset using FiftyOne
dataset = fo.zoo.load_zoo_dataset("coco-2017", split="train", dataset_dir=target_dir)

# Now, the COCO dataset will be downloaded and stored in the specified target directory
