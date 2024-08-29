





import os
import random
import shutil
import numpy as np

# Define paths
data_folder = "/path/to/the/dataset"
output_folder = "isic_alpha_1.0"

# Ensure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List subfolders (classes)
subfolders = sorted(os.listdir(data_folder))

# Assign each subfolder to a class label
class_labels = {subfolder: i for i, subfolder in enumerate(subfolders)}

# Function to copy files from source to destination
def copy_files(source, destination):
    if not os.path.exists(destination):
        os.makedirs(destination)
    for file in os.listdir(source):
        shutil.copy(os.path.join(source, file), destination)

# Parameters
num_clients = 4
num_shards_per_class = 200
eta = 1.0
num_samples_per_shard = 5

# Generate non-IID local datasets for each client
for i in range(num_clients):
    client_folder = os.path.join(output_folder, f"client_{i+1}")
    if not os.path.exists(client_folder):
        os.makedirs(client_folder)
    
    for subfolder in subfolders:
        class_folder = os.path.join(data_folder, subfolder)
        class_files = os.listdir(class_folder)
        
        # Shuffle the files to ensure randomness
        random.shuffle(class_files)
        
        # Determine the number of shards for this class
        num_shards = min(num_shards_per_class, len(class_files))
        
        # Create local dataset directory for this class
        local_dataset_folder = os.path.join(client_folder, subfolder)
        os.makedirs(local_dataset_folder)
        
        # Generate local dataset for this class
        for shard in range(num_shards):
            if random.random() < eta:
                # Draw samples from the current class shard with probability eta
                shard_samples = random.sample(class_files, num_samples_per_shard)
            else:
                # Draw samples from other class shards based on a normal distribution
                other_class_files = np.random.choice(class_files, size=num_samples_per_shard, replace=False)
                shard_samples = random.choices(other_class_files, k=num_samples_per_shard)
            
            # Copy the selected files to the local dataset folder
            for sample in shard_samples:
                shutil.copy(os.path.join(class_folder, sample), local_dataset_folder)







