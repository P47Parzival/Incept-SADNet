import os
import shutil

data_root = r'd:\DSADNet-main\data\raw'
files = [f for f in os.listdir(data_root) if f.endswith('.npy')]

print(f"Found {len(files)} .npy files in {data_root}")

for f in files:
    # f is like s01_x.npy
    subject = f.split('_')[0] # s01
    source = os.path.join(data_root, f)
    dest_dir = os.path.join(data_root, subject)
    
    # Check if subject dir exists
    if os.path.isdir(dest_dir):
        dest = os.path.join(dest_dir, f)
        print(f"Moving {source} -> {dest}")
        shutil.move(source, dest)
    else:
        print(f"Skipping {f}, directory {dest_dir} not found")

print("Move complete.")