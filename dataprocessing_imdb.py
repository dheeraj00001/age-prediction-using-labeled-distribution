import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "/home/dheeraj/Documents/Age Detection/archive/imdb_train_new_1024.csv"
IMAGE_ROOT = "/home/dheeraj/Documents/Age Detection/archive/imdb-clean-1024/imdb-clean-1024"
OUTPUT_DIR = "/home/dheeraj/Documents/Age Detection/final_imdb_dataset"
SAVE_GRID_PATH = "/home/dheeraj/Documents/Age Detection/crop_analysis_grid.jpg"

def create_analysis_grid(num_samples=5):
    print("Loading original CSV for sampling...")
    df = pd.read_csv(CSV_PATH)
    
    # Shuffle the dataframe to get random images every time you run this
    df = df.sample(frac=1).reset_index(drop=True)
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))
    fig.suptitle("Cropping Analysis: Before & After", fontsize=16, fontweight='bold')
    
    count = 0
    for idx, row in df.iterrows():
        if count >= num_samples:
            break  # Stop once we have enough valid pairs
            
        orig_filename = row["filename"]
        new_filename = orig_filename.replace("/", "_")
        
        orig_path = os.path.join(IMAGE_ROOT, orig_filename)
        cropped_path = os.path.join(OUTPUT_DIR, new_filename)
        
        # Check if BOTH the original and your newly processed image exist
        if os.path.exists(orig_path) and os.path.exists(cropped_path):
            
            # Read images and convert BGR (OpenCV default) to RGB (Matplotlib default)
            orig_img = cv2.imread(orig_path)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            
            cropped_img = cv2.imread(cropped_path)
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            
            # Draw the bounding box on the original image so you can see the target area
            x_min, y_min = int(row["x_min"]), int(row["y_min"])
            x_max, y_max = int(row["x_max"]), int(row["y_max"])
            cv2.rectangle(orig_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 4)
            
            # Plot Original on the left
            axes[count, 0].imshow(orig_img)
            axes[count, 0].set_title("Original (with Bounding Box)")
            axes[count, 0].axis("off")
            
            # Plot Cropped on the right
            axes[count, 1].imshow(cropped_img)
            axes[count, 1].set_title(f"Cropped ({cropped_img.shape[1]}x{cropped_img.shape[0]})")
            axes[count, 1].axis("off")
            
            count += 1
            
    plt.tight_layout()
    plt.savefig(SAVE_GRID_PATH, dpi=150)
    print(f"\nSuccess! Grid saved to: {SAVE_GRID_PATH}")

if __name__ == "__main__":
    create_analysis_grid(num_samples=5)