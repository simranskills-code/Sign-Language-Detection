import os
import cv2
import shutil
import random
from tqdm import tqdm

SOURCE_DIR = r'IMAGES'
DEST_DIR = 'data'
CLASSES = ['A', 'B', 'C']

def get_class_id(class_name):
    return CLASSES.index(class_name)

def convert_bbox_to_yolo(image, bbox):
    h, w, _ = image.shape
    x, y, bw, bh = bbox

    x_center = (x + bw / 2) / w
    y_center = (y + bh / 2) / h
    bw /= w
    bh /= h

    return f"{x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}"

def create_yolo_dataset(split_ratio=0.8):
    os.makedirs(f'{DEST_DIR}/images/train', exist_ok=True)
    os.makedirs(f'{DEST_DIR}/images/val', exist_ok=True)
    os.makedirs(f'{DEST_DIR}/labels/train', exist_ok=True)
    os.makedirs(f'{DEST_DIR}/labels/val', exist_ok=True)

    for class_name in tqdm(CLASSES, desc="Processing classes"):
        class_dir = os.path.join(SOURCE_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"❌ Folder not found: {class_dir}")
            continue

        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not image_files:
            print(f"⚠️ No images found in {class_dir}")
            continue

        random.shuffle(image_files)
        split_idx = int(len(image_files) * split_ratio)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]

        for split, files in [('train', train_files), ('val', val_files)]:
            for img_file in files:
                img_path = os.path.join(class_dir, img_file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (320, 320))  # Resize for faster training

                if img is None:
                    print(f"❌ Skipping unreadable image: {img_path}")
                    continue

                h, w, _ = img.shape
                label = get_class_id(class_name)
                yolo_bbox = convert_bbox_to_yolo(img, (0, 0, w, h))

                # Save image
                dest_img_path = os.path.join(DEST_DIR, 'images', split, f"{class_name}_{img_file}")
                shutil.copy(img_path, dest_img_path)

                # Save label
                base_name = os.path.splitext(img_file)[0]
                dest_label_path = os.path.join(DEST_DIR, 'labels', split, f"{class_name}_{base_name}.txt")
                with open(dest_label_path, 'w') as f:
                    f.write(f"{label} {yolo_bbox}\n")

                print(f"✅ Wrote label: {dest_label_path}")

    print("🎉 Dataset conversion complete.")

if __name__ == "__main__":
    create_yolo_dataset()
