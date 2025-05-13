import os
import shutil
import random

# --- 配置变量 ---
# 原始图片文件夹路径
image_dir = 'F:/新建文件夹/外包/dataset/images'
# 原始标注文件夹路径 (YOLO格式的.txt文件)
label_dir = 'F:/新建文件夹/外包/dataset/labels'
# 输出数据集的根目录，划分后的文件将保存在这里
output_dir = 'dataset'
# 训练集占总样本的比例 (例如 0.8 for 80% train, 20% val)
train_ratio = 0.8
# --- 结束配置 ---

def create_dirs(base_dir):
    """创建输出所需的目录结构"""
    os.makedirs(os.path.join(base_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'labels', 'val'), exist_ok=True)
    print(f"Created output directories in {base_dir}")

def get_file_pairs(image_dir, label_dir):
    """获取图片和标注文件的匹配对"""
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    label_files = [f for f in os.listdir(label_dir) if f.lower().endswith('.txt')]

    image_basenames = {os.path.splitext(f)[0] for f in image_files}
    label_basenames = {os.path.splitext(f)[0] for f in label_files}

    # 找到同时存在图片和标注的文件名（不含扩展名）
    common_basenames = list(image_basenames.intersection(label_basenames))

    if not common_basenames:
        print("Warning: No matching image and label files found based on filenames.")
        return []

    file_pairs = []
    for basename in common_basenames:
        img_ext = None
        # 找到匹配的图片文件（可能有不同的扩展名）
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
             if os.path.exists(os.path.join(image_dir, basename + ext)):
                 img_ext = ext
                 break

        if img_ext:
            image_path = os.path.join(image_dir, basename + img_ext)
            label_path = os.path.join(label_dir, basename + '.txt')
            file_pairs.append((image_path, label_path))
        else:
            print(f"Warning: Found label {basename}.txt but no matching image in {image_dir}.")


    print(f"Found {len(file_pairs)} matching image-label pairs.")
    return file_pairs

def split_data(file_pairs, train_ratio):
    """随机打乱并分割数据"""
    random.shuffle(file_pairs)
    train_size = int(len(file_pairs) * train_ratio)
    train_pairs = file_pairs[:train_size]
    val_pairs = file_pairs[train_size:]
    print(f"Splitting data: {len(train_pairs)} for training, {len(val_pairs)} for validation.")
    return train_pairs, val_pairs

def copy_files(pairs, output_base_dir, set_name):
    """复制文件到目标文件夹"""
    image_output_dir = os.path.join(output_base_dir, 'images', set_name)
    label_output_dir = os.path.join(output_base_dir, 'labels', set_name)
    copied_image_paths = []

    for img_path, lbl_path in pairs:
        img_name = os.path.basename(img_path)
        lbl_name = os.path.basename(lbl_path)

        dest_img_path = os.path.join(image_output_dir, img_name)
        dest_lbl_path = os.path.join(label_output_dir, lbl_name)

        shutil.copy(img_path, dest_img_path)
        shutil.copy(lbl_path, dest_lbl_path)

        # 记录相对路径，例如 'images/train/image.jpg'
        relative_image_path = os.path.join('images', set_name, img_name)
        copied_image_paths.append(relative_image_path)

    print(f"Copied {len(pairs)} pairs to {set_name} set.")
    return copied_image_paths

def write_txt_file(paths, output_file_path):
    """将图片路径写入txt文件"""
    with open(output_file_path, 'w') as f:
        for path in paths:
            f.write(f"{path}\n")
    print(f"Wrote {len(paths)} image paths to {output_file_path}")

if __name__ == "__main__":
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found at {image_dir}")
        exit()
    if not os.path.exists(label_dir):
        print(f"Error: Label directory not found at {label_dir}")
        exit()

    create_dirs(output_dir)

    file_pairs = get_file_pairs(image_dir, label_dir)

    if not file_pairs:
        print("No valid image-label pairs to process. Exiting.")
        exit()

    train_pairs, val_pairs = split_data(file_pairs, train_ratio)

    train_image_paths = copy_files(train_pairs, output_dir, 'train')
    val_image_paths = copy_files(val_pairs, output_dir, 'val')

    write_txt_file(train_image_paths, os.path.join(output_dir, 'train.txt'))
    write_txt_file(val_image_paths, os.path.join(output_dir, 'val.txt'))

    print("\nDataset splitting complete!")
    print(f"Output dataset is located in: {output_dir}")
    print("You can now configure your YOLOv5 data.yaml to point to this directory.")