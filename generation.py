import os
import random
import numpy as np
from PIL import Image
import albumentations as A
import cv2

# 配置参数
config = {
    "dataset_path": "./111",
    "output_path": "./outputs",
    "canvas_size": (640, 640),
    "num_augmented": 20000,
    "min_objects": 3,
    "max_objects": 7,
    "scale_range": (0.3, 1.0),
    "background_colors": [(255, 255, 255), (230, 230, 230), (0, 0, 0)],
    "max_overlap_ratio": 0.15,  # 允许的最大重叠比例（0-1）
    "max_attempts": 100,        # 每个对象的最大尝试放置次数
    "iou_threshold": 0.05,      # 判定为重叠的IOU阈值
    "class_map_file": "classes.txt"
}

def calculate_iou(box1, box2):
    """计算两个边界框的交并比（IoU）"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # 计算交集区域
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    
    # 计算并集区域
    union_area = w1 * h1 + w2 * h2 - inter_area
    
    return inter_area / union_area if union_area != 0 else 0

def is_valid_position(new_box, existing_boxes):
    """检查新位置是否满足重叠条件"""
    for existing_box in existing_boxes:
        iou = calculate_iou(new_box, existing_box)
        if iou > config["iou_threshold"]:
            return False
    return True

def load_dataset(dataset_path):
    """加载数据集（无子文件夹版本）"""
    image_paths = []
    labels = []
    for file in os.listdir(dataset_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 从文件名中提取标签（去除扩展名）
            label = os.path.splitext(file)[0]
            image_paths.append(os.path.join(dataset_path, file))
            labels.append(label)
    return image_paths, labels

def create_augmentation_pipeline():
    """创建数据增强管道（与原代码相同）"""
    return A.Compose([
        A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=255),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.RandomGamma(p=0.3),
        A.CLAHE(p=0.2),
        A.RandomToneCurve(p=0.2),
        A.ISONoise(p=0.2),
    ])

def process_image(img_path, augmentor):
    """处理单张图片（与原代码相同）"""
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)
    augmented = augmentor(image=img_np)
    return Image.fromarray(augmented['image'])

def generate_dataset():
    # 创建输出目录
    os.makedirs(os.path.join(config["output_path"], "images"), exist_ok=True)
    os.makedirs(os.path.join(config["output_path"], "labels"), exist_ok=True)

    # 加载原始数据
    image_paths, labels = load_dataset(config["dataset_path"])
    unique_labels = sorted(list(set(labels)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}

    class_map_path = os.path.join(config["output_path"], config["class_map_file"])
    with open(class_map_path, "w", encoding="utf-8") as f:
        for label, label_id in label_to_id.items():
            f.write(f"{label} {label_id}\n")
    # 创建增强器
    augmentor = create_augmentation_pipeline()

    # 生成增强数据
    for idx in range(config["num_augmented"]):
        canvas = Image.new('RGB', config["canvas_size"], random.choice(config["background_colors"]))
        annotations = []
        existing_boxes = []
        
        num_objects = random.randint(config["min_objects"], config["max_objects"])
        selected = random.choices(range(len(image_paths)), k=num_objects)

        for img_idx in selected:
            success = False
            for _ in range(config["max_attempts"]):
                try:
                    img = process_image(image_paths[img_idx], augmentor)
                    label = labels[img_idx]
                    label_id = label_to_id[label]

                    # 随机缩放
                    scale = random.uniform(*config["scale_range"])
                    w, h = [int(dim * scale) for dim in img.size]
                    
                    # 计算可放置区域
                    max_x = config["canvas_size"][0] - w
                    max_y = config["canvas_size"][1] - h
                    if max_x < 0 or max_y < 0:
                        continue

                    # 生成随机位置
                    paste_x = random.randint(0, max_x)
                    paste_y = random.randint(0, max_y)
                    
                    # 检查重叠
                    new_box = (paste_x, paste_y, w, h)
                    if is_valid_position(new_box, existing_boxes):
                        img = img.resize((w, h))
                        canvas.paste(img, (paste_x, paste_y))
                        
                        # 计算标注
                        x_center = (paste_x + w/2) / config["canvas_size"][0]
                        y_center = (paste_y + h/2) / config["canvas_size"][1]
                        width = w / config["canvas_size"][0]
                        height = h / config["canvas_size"][1]
                        
                        annotations.append(
                            f"{label_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                        )
                        existing_boxes.append(new_box)
                        success = True
                        break
                        
                except Exception as e:
                    print(f"处理图片时出错：{e}")
                    continue
                    
            if not success:
                print(f"无法为第{idx}张图片找到合适位置")

        # 保存结果...（保持相同）
        if annotations:
            image_path = os.path.join(config["output_path"], "images", f"aug_{idx}.jpg")
            label_path = os.path.join(config["output_path"], "labels", f"aug_{idx}.txt")
            
            canvas.save(image_path)
            with open(label_path, "w") as f:
                f.write("\n".join(annotations))
if __name__ == "__main__":
    generate_dataset()
    print("数据集增强完成！")