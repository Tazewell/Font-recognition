import cv2
import os
import numpy as np

def binarize_image(input_path, output_path, threshold=127, method=cv2.THRESH_BINARY):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图片: {input_path}")
        return
    
    resized_img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    
    _, binary_img = cv2.threshold(resized_img, threshold, 255, method)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    cv2.imwrite(output_path, binary_img)
    print(f"已处理并保存: {output_path}")

def process_images(threshold=127, method=cv2.THRESH_BINARY):
    input_dir = "data"
    output_dir = "binary_images"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not image_files:
        print("在images文件夹中没有找到图片文件")
        return
    
    # 处理每张图片
    for img_file in image_files:
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, img_file)
        binarize_image(input_path, output_path, threshold, method)

if __name__ == "__main__":
    threshold = 200 
    method = cv2.THRESH_BINARY  
    
    # cv2.THRESH_BINARY        # 标准二值化
    # cv2.THRESH_BINARY_INV    # 反转二值化
    # cv2.THRESH_TRUNC        # 截断阈值化
    # cv2.THRESH_TOZERO       # 低于阈值置零
    # cv2.THRESH_TOZERO_INV   # 高于阈值置零
    
    process_images(threshold, method)
    print("所有图片处理完成！") 