import os
import cv2
import numpy as np

def process_images(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 读取图片
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # 创建分割掩码
                mask = np.zeros_like(img, dtype=np.uint8)
                mask[img == 0] = 2  # 灰度值为0的位置设置为2
                mask[img == 128] = 1  # 灰度值小于等于128的位置设置为1
                mask[img == 255] = 0  # 灰度值为255的位置设置为0

                # 保存到输出文件夹
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, mask)

    print(f"处理完成，分割掩码已保存到 {output_folder}")

# 输入文件夹路径
input_folder = r'G:\2025work\autoimageseg\datasets\refuge2\labels'
# 输出文件夹路径
output_folder = r'G:\2025work\autoimageseg\datasets\refuge2\masks'

# 调用函数
process_images(input_folder, output_folder)