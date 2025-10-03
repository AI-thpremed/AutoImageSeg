# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import numpy as np

# 源文件夹路径
folder_path = r'G:\2025work\autoimageseg\datasets\PlantSegDatasets\PlantSegmentationDatasets\vinecuttings\masks'
# 新文件夹路径
new_folder_path = r'G:\2025work\autoimageseg\datasets\PlantSegDatasets\PlantSegmentationDatasets\vinecuttings\masks_processed'

# 创建新文件夹（如果不存在）
os.makedirs(new_folder_path, exist_ok=True)

# 遍历源文件夹中的所有文件
for filename in os.listdir(folder_path):
    # 构造完整的文件路径
    file_path = os.path.join(folder_path, filename)

    # 读取图像（灰度模式）
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    if image is not None:
        # 将灰度值255的像素改为1
        processed_image = np.where(image == 255, 1, image)

        # 构造新的文件路径
        new_file_path = os.path.join(new_folder_path, filename)

        # 保存处理后的图像
        cv2.imwrite(new_file_path, processed_image)
        print(f"Processed and saved: {new_file_path}")
    else:
        print(f"Failed to read image: {file_path}")