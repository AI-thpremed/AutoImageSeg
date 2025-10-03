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

def count_gray_levels(folder_path):
    gray_levels = set()  # 用于存储所有图片中出现的灰度值

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            # 读取图片
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # 获取图片中的所有灰度值
                unique_levels = np.unique(img)
                gray_levels.update(unique_levels)

    return gray_levels


folder_path = r'G:\2025work\autoimageseg\datasets\refuge2\masks'
gray_levels = count_gray_levels(folder_path)
print(f"灰度值数量: {len(gray_levels)}")
print(f"灰度值: {gray_levels}")