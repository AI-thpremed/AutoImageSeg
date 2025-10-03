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


def get_image_names(folder_path):
    # 获取文件夹中所有文件
    files = os.listdir(folder_path)
    # 过滤出图片文件（假设图片文件后缀为常见的.jpg, .png, .jpeg, .bmp等）
    image_files = [f for f in files if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.gif', '.tiff'))]
    # 提取文件名（去掉后缀）
    image_names = [os.path.splitext(f)[0] for f in image_files]
    return set(image_names)


def compare_image_names(folder1, folder2):
    # 获取两个文件夹中的图片名字
    names1 = get_image_names(folder1)
    names2 = get_image_names(folder2)

    # 比较两个集合
    if names1 == names2:
        return True
    else:
        return False


# 示例本地文件夹路径
folder1 = r"G:\2025work\autoimageseg\datasets\refuge2\test\masks"
folder2 = r"G:\2025work\autoimageseg\datasets\refuge2\test\images"

if compare_image_names(folder1, folder2):
    print("两个文件夹中的图片文件名（去掉后缀后）完全一致")
else:
    print("两个文件夹中的图片文件名（去掉后缀后）不完全一致")