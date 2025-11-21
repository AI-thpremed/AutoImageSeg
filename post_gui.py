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

import json
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QFileDialog, QMessageBox
)
from ui_post_window import Ui_PostForm    # 由 pyside6-uic post.ui -o ui_post.py
import cv2
import numpy as np
from PySide6.QtWidgets import QMessageBox

import datetime

from skimage import measure
from PySide6.QtWidgets import QProgressDialog
from PySide6.QtCore import Qt
from config_manager import ConfigManager
from PySide6.QtCore import Qt, QCoreApplication


class PostWindow(QWidget, Ui_PostForm):
    def __init__(self,config: ConfigManager):
        super().__init__()
        self.setupUi(self)
        self.config = config


        self.btn_mask.clicked.connect(lambda: self._select_dir(self.line_mask))
        self.btn_json.clicked.connect(lambda: self._select_file(self.line_json))

        self.btn_labelme.clicked.connect(self.on_mask_to_labelme)
        self.btn_color.clicked.connect(self.on_mask_to_color)

    def _select_dir(self, line):
        folder = QFileDialog.getExistingDirectory(self, "Select Directory")
        if folder:
            line.setText(folder)

    def _select_file(self, line):
        file, _ = QFileDialog.getOpenFileName(self, "Select File", filter="JSON (*.json)")
        if file:
            line.setText(file)


    def on_mask_to_labelme(self):


        progress = QProgressDialog("Generating LabelMe JSON, please wait...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)  # 不可取消
        progress.setAutoClose(True)
        progress.show()

        from PySide6.QtCore import QCoreApplication
        QCoreApplication.processEvents()

        mask_dir = Path(self.line_mask.text().strip())
        json_path = Path(self.line_json.text().strip())

        suffix=self.combo_suffix.currentText()


        if not mask_dir.is_dir():
            QMessageBox.warning(self, "Path Error", "The mask directory does not exist")
            return
        if not json_path.is_file():
            QMessageBox.warning(self, "Path Error", "label_mapping.json does not exist")
            return

        # 2. Read the mapping and check its validity
        with open(json_path, encoding="utf-8") as f:
            try:
                label2id = json.load(f)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Unable to parse label_mapping.json: {e}")
                return


        if not isinstance(label2id, dict):
            QMessageBox.warning(self, "Error", "label_mapping.json must be a dictionary")
            return

        if "_background_" not in label2id or label2id["_background_"] != 0:
            QMessageBox.warning(self, "Error", "The JSON file must contain '_background_': 0")
            return

        lesion_ids = [v for k, v in label2id.items() if k != "_background_"]
        if len(lesion_ids) > 99:
            QMessageBox.warning(self, "Error", "The number of lesions must not exceed 99")
            return

        try:
            if sorted(lesion_ids) != list(range(1, len(lesion_ids) + 1)):
                QMessageBox.warning(self, "Error", "Lesion numbers must be consecutive integers starting from 1")
                return
        except TypeError:
            QMessageBox.warning(self, "Error", "Lesion values must be integers")
            return

        id2label = {v: k for k, v in label2id.items()}
        if not id2label:
            QMessageBox.warning(self, "Error", "label_mapping.json is empty")
            return


        root_results = Path(__file__).parent / 'results'
        root_results.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        res_dir = root_results / f'post_project_{timestamp}'
        res_dir.mkdir(exist_ok=True)
        out_dir = res_dir / "labelme_jsons"
        out_dir.mkdir(exist_ok=True)

        min_area = self.config["labelme_conversion"].get("min_area", 10.0)
        dp_tolerance = self.config["labelme_conversion"].get("dp_tolerance", 1.5)
        threshold_image = self.config["labelme_conversion"].get("threshold_image",0)


        mask_files = [p for p in mask_dir.iterdir()
                      if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}]
        if not mask_files:
            QMessageBox.warning(self, "Warning", "Only support .png, .jpg, .jpeg, .tif, .tiff files.")
            return

        # mask_files = list(mask_dir.glob("*.png")) + list(mask_dir.glob("*.jpg")) + list(mask_dir.glob("*.jpeg"))
        # if not mask_files:
        #     QMessageBox.warning(self, "Warning", "No PNG, JPG, or JPEG files found in the mask directory")
        #     return

        post_count = self.config["restriction"].get("post_count", 1000)



        if post_count > 0 and len(mask_files) > post_count:
            QMessageBox.warning(self, "Error",
                                f"The number of mask files ({len(mask_files)}) exceeds the maximum allowed ({post_count})")
            return


        if threshold_image > 0:
            for mask_file in mask_files:
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    continue
                total_targets = 0
                for gray_id in id2label.keys():
                    if gray_id == 0:
                        continue
                    binary = (mask == gray_id).astype(np.uint8)
                    if not np.any(binary):
                        continue
                    contours = measure.find_contours(binary, 0.5)
                    total_targets += len(contours)
                if total_targets > threshold_image:
                    QMessageBox.warning(
                        self,
                        "Prediction Quality Alert",
                        f"Mask '{mask_file.name}' contains {total_targets} objects, "
                        f"exceeding the allowed threshold ({threshold_image}).\n"
                        f"Conversion aborted to avoid low-quality results.\n"
                        f"Please perform mask to color image conversion to check the segmentation quality results."
                    )
                    progress.close()
                    return


        invalid_masks = []
        for mask_file in mask_files:
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            h, w = mask.shape


            unique_vals = set(np.unique(mask))
            valid_vals = set(id2label.keys())
            if not unique_vals.issubset(valid_vals):
                invalid_masks.append(mask_file.name)
                continue

            shapes = []
            for gray_id, label in id2label.items():
                if gray_id == 0:  #  0 =Background
                    continue
                binary = (mask == gray_id).astype(np.uint8)
                if not np.any(binary):
                    continue

                contours = measure.find_contours(binary, 0.5)
                for cnt in contours:

                    cnt = measure.approximate_polygon(cnt, tolerance=dp_tolerance)
                    points = [[float(x), float(y)] for y, x in cnt]

                    if len(points) < 3:
                        continue


                    polygon = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                    area = cv2.contourArea(polygon)

                    if area < min_area:
                        continue

                    shapes.append({
                        "label": label,
                        "points": points,
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {}
                    })

            # build LabelMe JSON
            labelme_json = {
                "version": "4.5.6",
                "flags": {},
                "shapes": shapes,
                "imagePath": mask_file.with_suffix(suffix).name,
                "imageData": None,
                "imageHeight": mask.shape[0],
                "imageWidth": mask.shape[1]
            }

            with open(out_dir / f"{mask_file.stem}.json", "w", encoding="utf-8") as f:
                json.dump(labelme_json, f, ensure_ascii=False, indent=2)


        progress.close()

        if invalid_masks:
            QMessageBox.warning(
                self, "Warning",
                f"The following files contain undefined labels and have been skipped:\n" + "\n".join(
                    invalid_masks[:10]) +
                (f"\n...and {len(invalid_masks) - 10} more files" if len(invalid_masks) > 10 else "")
            )

        QMessageBox.information(
            self, "Completed",
            f"Successfully processed {len(mask_files) - len(invalid_masks)}/{len(mask_files)} files\n"
            f"Results have been saved to:\n{out_dir}"
        )

    def on_mask_to_color(self):
        progress = QProgressDialog("Generating color maps, please wait...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setCancelButton(None)
        progress.setAutoClose(True)
        progress.show()

        QCoreApplication.processEvents()

        mask_dir = Path(self.line_mask.text().strip())


        post_count = self.config["restriction"].get("post_count", 1000)


        if not mask_dir.is_dir():
            QMessageBox.warning(self, "Error", "The mask directory does not exist")
            return

        mask_files = [p for p in mask_dir.iterdir()
                      if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}]
        if not mask_files:
            QMessageBox.warning(self, "Warning", "Only support .png, .jpg, .jpeg, .tif, .tiff files.")
            return

        # mask_files = list(mask_dir.glob("*.png")) + list(mask_dir.glob("*.jpg")) + list(mask_dir.glob("*.jpeg"))
        # if not mask_files:
        #     QMessageBox.warning(self, "Warning", "No PNG, JPG, or JPEG files found in the mask directory")
        #     return


        if post_count > 0 and len(mask_files) > post_count:
            QMessageBox.warning(self, "Error",
                                f"The number of mask files ({len(mask_files)}) exceeds the maximum allowed ({post_count})")
            return

        # 固定映射文件：程序目录下 color_mapping.json
        json_path = Path(__file__).parent / "color_mapping.json"
        if not json_path.is_file():
            default_map = {
                "0": "#000000",  # Background
                "1": "#FF0000",
                "2": "#00FF00",
                "3": "#0000FF",
                "4": "#FFFF00",
                "5": "#FF00FF",
                "6": "#00FFFF",
                "7": "#800000",
                "8": "#008000",
                "9": "#000080",
                "10": "#808000",
                "11": "#800080",
                "12": "#008080",
                "13": "#C0C0C0",
                "14": "#808080",
                "15": "#FFA500",
                "16": "#FFC0CB",
                "17": "#A52A2A",
                "18": "#7FFFD4",
                "19": "#D2691E",
                "20": "#FFD700"
            }
            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(default_map, f, indent=4, ensure_ascii=False)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Unable to create color_mapping.json: {e}")
                return

        try:
            with open(json_path, encoding="utf-8") as f:
                id2hex = json.load(f)

            id2color = {}
            for k, v in id2hex.items():
                col = v.lstrip('#')
                if len(col) != 6:
                    raise ValueError
                r, g, b = bytes.fromhex(col)
                id2color[int(k)] = (b, g, r)  # OpenCV is in BGR
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Invalid format in color_mapping.json: {e}")
            return


        root_results = Path(__file__).parent / 'results'
        root_results.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        res_dir = root_results / f'post_project_{timestamp}'
        res_dir.mkdir(exist_ok=True)
        out_dir = res_dir / "color_vis"
        out_dir.mkdir(exist_ok=True)


        max_key = max(id2color.keys())
        for mask_file in mask_files:
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            h, w = mask.shape
            color = np.zeros((h, w, 3), dtype=np.uint8)


            for val in np.unique(mask):

                key = val % (max_key + 1)
                color[mask == val] = id2color[key]

            cv2.imwrite(str(out_dir / mask_file.name), color)

        progress.close()

        QMessageBox.information(self, "Completed", f"The color map has been saved to {out_dir}")
