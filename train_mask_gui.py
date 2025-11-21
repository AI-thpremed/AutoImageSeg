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

import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from ui_train_window_mask import Ui_TrainWindowMask   # pyuic6
from PySide6.QtCore import QObject, QTimer, QThread, Signal
from multiprocessing import Process, Queue
import os, sys
sys.path.append(os.path.dirname(__file__))
from train_worker_mask import train_worker_mask
from PySide6.QtWidgets import QWidget
from config_manager import ConfigManager
from typing import Dict, List, Optional
from pathlib import Path
import json
from PIL import Image


class TrainingManager(QObject):
    log = Signal(str)
    finished = Signal()

    def __init__(self):
        super().__init__()
        self.proc = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._read_log)

    def start_training(self, cfg: dict):
        if self.proc and self.proc.is_alive():
            return
        self.log_q = Queue()
        self.proc = Process(
            target=train_worker_mask,
            args=(cfg, self.log_q),
            daemon=True
        )
        self.proc.start()
        self.timer.start(200)

    def _read_log(self):
        while not self.log_q.empty():
            msg = self.log_q.get()
            if msg is None:
                self.timer.stop()
                self.finished.emit()
                self.proc.join()
                break
            self.log.emit(str(msg))

    def stop(self):
        if self.proc and self.proc.is_alive():
            self.proc.terminate()
            self.proc.join()




class TrainMaskWindow(QWidget, Ui_TrainWindowMask):
    def __init__(self,config: ConfigManager):
        super().__init__()
        self.config = config
        self.setupUi(self)
        self.check_passed = False

        self.training_mgr = TrainingManager()
        self.training_mgr.log.connect(self.append_log)
        self.training_mgr.finished.connect(self.on_training_finished)


        self.btn_check.clicked.connect(self.on_check)
        self.btn_start_train.clicked.connect(self.on_start_train)
        self.btn_stop.clicked.connect(self.on_stop_train)

        for btn, line in [
            (self.btn_train_img,   self.line_train_img),
            (self.btn_train_mask,  self.line_train_mask),
            (self.btn_test_img,    self.line_test_img),
            (self.btn_test_mask,   self.line_test_mask),
        ]:
            btn.clicked.connect(lambda _, l=line: self.select_folder(l))

        self.btn_json.clicked.connect(lambda: self._select_file(self.line_json))


    def _select_file(self, line):
        file, _ = QFileDialog.getOpenFileName(self, "Select File", filter="JSON (*.json)")
        if file:
            line.setText(file)

    def append_log(self, txt):
        self.text_log.append(txt)
        self.text_log.ensureCursorVisible()

    def on_training_finished(self):
        QMessageBox.information(self, "Completed", "Training has finished!")

    def on_stop_train(self):
        self.training_mgr.stop()
        QMessageBox.information(self, "Stopped", "Training has been manually terminated")

    def select_folder(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            line_edit.setText(folder)






    def collect_cfg(self) -> dict:
        return {
            "train_img":  self.line_train_img.text().strip(),
            "train_mask": self.line_train_mask.text().strip(),
            "test_img":   self.line_test_img.text().strip(),
            "test_mask":  self.line_test_mask.text().strip(),
            "algo":           self.combo_algo.currentText(),
            "loss":           self.combo_loss.currentText(),
            "epochs":     self.spin_epochs.value(),
            "line_json": self.line_json.text().strip(),
        }




    def on_check(self):
        cfg = self.collect_cfg()

        if cfg["epochs"] < 2:
            QMessageBox.warning(self, "Validation Failed", "The number of epochs must be ≥ 2")
            self.check_passed = False
            return

        if not cfg["line_json"].strip():
            QMessageBox.warning(self, "Validation Failed", "The JSON file path must not be empty")
            self.check_passed = False
            return

        try:
            with open(cfg["line_json"].strip(), "r") as f:
                json_data = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, "Validation Failed", f"Failed to read JSON file: {e}")
            self.check_passed = False
            return

        if not isinstance(json_data, dict) or "_background_" not in json_data or json_data["_background_"] != 0:
            QMessageBox.warning(self, "Validation Failed", "The JSON file must contain '_background_': 0")
            self.check_passed = False
            return

        lesion_keys = [k for k in json_data.keys() if k != "_background_"]
        if len(lesion_keys) > 99:
            QMessageBox.warning(self, "Validation Failed", "The number of lesions must not exceed 99")
            self.check_passed = False
            return

        try:
            lesion_values = sorted([int(json_data[k]) for k in lesion_keys])
            if lesion_values != list(range(1, len(lesion_keys) + 1)):
                QMessageBox.warning(self, "Validation Failed",
                                    "Lesion numbers must be consecutive integers starting from 1")
                self.check_passed = False
                return
        except ValueError:
            QMessageBox.warning(self, "Validation Failed", "Lesion values must be integers")
            self.check_passed = False
            return

        errs = self.check_paths(cfg)
        if errs:
            QMessageBox.warning(self, "Validation Failed", "\n".join(errs))
            self.check_passed = False
        else:
            QMessageBox.information(self, "Validation Passed", "All validation checks have passed!")
            self.check_passed = True




    def on_start_train(self):
        if not self.check_passed:
            QMessageBox.warning(self, "Cannot Start",
                                "Please click 'Check' first and ensure that the validation passes!")
            return
        cfg = self.collect_cfg()
        self.training_mgr.start_training(cfg)

    def check_paths(self, cfg: Dict[str, str]) -> Optional[List[str]]:
        errors = []

        # ---------- 1. 路径存在性 ----------
        path_keys = ["train_img", "train_mask", "test_img", "test_mask"]
        for k in path_keys:
            if not os.path.exists(cfg[k]):
                errors.append(f"{k} path does not exist: {cfg[k]}")
        if errors:
            return errors

        # ---------- 2. 统一的内部函数 ----------
        def _check_group(img_dir: str, mask_dir: str, tag: str,
                         max_length: int, max_count: int):
            img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

            # 记录第一张合法图片/mask的后缀，用于统一性检查
            ref_img_ext = None
            ref_mask_ext = None

            # 收集文件名（无后缀）的同时做后缀一致性检查
            imgs = set()
            for f in os.listdir(img_dir):
                ext = os.path.splitext(f)[1].lower()
                if ext not in img_exts:
                    continue
                name = os.path.splitext(f)[0]
                imgs.add(name)

                if ref_img_ext is None:
                    ref_img_ext = ext
                elif ext != ref_img_ext:
                    errors.append(f"{tag} image directory has mixed extensions: found {ext} vs {ref_img_ext}")

                    ref_img_ext = ext

            masks = set()
            for f in os.listdir(mask_dir):
                ext = os.path.splitext(f)[1].lower()
                if ext not in img_exts:
                    continue
                name = os.path.splitext(f)[0]
                masks.add(name)

                if ref_mask_ext is None:
                    ref_mask_ext = ext
                elif ext != ref_mask_ext:
                    errors.append(f"{tag} mask directory has mixed extensions: found {ext} vs {ref_mask_ext}")
                    ref_mask_ext = ext



            if len(imgs) != len(masks):
                errors.append(
                    f"{tag} number of images ({len(imgs)}) does not match number of masks ({len(masks)})")

            only_img = imgs - masks
            only_mask = masks - imgs
            if only_img:
                errors.append(f"{tag} images missing corresponding mask files: {sorted(only_img)}")
            if only_mask:
                errors.append(f"{tag} mask files missing corresponding images: {sorted(only_mask)}")

            if max_count > 0 and len(imgs) > max_count:
                errors.append(f"{tag} number of images ({len(imgs)}) exceeds the maximum allowed ({max_count})")

            if max_length > 0:
                for f in os.listdir(img_dir):
                    if os.path.splitext(f)[1].lower() not in img_exts:
                        continue
                    img_path = os.path.join(img_dir, f)
                    try:
                        with Image.open(img_path) as img:
                            width, height = img.size
                            if max(width, height) > max_length:
                                errors.append(f"{tag} image {f} exceeds maximum allowed length ({max_length})")
                    except Exception as e:
                        errors.append(f"{tag} read image {f} failed: {e}")

        # ---------- 3. 训练集 / 测试集分别检查 ----------
        image_max_length = self.config["restriction"].get("image_max_length", 3000)
        train_count = self.config["restriction"].get("train_count", 1000)
        test_count = self.config["restriction"].get("test_count", 1000)

        _check_group(cfg["train_img"], cfg["train_mask"], "Train", image_max_length, train_count)
        _check_group(cfg["test_img"], cfg["test_mask"], "Test", image_max_length, test_count)

        return errors or None


    #
    #
    # def check_paths(self, cfg: Dict[str, str]) -> Optional[List[str]]:
    #
    #     errors = []
    #
    #     path_keys = ["train_img", "train_mask", "test_img", "test_mask"]
    #     for k in path_keys:
    #         if not os.path.exists(cfg[k]):
    #             errors.append(f"{k} path does not exist: {cfg[k]}")
    #
    #     if errors:
    #         return errors
    #
    #     def _check_group(img_dir: str, mask_dir: str, tag: str, max_length: int, max_count: int):
    #         img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    #         imgs = {os.path.splitext(f)[0]
    #                 for f in os.listdir(img_dir)
    #                 if os.path.splitext(f)[1].lower() in img_exts}
    #         masks = {os.path.splitext(f)[0]
    #                  for f in os.listdir(mask_dir)
    #                  if os.path.splitext(f)[1].lower() in img_exts}
    #
    #         if len(imgs) != len(masks):
    #             errors.append(
    #                 f"{tag} number of images ({len(imgs)}) does not match number of masks ({len(masks)})")
    #
    #         only_img = imgs - masks
    #         only_mask = masks - imgs
    #         if only_img:
    #             errors.append(f"{tag} images missing corresponding mask files: {sorted(only_img)}")
    #         if only_mask:
    #             errors.append(f"{tag} mask files missing corresponding images: {sorted(only_mask)}")
    #
    #         if max_count > 0 and len(imgs) > max_count:
    #             errors.append(f"{tag} number of images ({len(imgs)}) exceeds the maximum allowed ({max_count})")
    #
    #         if max_length > 0:
    #             for f in os.listdir(img_dir):
    #                 if os.path.splitext(f)[1].lower() in img_exts:
    #                     img_path = os.path.join(img_dir, f)
    #                     img = Image.open(img_path)
    #                     width, height = img.size
    #                     if max(width, height) > max_length:
    #                         errors.append(f"{tag} image {f} exceeds maximum allowed length ({max_length})")
    #
    #     image_max_length = self.config["restriction"].get("image_max_length", 3000)
    #     train_count = self.config["restriction"].get("train_count", 1000)
    #     test_count = self.config["restriction"].get("test_count", 1000)
    #
    #     _check_group(cfg["train_img"], cfg["train_mask"], "Train", image_max_length, train_count)
    #     _check_group(cfg["test_img"], cfg["test_mask"], "Test", image_max_length, test_count)
    #
    #     return errors or None
