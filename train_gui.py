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
from ui_train_window import Ui_TrainWindow   # pyuic6
from PySide6.QtCore import QObject, QTimer, QThread, Signal
from multiprocessing import Process, Queue
import os, sys
sys.path.append(os.path.dirname(__file__))
from train_worker import train_worker
from PySide6.QtWidgets import QTableWidgetItem   # PySide6
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
            target=train_worker,
            args=(cfg, self.log_q),
            daemon=True
        )
        self.proc.start()
        self.timer.start(200)   #  200 ms per read

    def _read_log(self):
        while not self.log_q.empty():
            msg = self.log_q.get()
            if msg is None:      # sub process end signal
                self.timer.stop()
                self.finished.emit()
                self.proc.join()
                break
            self.log.emit(str(msg))

    def stop(self):
        if self.proc and self.proc.is_alive():
            self.proc.terminate()
            self.proc.join()




class TrainWindow(QWidget, Ui_TrainWindow):
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
            (self.btn_train_json,  self.line_train_json),
            (self.btn_test_img,    self.line_test_img),
            (self.btn_test_json,   self.line_test_json),
        ]:
            btn.clicked.connect(lambda _, l=line: self.select_folder(l))

    def _select_file(self, line):
        file, _ = QFileDialog.getOpenFileName(self, "Select File", filter="JSON (*.json)")
        if file:
            line.setText(file)

    def append_log(self, txt):
        """add log to QTextEdit"""
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
            "train_json": self.line_train_json.text().strip(),
            "test_img":   self.line_test_img.text().strip(),
            "test_json":  self.line_test_json.text().strip(),
            "algo":           self.combo_algo.currentText(),
            "loss":           self.combo_loss.currentText(),
            "epochs":     self.spin_epochs.value(),
            # "source": self.combo_data_source.currentText(),
            "labels_to_use": self.line_label_filter.text().strip(),
            "policies": self.line_policy.text().strip(),
            # "line_json": self.line_json.text().strip(),
        }



    def on_check(self):
        cfg = self.collect_cfg()
        if cfg["epochs"] < 2:
            QMessageBox.warning(self, "Validation Failed", "The number of epochs must be ≥ 2")
            self.check_passed = False
            return


        labels = [t.strip() for t in cfg["labels_to_use"].split(";") if t.strip()]
        policies = [p.strip() for p in cfg["policies"].split(";") if p.strip()]


        if not labels:
            QMessageBox.warning(self, "Validation Failed", "The number of classes must be ≥ 1")
            self.check_passed = False
            return
        if len(labels) != len(policies):
            QMessageBox.warning(self, "Validation Failed",
                                f"The number of labels ({len(labels)}) does not match the number of policies ({len(policies)})")
            self.check_passed = False
            return
        invalid = [p for p in policies if p not in {"Closure", "Follow"}]
        if invalid:
            QMessageBox.warning(self, "Validation Failed",
                                f"Policies can only be 'Closure' or 'Follow'. Invalid values found: {invalid}")
            self.check_passed = False
            return

        self._refresh_policy_table(labels, policies)

        errs = self.check_paths(cfg)

        if errs:
            QMessageBox.warning(self, "Validation Failed", "\n".join(errs))
            self.check_passed = False
        else:
            QMessageBox.information(self, "Validation Passed", "All validation checks have passed!")
            self.check_passed = True


    def _refresh_policy_table(self, labels: list, policies: list):
        self.table_policy.setRowCount(len(labels))
        for row, (lab, pol) in enumerate(zip(labels, policies)):
            self.table_policy.setItem(row, 0, QTableWidgetItem(lab))
            self.table_policy.setItem(row, 1, QTableWidgetItem(pol))

    def on_start_train(self):
        if not self.check_passed:
            QMessageBox.warning(self, "Cannot Start",
                                "Please click 'Check' first and ensure that the validation passes!")
            return
        cfg = self.collect_cfg()
        self.training_mgr.start_training(cfg)





    def check_paths(self, cfg: Dict[str, str]) -> Optional[List[str]]:

        errors = []

        path_keys = ["train_img", "train_json", "test_img", "test_json"]
        for k in path_keys:
            if not os.path.exists(cfg[k]):
                errors.append(f"{k} path does not exist: {cfg[k]}")

        if errors:
            return errors

        def _check_group(img_dir: str, json_dir: str, tag: str, max_length: int, max_count: int):

            img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
            imgs = {os.path.splitext(f)[0]
                    for f in os.listdir(img_dir)
                    if os.path.splitext(f)[1].lower() in img_exts}
            jsons = {os.path.splitext(f)[0]
                     for f in os.listdir(json_dir)
                     if f.endswith(".json")}

            if len(imgs) != len(jsons):
                errors.append(
                    f"{tag} number of images ({len(imgs)}) does not match number of JSON files ({len(jsons)})")

            only_img = imgs - jsons
            only_json = jsons - imgs
            if only_img:
                errors.append(f"{tag} images missing corresponding JSON files: {sorted(only_img)}")
            if only_json:
                errors.append(f"{tag} JSON files missing corresponding images: {sorted(only_json)}")

            if max_count > 0 and len(imgs) > max_count:
                errors.append(f"{tag} number of images ({len(imgs)}) exceeds the maximum allowed ({max_count})")

            if max_length > 0:
                for f in os.listdir(img_dir):
                    if os.path.splitext(f)[1].lower() in img_exts:
                        img_path = os.path.join(img_dir, f)
                        img = Image.open(img_path)
                        width, height = img.size
                        if max(width, height) > max_length:
                            errors.append(f"{tag} image {f} exceeds maximum allowed length ({max_length})")

        image_max_length = self.config["restriction"].get("image_max_length", 3000)
        train_count = self.config["restriction"].get("train_count", 1000)
        test_count = self.config["restriction"].get("test_count", 1000)

        _check_group(cfg["train_img"], cfg["train_json"], "Train", image_max_length, train_count)
        _check_group(cfg["test_img"], cfg["test_json"], "Test", image_max_length, test_count)

        return errors or None
