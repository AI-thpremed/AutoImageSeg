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

from PySide6.QtWidgets import QWidget, QFileDialog, QMessageBox
from PySide6.QtCore import QObject, Signal, QThread
from ui_infer import Ui_InferWindow   # pyside6-uic infer_gui.ui > infer_gui_ui.py
import os
from pathlib import Path
from infer_worker import infer_worker
from PySide6.QtCore import QObject, QTimer, QThread, Signal
from multiprocessing import Process, Queue
from config_manager import ConfigManager

from PIL import Image


class InferManager(QObject):
    log = Signal(str)
    finished = Signal()

    def __init__(self):
        super().__init__()
        self.proc = None

    def run(self, cfg: dict):
        if self.proc and self.proc.is_alive():
            return
        self.queue = Queue()
        self.proc = Process(target=infer_worker, args=(cfg, self.queue), daemon=True)
        self.proc.start()

        # read log
        self.timer = QTimer()
        self.timer.timeout.connect(self._read_log)
        self.timer.start(200)

    def _read_log(self):
        while not self.queue.empty():
            msg = self.queue.get()
            if msg is None:
                self.timer.stop()
                self.finished.emit()
                self.proc.join()
                return
            self.log.emit(str(msg))

    def stop(self):
        if self.proc and self.proc.is_alive():
            self.proc.terminate()
            self.proc.join()




class InferWindow(QWidget, Ui_InferWindow):
    def __init__(self,config: ConfigManager):
        super().__init__()
        self.config = config

        self.setupUi(self)
        self.check_passed = False

        for btn, line in [(self.btn_model, self.line_model),
                          (self.btn_img,   self.line_img)]:
            btn.clicked.connect(lambda _, l=line: self._select_dir(l))

        # button
        self.btn_check.clicked.connect(self.on_check)
        self.btn_start.clicked.connect(self.on_start)

        self.infer_mgr = InferManager()
        self.infer_mgr.log.connect(self.append_log)
        self.infer_mgr.finished.connect(self.on_finished)


    def _select_dir(self, line_edit):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            line_edit.setText(folder)

    def collect_cfg(self):
        return {
            "model_dir": self.line_model.text().strip(),
            "img_dir":   self.line_img.text().strip(),
        }


        image_max_length = self.config["restriction"].get("image_max_length", 3000)

    def on_check(self):
        cfg = self.collect_cfg()
        errs = []

        # 1. model_dir
        model_dir = Path(cfg["model_dir"])
        if not model_dir.is_dir():
            errs.append("The model directory does not exist")
        else:
            for must in ("label_mapping.json", "best.pth", "training_config.json"):
                if not (model_dir / must).exists():
                    errs.append(f"The model directory is missing {must}")

        # 2. img_dir
        img_dir = Path(cfg["img_dir"])
        if not img_dir.is_dir():
            errs.append("The image directory does not exist")
        else:
            img_files = sorted(img_dir.glob("*"))
            img_files = [f for f in img_files if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
            if not img_files:
                errs.append("No images found in the image directory")
            img_stems = {f.stem for f in img_files}

            # restriction
            image_max_length = self.config["restriction"].get("image_max_length", 3000)
            infer_count = self.config["restriction"].get("infer_count", 1000)

            if infer_count > 0 and len(img_files) > infer_count:
                errs.append(f"The number of images ({len(img_files)}) exceeds the maximum allowed ({infer_count})")

            # size restriction
            if image_max_length > 0:
                for img_file in img_files:
                    img = Image.open(img_file)
                    width, height = img.size
                    if max(width, height) > image_max_length:
                        errs.append(f"Image {img_file.name} exceeds maximum allowed length ({image_max_length})")

        if errs:
            QMessageBox.warning(self, "Validation Failed", "\n".join(errs))
            self.check_passed = False
        else:
            QMessageBox.information(self, "Validation Passed", "All checks have passed!")
            self.check_passed = True

    def on_start(self):
        if not self.check_passed:
            QMessageBox.warning(self, "Unable to start",
                                "Please click 'Check' first and ensure that the validation passes!")
            return

        cfg = self.collect_cfg()
        self.infer_mgr.run(cfg)

    def append_log(self, txt):
        self.text_log.append(txt)
        self.text_log.ensureCursorVisible()

    def on_finished(self):
        QMessageBox.information(self, "Done", "Inference finished!")