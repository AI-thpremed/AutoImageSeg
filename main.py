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

# -*- coding: utf-8 -*-

import sys
import os
import io
os.environ['TORCH_DYNAMO'] = '0'


# ==== Must be executed before all Qt imports ====
if getattr(sys, 'frozen', False):
    # Fix the special path structure of Nuitka
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    plugin_path = os.path.normpath(os.path.join(base_path, "PySide6", "qt-plugins"))  # Note that it is qt-plugins

    # Double path locking
    os.environ["QT_PLUGIN_PATH"] = plugin_path
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path
    sys.path.append(plugin_path)

    # Prevent plugins from being unloaded
    if os.name == 'nt':
        os.add_dll_directory(plugin_path)  # Windows-specific

    print(f"path: {plugin_path}")


import sys

from train_gui import TrainWindow
from train_mask_gui import TrainMaskWindow

from infer_gui import InferWindow
from post_gui import PostWindow
from eval_gui import EvalWindow
from PySide6.QtGui import QIcon


# Force UTF-8
def configure_encoding():
    if sys.stdout.encoding != 'UTF-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if sys.stderr.encoding != 'UTF-8':
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["QT_DEBUG_PLUGINS"] = "1"



configure_encoding()

from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QMessageBox
from PySide6.QtUiTools import QUiLoader
from config_manager import ConfigManager
from PySide6.QtCore import QTimer, QLibraryInfo
import json


def setup_qt_plugins():
    if hasattr(sys, '_MEIPASS'):
        plugin_path = os.path.join(sys._MEIPASS, 'PySide6', 'plugins')
        os.environ['QT_PLUGIN_PATH'] = plugin_path
    else:
        plugin_path = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
        os.environ['QT_PLUGIN_PATH'] = plugin_path


setup_qt_plugins()


def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

class MainWindow(QMainWindow):
    def __init__(self, config: ConfigManager):
        super().__init__()
        self.config = config.config
        self.setWindowTitle("AutoImageSeg v1.0")
        self.resize(600, 720)
        self.setWindowIcon(QIcon("./images/logo.png"))
        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)
        loader = QUiLoader()
        ui_path = resource_path("./UI/introduction.ui")
        welcome_widget = loader.load(ui_path)
        self.tab_widget.addTab(welcome_widget, "Home")
        self.tab_widget.addTab(TrainWindow(self.config), "Training-JSON")
        self.tab_widget.addTab(TrainMaskWindow(self.config), "Training-Mask")
        self.tab_widget.addTab(InferWindow(self.config), "Inference")
        self.tab_widget.addTab(PostWindow(self.config), "Post-Processing")
        self.tab_widget.addTab(EvalWindow(self.config), "Evaluation")


def handle_exception(exc_type, exc_value, exc_traceback):
    QMessageBox.critical(
        None,
        "Fatal Error",
        f"Unhandled exception:\n\n{str(exc_value)}",
        QMessageBox.Ok
    )
    QTimer.singleShot(1000, lambda: sys.exit(1))


if __name__ == "__main__":
    sys.excepthook = handle_exception

    app = QApplication([])
    config = ConfigManager()
    main_window = MainWindow(config)
    main_window.show()
    sys.exit(app.exec())