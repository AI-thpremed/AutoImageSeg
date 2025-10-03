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

################################################################################
## Form generated from reading UI file 'train_window_mask.ui'
##
## Created by: Qt User Interface Compiler version 6.9.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSizePolicy, QSpinBox, QTextEdit, QVBoxLayout,
    QWidget)

class Ui_TrainWindowMask(object):
    def setupUi(self, TrainWindowMask):
        if not TrainWindowMask.objectName():
            TrainWindowMask.setObjectName(u"TrainWindowMask")
        TrainWindowMask.resize(465, 720)
        self.verticalLayout = QVBoxLayout(TrainWindowMask)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.groupPaths = QGroupBox(TrainWindowMask)
        self.groupPaths.setObjectName(u"groupPaths")
        self.formLayout = QFormLayout(self.groupPaths)
        self.formLayout.setObjectName(u"formLayout")
        self.labelTrainImg = QLabel(self.groupPaths)
        self.labelTrainImg.setObjectName(u"labelTrainImg")

        self.formLayout.setWidget(0, QFormLayout.ItemRole.LabelRole, self.labelTrainImg)

        self.hboxLayout = QHBoxLayout()
        self.hboxLayout.setObjectName(u"hboxLayout")
        self.line_train_img = QLineEdit(self.groupPaths)
        self.line_train_img.setObjectName(u"line_train_img")

        self.hboxLayout.addWidget(self.line_train_img)

        self.btn_train_img = QPushButton(self.groupPaths)
        self.btn_train_img.setObjectName(u"btn_train_img")

        self.hboxLayout.addWidget(self.btn_train_img)


        self.formLayout.setLayout(0, QFormLayout.ItemRole.FieldRole, self.hboxLayout)

        self.labelTrainJson = QLabel(self.groupPaths)
        self.labelTrainJson.setObjectName(u"labelTrainJson")

        self.formLayout.setWidget(1, QFormLayout.ItemRole.LabelRole, self.labelTrainJson)

        self.hboxLayout1 = QHBoxLayout()
        self.hboxLayout1.setObjectName(u"hboxLayout1")
        self.line_train_mask = QLineEdit(self.groupPaths)
        self.line_train_mask.setObjectName(u"line_train_mask")

        self.hboxLayout1.addWidget(self.line_train_mask)

        self.btn_train_mask = QPushButton(self.groupPaths)
        self.btn_train_mask.setObjectName(u"btn_train_mask")

        self.hboxLayout1.addWidget(self.btn_train_mask)


        self.formLayout.setLayout(1, QFormLayout.ItemRole.FieldRole, self.hboxLayout1)

        self.labelTestImg = QLabel(self.groupPaths)
        self.labelTestImg.setObjectName(u"labelTestImg")

        self.formLayout.setWidget(2, QFormLayout.ItemRole.LabelRole, self.labelTestImg)

        self.hboxLayout2 = QHBoxLayout()
        self.hboxLayout2.setObjectName(u"hboxLayout2")
        self.line_test_img = QLineEdit(self.groupPaths)
        self.line_test_img.setObjectName(u"line_test_img")

        self.hboxLayout2.addWidget(self.line_test_img)

        self.btn_test_img = QPushButton(self.groupPaths)
        self.btn_test_img.setObjectName(u"btn_test_img")

        self.hboxLayout2.addWidget(self.btn_test_img)


        self.formLayout.setLayout(2, QFormLayout.ItemRole.FieldRole, self.hboxLayout2)

        self.labelTestJson = QLabel(self.groupPaths)
        self.labelTestJson.setObjectName(u"labelTestJson")

        self.formLayout.setWidget(3, QFormLayout.ItemRole.LabelRole, self.labelTestJson)

        self.hboxLayout3 = QHBoxLayout()
        self.hboxLayout3.setObjectName(u"hboxLayout3")
        self.line_test_mask = QLineEdit(self.groupPaths)
        self.line_test_mask.setObjectName(u"line_test_mask")

        self.hboxLayout3.addWidget(self.line_test_mask)

        self.btn_test_mask = QPushButton(self.groupPaths)
        self.btn_test_mask.setObjectName(u"btn_test_mask")

        self.hboxLayout3.addWidget(self.btn_test_mask)


        self.formLayout.setLayout(3, QFormLayout.ItemRole.FieldRole, self.hboxLayout3)


        self.verticalLayout.addWidget(self.groupPaths)

        self.groupParams = QGroupBox(TrainWindowMask)
        self.groupParams.setObjectName(u"groupParams")
        self.formLayout_2 = QFormLayout(self.groupParams)
        self.formLayout_2.setObjectName(u"formLayout_2")
        self.labelAlgo = QLabel(self.groupParams)
        self.labelAlgo.setObjectName(u"labelAlgo")

        self.formLayout_2.setWidget(0, QFormLayout.ItemRole.LabelRole, self.labelAlgo)

        self.combo_algo = QComboBox(self.groupParams)
        self.combo_algo.addItem("")
        self.combo_algo.addItem("")
        self.combo_algo.addItem("")
        self.combo_algo.addItem("")
        self.combo_algo.addItem("")
        self.combo_algo.addItem("")
        self.combo_algo.addItem("")
        self.combo_algo.addItem("")
        self.combo_algo.addItem("")
        self.combo_algo.setObjectName(u"combo_algo")

        self.formLayout_2.setWidget(0, QFormLayout.ItemRole.FieldRole, self.combo_algo)

        self.labelLoss = QLabel(self.groupParams)
        self.labelLoss.setObjectName(u"labelLoss")

        self.formLayout_2.setWidget(1, QFormLayout.ItemRole.LabelRole, self.labelLoss)

        self.combo_loss = QComboBox(self.groupParams)
        self.combo_loss.addItem("")
        self.combo_loss.addItem("")
        self.combo_loss.addItem("")
        self.combo_loss.addItem("")
        self.combo_loss.setObjectName(u"combo_loss")

        self.formLayout_2.setWidget(1, QFormLayout.ItemRole.FieldRole, self.combo_loss)

        self.labelEpochs = QLabel(self.groupParams)
        self.labelEpochs.setObjectName(u"labelEpochs")

        self.formLayout_2.setWidget(2, QFormLayout.ItemRole.LabelRole, self.labelEpochs)

        self.spin_epochs = QSpinBox(self.groupParams)
        self.spin_epochs.setObjectName(u"spin_epochs")
        self.spin_epochs.setMinimum(1)
        self.spin_epochs.setMaximum(1000)
        self.spin_epochs.setValue(50)

        self.formLayout_2.setWidget(2, QFormLayout.ItemRole.FieldRole, self.spin_epochs)

        self.label_json = QLabel(self.groupParams)
        self.label_json.setObjectName(u"label_json")

        self.formLayout_2.setWidget(3, QFormLayout.ItemRole.LabelRole, self.label_json)

        self.horizontalLayout_json = QHBoxLayout()
        self.horizontalLayout_json.setObjectName(u"horizontalLayout_json")
        self.line_json = QLineEdit(self.groupParams)
        self.line_json.setObjectName(u"line_json")

        self.horizontalLayout_json.addWidget(self.line_json)

        self.btn_json = QPushButton(self.groupParams)
        self.btn_json.setObjectName(u"btn_json")

        self.horizontalLayout_json.addWidget(self.btn_json)


        self.formLayout_2.setLayout(3, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_json)


        self.verticalLayout.addWidget(self.groupParams)

        self.hLayoutButtons = QHBoxLayout()
        self.hLayoutButtons.setObjectName(u"hLayoutButtons")
        self.btn_check = QPushButton(TrainWindowMask)
        self.btn_check.setObjectName(u"btn_check")

        self.hLayoutButtons.addWidget(self.btn_check)

        self.btn_start_train = QPushButton(TrainWindowMask)
        self.btn_start_train.setObjectName(u"btn_start_train")

        self.hLayoutButtons.addWidget(self.btn_start_train)

        self.btn_stop = QPushButton(TrainWindowMask)
        self.btn_stop.setObjectName(u"btn_stop")

        self.hLayoutButtons.addWidget(self.btn_stop)


        self.verticalLayout.addLayout(self.hLayoutButtons)

        self.groupLog = QGroupBox(TrainWindowMask)
        self.groupLog.setObjectName(u"groupLog")
        self.vLayoutLog = QVBoxLayout(self.groupLog)
        self.vLayoutLog.setObjectName(u"vLayoutLog")
        self.text_log = QTextEdit(self.groupLog)
        self.text_log.setObjectName(u"text_log")
        self.text_log.setReadOnly(True)

        self.vLayoutLog.addWidget(self.text_log)


        self.verticalLayout.addWidget(self.groupLog)


        self.retranslateUi(TrainWindowMask)

        QMetaObject.connectSlotsByName(TrainWindowMask)
    # setupUi

    def retranslateUi(self, TrainWindowMask):
        TrainWindowMask.setWindowTitle(QCoreApplication.translate("TrainWindowMask", u"Train Config", None))
        self.groupPaths.setTitle(QCoreApplication.translate("TrainWindowMask", u"Image Path", None))
        self.labelTrainImg.setText(QCoreApplication.translate("TrainWindowMask", u"Training Image Path\uff1a", None))
        self.line_train_img.setText(QCoreApplication.translate("TrainWindowMask", u"G:\\2025-research-work\\autoimageseg\\datasets\\amd-sd\\train\\images", None))
        self.btn_train_img.setText(QCoreApplication.translate("TrainWindowMask", u"Select", None))
        self.labelTrainJson.setText(QCoreApplication.translate("TrainWindowMask", u"Training Mask Path\uff1a", None))
        self.line_train_mask.setText(QCoreApplication.translate("TrainWindowMask", u"G:\\2025-research-work\\autoimageseg\\datasets\\amd-sd\\train\\masks", None))
        self.btn_train_mask.setText(QCoreApplication.translate("TrainWindowMask", u"Select", None))
        self.labelTestImg.setText(QCoreApplication.translate("TrainWindowMask", u"Test Image Path\uff1a", None))
        self.line_test_img.setText(QCoreApplication.translate("TrainWindowMask", u"G:\\2025-research-work\\autoimageseg\\datasets\\amd-sd\\test\\images", None))
        self.btn_test_img.setText(QCoreApplication.translate("TrainWindowMask", u"Select", None))
        self.labelTestJson.setText(QCoreApplication.translate("TrainWindowMask", u"Test Mask Path\uff1a", None))
        self.line_test_mask.setText(QCoreApplication.translate("TrainWindowMask", u"G:\\2025-research-work\\autoimageseg\\datasets\\amd-sd\\test\\masks", None))
        self.btn_test_mask.setText(QCoreApplication.translate("TrainWindowMask", u"Select", None))
        self.groupParams.setTitle(QCoreApplication.translate("TrainWindowMask", u"Training Setting", None))
        self.labelAlgo.setText(QCoreApplication.translate("TrainWindowMask", u"Algorithm:", None))
        self.combo_algo.setItemText(0, QCoreApplication.translate("TrainWindowMask", u"MobileUNet", None))
        self.combo_algo.setItemText(1, QCoreApplication.translate("TrainWindowMask", u"FastSCNN", None))
        self.combo_algo.setItemText(2, QCoreApplication.translate("TrainWindowMask", u"LinkNet", None))
        self.combo_algo.setItemText(3, QCoreApplication.translate("TrainWindowMask", u"UNet", None))
        self.combo_algo.setItemText(4, QCoreApplication.translate("TrainWindowMask", u"UNext", None))
        self.combo_algo.setItemText(5, QCoreApplication.translate("TrainWindowMask", u"AttU_Net", None))
        self.combo_algo.setItemText(6, QCoreApplication.translate("TrainWindowMask", u"NestedUNet", None))
        self.combo_algo.setItemText(7, QCoreApplication.translate("TrainWindowMask", u"UNetResnet", None))
        self.combo_algo.setItemText(8, QCoreApplication.translate("TrainWindowMask", u"FCN", None))

        self.labelLoss.setText(QCoreApplication.translate("TrainWindowMask", u"Loss Function:", None))
        self.combo_loss.setItemText(0, QCoreApplication.translate("TrainWindowMask", u"CrossEntropyLoss", None))
        self.combo_loss.setItemText(1, QCoreApplication.translate("TrainWindowMask", u"CE_DiceLoss", None))
        self.combo_loss.setItemText(2, QCoreApplication.translate("TrainWindowMask", u"FocalLoss", None))
        self.combo_loss.setItemText(3, QCoreApplication.translate("TrainWindowMask", u"LovaszSoftmax", None))

        self.labelEpochs.setText(QCoreApplication.translate("TrainWindowMask", u"Epoch:", None))
        self.label_json.setText(QCoreApplication.translate("TrainWindowMask", u"Label Mapping File:", None))
        self.btn_json.setText(QCoreApplication.translate("TrainWindowMask", u"Select", None))
        self.btn_check.setText(QCoreApplication.translate("TrainWindowMask", u"Check Before Train", None))
        self.btn_start_train.setText(QCoreApplication.translate("TrainWindowMask", u"Start Training", None))
        self.btn_stop.setText(QCoreApplication.translate("TrainWindowMask", u"Stop Training", None))
        self.groupLog.setTitle(QCoreApplication.translate("TrainWindowMask", u"Log", None))
    # retranslateUi

