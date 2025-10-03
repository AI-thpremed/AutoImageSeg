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
## Form generated from reading UI file 'infer.ui'
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
from PySide6.QtWidgets import (QApplication, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSizePolicy,
    QTextEdit, QVBoxLayout, QWidget)

class Ui_InferWindow(object):
    def setupUi(self, InferWindow):
        if not InferWindow.objectName():
            InferWindow.setObjectName(u"InferWindow")
        InferWindow.resize(600, 480)
        self.verticalLayout = QVBoxLayout(InferWindow)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.groupPaths = QGroupBox(InferWindow)
        self.groupPaths.setObjectName(u"groupPaths")
        self.formLayout = QFormLayout(self.groupPaths)
        self.formLayout.setObjectName(u"formLayout")
        self.labelModel = QLabel(self.groupPaths)
        self.labelModel.setObjectName(u"labelModel")

        self.formLayout.setWidget(0, QFormLayout.ItemRole.LabelRole, self.labelModel)

        self.hboxLayout = QHBoxLayout()
        self.hboxLayout.setObjectName(u"hboxLayout")
        self.line_model = QLineEdit(self.groupPaths)
        self.line_model.setObjectName(u"line_model")

        self.hboxLayout.addWidget(self.line_model)

        self.btn_model = QPushButton(self.groupPaths)
        self.btn_model.setObjectName(u"btn_model")

        self.hboxLayout.addWidget(self.btn_model)


        self.formLayout.setLayout(0, QFormLayout.ItemRole.FieldRole, self.hboxLayout)

        self.labelImg = QLabel(self.groupPaths)
        self.labelImg.setObjectName(u"labelImg")

        self.formLayout.setWidget(1, QFormLayout.ItemRole.LabelRole, self.labelImg)

        self.hboxLayout1 = QHBoxLayout()
        self.hboxLayout1.setObjectName(u"hboxLayout1")
        self.line_img = QLineEdit(self.groupPaths)
        self.line_img.setObjectName(u"line_img")

        self.hboxLayout1.addWidget(self.line_img)

        self.btn_img = QPushButton(self.groupPaths)
        self.btn_img.setObjectName(u"btn_img")

        self.hboxLayout1.addWidget(self.btn_img)


        self.formLayout.setLayout(1, QFormLayout.ItemRole.FieldRole, self.hboxLayout1)


        self.verticalLayout.addWidget(self.groupPaths)

        self.hboxLayout2 = QHBoxLayout()
        self.hboxLayout2.setObjectName(u"hboxLayout2")
        self.btn_check = QPushButton(InferWindow)
        self.btn_check.setObjectName(u"btn_check")

        self.hboxLayout2.addWidget(self.btn_check)

        self.btn_start = QPushButton(InferWindow)
        self.btn_start.setObjectName(u"btn_start")

        self.hboxLayout2.addWidget(self.btn_start)


        self.verticalLayout.addLayout(self.hboxLayout2)

        self.groupLog = QGroupBox(InferWindow)
        self.groupLog.setObjectName(u"groupLog")
        self.vboxLayout = QVBoxLayout(self.groupLog)
        self.vboxLayout.setObjectName(u"vboxLayout")
        self.text_log = QTextEdit(self.groupLog)
        self.text_log.setObjectName(u"text_log")

        self.vboxLayout.addWidget(self.text_log)


        self.verticalLayout.addWidget(self.groupLog)


        self.retranslateUi(InferWindow)

        QMetaObject.connectSlotsByName(InferWindow)
    # setupUi

    def retranslateUi(self, InferWindow):
        InferWindow.setWindowTitle(QCoreApplication.translate("InferWindow", u"Inference", None))
        self.groupPaths.setTitle(QCoreApplication.translate("InferWindow", u"Paths", None))
        self.labelModel.setText(QCoreApplication.translate("InferWindow", u"Model Result:", None))
        self.btn_model.setText(QCoreApplication.translate("InferWindow", u"Select", None))
        self.labelImg.setText(QCoreApplication.translate("InferWindow", u"Image Dir:", None))
        self.btn_img.setText(QCoreApplication.translate("InferWindow", u"Select", None))
        self.btn_check.setText(QCoreApplication.translate("InferWindow", u"Check", None))
        self.btn_start.setText(QCoreApplication.translate("InferWindow", u"Start Inference", None))
        self.groupLog.setTitle(QCoreApplication.translate("InferWindow", u"Log", None))
    # retranslateUi

