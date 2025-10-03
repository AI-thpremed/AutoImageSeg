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
## Form generated from reading UI file 'post_window.ui'
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
    QSizePolicy, QSpacerItem, QVBoxLayout, QWidget)

class Ui_PostForm(object):
    def setupUi(self, PostForm):
        if not PostForm.objectName():
            PostForm.setObjectName(u"PostForm")
        PostForm.resize(550, 300)
        self.verticalLayout = QVBoxLayout(PostForm)
        self.verticalLayout.setSpacing(10)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.groupInput = QGroupBox(PostForm)
        self.groupInput.setObjectName(u"groupInput")
        self.verticalLayout_2 = QVBoxLayout(self.groupInput)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.formLayout_mask = QFormLayout()
        self.formLayout_mask.setObjectName(u"formLayout_mask")
        self.label_mask = QLabel(self.groupInput)
        self.label_mask.setObjectName(u"label_mask")

        self.formLayout_mask.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_mask)

        self.horizontalLayout_mask_input = QHBoxLayout()
        self.horizontalLayout_mask_input.setObjectName(u"horizontalLayout_mask_input")
        self.line_mask = QLineEdit(self.groupInput)
        self.line_mask.setObjectName(u"line_mask")

        self.horizontalLayout_mask_input.addWidget(self.line_mask)

        self.btn_mask = QPushButton(self.groupInput)
        self.btn_mask.setObjectName(u"btn_mask")

        self.horizontalLayout_mask_input.addWidget(self.btn_mask)


        self.formLayout_mask.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_mask_input)


        self.verticalLayout_2.addLayout(self.formLayout_mask)

        self.formLayout_json = QFormLayout()
        self.formLayout_json.setObjectName(u"formLayout_json")
        self.label_json = QLabel(self.groupInput)
        self.label_json.setObjectName(u"label_json")

        self.formLayout_json.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_json)

        self.horizontalLayout_json_input = QHBoxLayout()
        self.horizontalLayout_json_input.setObjectName(u"horizontalLayout_json_input")
        self.line_json = QLineEdit(self.groupInput)
        self.line_json.setObjectName(u"line_json")

        self.horizontalLayout_json_input.addWidget(self.line_json)

        self.btn_json = QPushButton(self.groupInput)
        self.btn_json.setObjectName(u"btn_json")

        self.horizontalLayout_json_input.addWidget(self.btn_json)


        self.formLayout_json.setLayout(0, QFormLayout.ItemRole.FieldRole, self.horizontalLayout_json_input)


        self.verticalLayout_2.addLayout(self.formLayout_json)

        self.formLayout_suffix = QFormLayout()
        self.formLayout_suffix.setObjectName(u"formLayout_suffix")
        self.label_suffix = QLabel(self.groupInput)
        self.label_suffix.setObjectName(u"label_suffix")

        self.formLayout_suffix.setWidget(0, QFormLayout.ItemRole.LabelRole, self.label_suffix)

        self.combo_suffix = QComboBox(self.groupInput)
        self.combo_suffix.addItem("")
        self.combo_suffix.addItem("")
        self.combo_suffix.addItem("")
        self.combo_suffix.setObjectName(u"combo_suffix")

        self.formLayout_suffix.setWidget(0, QFormLayout.ItemRole.FieldRole, self.combo_suffix)


        self.verticalLayout_2.addLayout(self.formLayout_suffix)


        self.verticalLayout.addWidget(self.groupInput)

        self.groupActions = QGroupBox(PostForm)
        self.groupActions.setObjectName(u"groupActions")
        self.horizontalLayout = QHBoxLayout(self.groupActions)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.btn_color = QPushButton(self.groupActions)
        self.btn_color.setObjectName(u"btn_color")

        self.horizontalLayout.addWidget(self.btn_color)

        self.btn_labelme = QPushButton(self.groupActions)
        self.btn_labelme.setObjectName(u"btn_labelme")

        self.horizontalLayout.addWidget(self.btn_labelme)


        self.verticalLayout.addWidget(self.groupActions)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)


        self.retranslateUi(PostForm)

        QMetaObject.connectSlotsByName(PostForm)
    # setupUi

    def retranslateUi(self, PostForm):
        PostForm.setWindowTitle(QCoreApplication.translate("PostForm", u"Post-Processing", None))
        PostForm.setStyleSheet(QCoreApplication.translate("PostForm", u"\n"
"        QWidget { font-family: \"Microsoft YaHei\"; font-size: 10pt; }\n"
"        QGroupBox { border: 1px solid #CCCCCC; border-radius: 4px; margin-top: 10px; padding-top: 15px; }\n"
"        QPushButton { min-width: 80px; padding: 5px; }\n"
"        QLineEdit { padding: 5px; }\n"
"      ", None))
        self.groupInput.setTitle(QCoreApplication.translate("PostForm", u"File Path", None))
        self.label_mask.setText(QCoreApplication.translate("PostForm", u"Mask Path:", None))
#if QT_CONFIG(tooltip)
        self.label_mask.setToolTip(QCoreApplication.translate("PostForm", u"Select the path to the mask file.", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.line_mask.setToolTip(QCoreApplication.translate("PostForm", u"Enter the path to the mask file or use the 'Select' button to browse.", None))
#endif // QT_CONFIG(tooltip)
        self.btn_mask.setText(QCoreApplication.translate("PostForm", u"Select", None))
#if QT_CONFIG(tooltip)
        self.btn_mask.setToolTip(QCoreApplication.translate("PostForm", u"Browse for the mask file.", None))
#endif // QT_CONFIG(tooltip)
        self.label_json.setText(QCoreApplication.translate("PostForm", u"Label Mapping File:", None))
#if QT_CONFIG(tooltip)
        self.label_json.setToolTip(QCoreApplication.translate("PostForm", u"Select the path to the label mapping file.", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(tooltip)
        self.line_json.setToolTip(QCoreApplication.translate("PostForm", u"Enter the path to the label mapping file or use the 'Select' button to browse.", None))
#endif // QT_CONFIG(tooltip)
        self.btn_json.setText(QCoreApplication.translate("PostForm", u"Select", None))
#if QT_CONFIG(tooltip)
        self.btn_json.setToolTip(QCoreApplication.translate("PostForm", u"Browse for the label mapping file.", None))
#endif // QT_CONFIG(tooltip)
        self.label_suffix.setText(QCoreApplication.translate("PostForm", u"Original Image Suffix:", None))
#if QT_CONFIG(tooltip)
        self.label_suffix.setToolTip(QCoreApplication.translate("PostForm", u"Required for generating JSON files. The suffix must match the original image names (including suffix) to display correctly in LabelMe. It is recommended to keep the same suffix for all original images.", None))
#endif // QT_CONFIG(tooltip)
        self.combo_suffix.setItemText(0, QCoreApplication.translate("PostForm", u".jpg", None))
        self.combo_suffix.setItemText(1, QCoreApplication.translate("PostForm", u".png", None))
        self.combo_suffix.setItemText(2, QCoreApplication.translate("PostForm", u".jpeg", None))

        self.groupActions.setTitle(QCoreApplication.translate("PostForm", u"Post Processing", None))
        self.btn_color.setText(QCoreApplication.translate("PostForm", u"Transfer Masks into Color Images", None))
        self.btn_color.setStyleSheet(QCoreApplication.translate("PostForm", u"background-color: #2196F3; color: white;", None))
        self.btn_labelme.setText(QCoreApplication.translate("PostForm", u"Transfer Masks into LabelMe Files", None))
        self.btn_labelme.setStyleSheet(QCoreApplication.translate("PostForm", u"background-color: #4CAF50; color: white;", None))
    # retranslateUi

