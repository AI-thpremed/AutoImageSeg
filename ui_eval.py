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
## Form generated from reading UI file 'eval.ui'
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
from PySide6.QtWidgets import (QApplication, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QPushButton, QSizePolicy,
    QSpacerItem, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget)

class Ui_EvalForm(object):
    def setupUi(self, EvalForm):
        if not EvalForm.objectName():
            EvalForm.setObjectName(u"EvalForm")
        EvalForm.resize(600, 720)
        self.verticalLayout = QVBoxLayout(EvalForm)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(9, 9, 9, 9)
        self.groupLabelme = QGroupBox(EvalForm)
        self.groupLabelme.setObjectName(u"groupLabelme")
        self.verticalLayout_2 = QVBoxLayout(self.groupLabelme)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.horizontalLayout_json = QHBoxLayout()
        self.horizontalLayout_json.setObjectName(u"horizontalLayout_json")
        self.labelTrainJson = QLabel(self.groupLabelme)
        self.labelTrainJson.setObjectName(u"labelTrainJson")

        self.horizontalLayout_json.addWidget(self.labelTrainJson)

        self.lineTrainJson = QLineEdit(self.groupLabelme)
        self.lineTrainJson.setObjectName(u"lineTrainJson")

        self.horizontalLayout_json.addWidget(self.lineTrainJson)

        self.btnSelectJson = QPushButton(self.groupLabelme)
        self.btnSelectJson.setObjectName(u"btnSelectJson")

        self.horizontalLayout_json.addWidget(self.btnSelectJson)


        self.verticalLayout_2.addLayout(self.horizontalLayout_json)

        self.labelPolicy = QLabel(self.groupLabelme)
        self.labelPolicy.setObjectName(u"labelPolicy")

        self.verticalLayout_2.addWidget(self.labelPolicy)

        self.lineLabelFilter = QLineEdit(self.groupLabelme)
        self.lineLabelFilter.setObjectName(u"lineLabelFilter")

        self.verticalLayout_2.addWidget(self.lineLabelFilter)

        self.linePolicy = QLineEdit(self.groupLabelme)
        self.linePolicy.setObjectName(u"linePolicy")

        self.verticalLayout_2.addWidget(self.linePolicy)

        self.tablePolicy = QTableWidget(self.groupLabelme)
        if (self.tablePolicy.columnCount() < 2):
            self.tablePolicy.setColumnCount(2)
        __qtablewidgetitem = QTableWidgetItem()
        self.tablePolicy.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        self.tablePolicy.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        if (self.tablePolicy.rowCount() < 1):
            self.tablePolicy.setRowCount(1)
        self.tablePolicy.setObjectName(u"tablePolicy")
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.tablePolicy.sizePolicy().hasHeightForWidth())
        self.tablePolicy.setSizePolicy(sizePolicy)
        self.tablePolicy.setColumnCount(2)
        self.tablePolicy.horizontalHeader().setVisible(True)

        self.verticalLayout_2.addWidget(self.tablePolicy)

        self.btnLabelme2Mask = QPushButton(self.groupLabelme)
        self.btnLabelme2Mask.setObjectName(u"btnLabelme2Mask")

        self.verticalLayout_2.addWidget(self.btnLabelme2Mask)


        self.verticalLayout.addWidget(self.groupLabelme)

        self.groupEval = QGroupBox(EvalForm)
        self.groupEval.setObjectName(u"groupEval")
        self.verticalLayout_3 = QVBoxLayout(self.groupEval)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.horizontalLayout_mask1 = QHBoxLayout()
        self.horizontalLayout_mask1.setObjectName(u"horizontalLayout_mask1")
        self.labelMask1 = QLabel(self.groupEval)
        self.labelMask1.setObjectName(u"labelMask1")

        self.horizontalLayout_mask1.addWidget(self.labelMask1)

        self.lineMask1 = QLineEdit(self.groupEval)
        self.lineMask1.setObjectName(u"lineMask1")

        self.horizontalLayout_mask1.addWidget(self.lineMask1)

        self.btnSelectMask1 = QPushButton(self.groupEval)
        self.btnSelectMask1.setObjectName(u"btnSelectMask1")

        self.horizontalLayout_mask1.addWidget(self.btnSelectMask1)


        self.verticalLayout_3.addLayout(self.horizontalLayout_mask1)

        self.horizontalLayout_mask2 = QHBoxLayout()
        self.horizontalLayout_mask2.setObjectName(u"horizontalLayout_mask2")
        self.labelMask2 = QLabel(self.groupEval)
        self.labelMask2.setObjectName(u"labelMask2")

        self.horizontalLayout_mask2.addWidget(self.labelMask2)

        self.lineMask2 = QLineEdit(self.groupEval)
        self.lineMask2.setObjectName(u"lineMask2")

        self.horizontalLayout_mask2.addWidget(self.lineMask2)

        self.btnSelectMask2 = QPushButton(self.groupEval)
        self.btnSelectMask2.setObjectName(u"btnSelectMask2")

        self.horizontalLayout_mask2.addWidget(self.btnSelectMask2)


        self.verticalLayout_3.addLayout(self.horizontalLayout_mask2)

        self.btnEval = QPushButton(self.groupEval)
        self.btnEval.setObjectName(u"btnEval")

        self.verticalLayout_3.addWidget(self.btnEval)


        self.verticalLayout.addWidget(self.groupEval)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)

        self.verticalLayout.addItem(self.verticalSpacer)


        self.retranslateUi(EvalForm)

        QMetaObject.connectSlotsByName(EvalForm)
    # setupUi

    def retranslateUi(self, EvalForm):
        EvalForm.setWindowTitle(QCoreApplication.translate("EvalForm", u"Evaluation", None))
        self.groupLabelme.setTitle(QCoreApplication.translate("EvalForm", u"Convert LabelMe JSON", None))
        self.labelTrainJson.setText(QCoreApplication.translate("EvalForm", u"JSON path\uff1a", None))
        self.btnSelectJson.setText(QCoreApplication.translate("EvalForm", u"Select", None))
        self.labelPolicy.setText(QCoreApplication.translate("EvalForm", u"Label Rendering Policy:", None))
        self.lineLabelFilter.setPlaceholderText(QCoreApplication.translate("EvalForm", u"Enter labels (e.g., cat;dog;person)", None))
        self.linePolicy.setPlaceholderText(QCoreApplication.translate("EvalForm", u"Enter policies (e.g., Closure;Follow;Follow)", None))
        ___qtablewidgetitem = self.tablePolicy.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("EvalForm", u"Label", None));
        ___qtablewidgetitem1 = self.tablePolicy.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("EvalForm", u"Policy", None));
        self.btnLabelme2Mask.setText(QCoreApplication.translate("EvalForm", u"Transfer LabelMe JSON into Mask Images", None))
        self.groupEval.setTitle(QCoreApplication.translate("EvalForm", u"Segmentation Evaluation", None))
        self.labelMask1.setText(QCoreApplication.translate("EvalForm", u"Mask File Path 1:", None))
        self.btnSelectMask1.setText(QCoreApplication.translate("EvalForm", u"Select", None))
        self.labelMask2.setText(QCoreApplication.translate("EvalForm", u"Mask File Path 2:", None))
        self.btnSelectMask2.setText(QCoreApplication.translate("EvalForm", u"Select", None))
        self.btnEval.setText(QCoreApplication.translate("EvalForm", u"Make Evaluation", None))
    # retranslateUi

