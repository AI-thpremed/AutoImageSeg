# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'train_window.ui'
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
    QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QPushButton, QSizePolicy, QSpinBox, QTableWidget,
    QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget)

class Ui_TrainWindow(object):
    def setupUi(self, TrainWindow):
        if not TrainWindow.objectName():
            TrainWindow.setObjectName(u"TrainWindow")
        TrainWindow.resize(465, 720)
        self.verticalLayout = QVBoxLayout(TrainWindow)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.groupPaths = QGroupBox(TrainWindow)
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
        self.line_train_json = QLineEdit(self.groupPaths)
        self.line_train_json.setObjectName(u"line_train_json")

        self.hboxLayout1.addWidget(self.line_train_json)

        self.btn_train_json = QPushButton(self.groupPaths)
        self.btn_train_json.setObjectName(u"btn_train_json")

        self.hboxLayout1.addWidget(self.btn_train_json)


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
        self.line_test_json = QLineEdit(self.groupPaths)
        self.line_test_json.setObjectName(u"line_test_json")

        self.hboxLayout3.addWidget(self.line_test_json)

        self.btn_test_json = QPushButton(self.groupPaths)
        self.btn_test_json.setObjectName(u"btn_test_json")

        self.hboxLayout3.addWidget(self.btn_test_json)


        self.formLayout.setLayout(3, QFormLayout.ItemRole.FieldRole, self.hboxLayout3)


        self.verticalLayout.addWidget(self.groupPaths)

        self.groupParams = QGroupBox(TrainWindow)
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

        self.labelPolicy = QLabel(self.groupParams)
        self.labelPolicy.setObjectName(u"labelPolicy")

        self.formLayout_2.setWidget(3, QFormLayout.ItemRole.LabelRole, self.labelPolicy)

        self.vboxLayout = QVBoxLayout()
        self.vboxLayout.setObjectName(u"vboxLayout")
        self.line_label_filter = QLineEdit(self.groupParams)
        self.line_label_filter.setObjectName(u"line_label_filter")

        self.vboxLayout.addWidget(self.line_label_filter)

        self.line_policy = QLineEdit(self.groupParams)
        self.line_policy.setObjectName(u"line_policy")

        self.vboxLayout.addWidget(self.line_policy)

        self.table_policy = QTableWidget(self.groupParams)
        if (self.table_policy.columnCount() < 2):
            self.table_policy.setColumnCount(2)
        __qtablewidgetitem = QTableWidgetItem()
        self.table_policy.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        __qtablewidgetitem1.setTextAlignment(Qt.AlignLeading|Qt.AlignVCenter);
        self.table_policy.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        if (self.table_policy.rowCount() < 1):
            self.table_policy.setRowCount(1)
        self.table_policy.setObjectName(u"table_policy")
        self.table_policy.setColumnCount(2)
        self.table_policy.horizontalHeader().setVisible(True)
        self.table_policy.horizontalHeader().setCascadingSectionResizes(False)

        self.vboxLayout.addWidget(self.table_policy)


        self.formLayout_2.setLayout(3, QFormLayout.ItemRole.FieldRole, self.vboxLayout)


        self.verticalLayout.addWidget(self.groupParams)

        self.hLayoutButtons = QHBoxLayout()
        self.hLayoutButtons.setObjectName(u"hLayoutButtons")
        self.btn_check = QPushButton(TrainWindow)
        self.btn_check.setObjectName(u"btn_check")

        self.hLayoutButtons.addWidget(self.btn_check)

        self.btn_start_train = QPushButton(TrainWindow)
        self.btn_start_train.setObjectName(u"btn_start_train")

        self.hLayoutButtons.addWidget(self.btn_start_train)

        self.btn_stop = QPushButton(TrainWindow)
        self.btn_stop.setObjectName(u"btn_stop")

        self.hLayoutButtons.addWidget(self.btn_stop)


        self.verticalLayout.addLayout(self.hLayoutButtons)

        self.groupLog = QGroupBox(TrainWindow)
        self.groupLog.setObjectName(u"groupLog")
        self.vLayoutLog = QVBoxLayout(self.groupLog)
        self.vLayoutLog.setObjectName(u"vLayoutLog")
        self.text_log = QTextEdit(self.groupLog)
        self.text_log.setObjectName(u"text_log")
        self.text_log.setReadOnly(True)

        self.vLayoutLog.addWidget(self.text_log)


        self.verticalLayout.addWidget(self.groupLog)


        self.retranslateUi(TrainWindow)

        QMetaObject.connectSlotsByName(TrainWindow)
    # setupUi

    def retranslateUi(self, TrainWindow):
        TrainWindow.setWindowTitle(QCoreApplication.translate("TrainWindow", u"Train Config", None))
        self.groupPaths.setTitle(QCoreApplication.translate("TrainWindow", u"Image Path", None))
        self.labelTrainImg.setText(QCoreApplication.translate("TrainWindow", u"Training Image Path\uff1a", None))
        self.line_train_img.setText("")
        self.btn_train_img.setText(QCoreApplication.translate("TrainWindow", u"Select", None))
        self.labelTrainJson.setText(QCoreApplication.translate("TrainWindow", u"Training LabelMe JSON Path\uff1a", None))
        self.line_train_json.setText("")
        self.btn_train_json.setText(QCoreApplication.translate("TrainWindow", u"Select", None))
        self.labelTestImg.setText(QCoreApplication.translate("TrainWindow", u"Test Image Path\uff1a", None))
        self.line_test_img.setText("")
        self.btn_test_img.setText(QCoreApplication.translate("TrainWindow", u"Select", None))
        self.labelTestJson.setText(QCoreApplication.translate("TrainWindow", u"Test LabelMe JSON Path\uff1a", None))
        self.line_test_json.setText("")
        self.btn_test_json.setText(QCoreApplication.translate("TrainWindow", u"Select", None))
        self.groupParams.setTitle(QCoreApplication.translate("TrainWindow", u"Training Setting", None))
        self.labelAlgo.setText(QCoreApplication.translate("TrainWindow", u"Algorithm:", None))
        self.combo_algo.setItemText(0, QCoreApplication.translate("TrainWindow", u"MobileUNet", None))
        self.combo_algo.setItemText(1, QCoreApplication.translate("TrainWindow", u"FastSCNN", None))
        self.combo_algo.setItemText(2, QCoreApplication.translate("TrainWindow", u"LinkNet", None))
        self.combo_algo.setItemText(3, QCoreApplication.translate("TrainWindow", u"UNet_Base", None))
        self.combo_algo.setItemText(4, QCoreApplication.translate("TrainWindow", u"UNet_Medium", None))
        self.combo_algo.setItemText(5, QCoreApplication.translate("TrainWindow", u"UNet_Small", None))
        self.combo_algo.setItemText(6, QCoreApplication.translate("TrainWindow", u"UNet_Tiny", None))
        self.combo_algo.setItemText(7, QCoreApplication.translate("TrainWindow", u"UNext", None))
        self.combo_algo.setItemText(8, QCoreApplication.translate("TrainWindow", u"AttU_Net", None))
        self.combo_algo.setItemText(9, QCoreApplication.translate("TrainWindow", u"NestedUNet", None))
        self.combo_algo.setItemText(10, QCoreApplication.translate("TrainWindow", u"UNetResnet", None))
        self.combo_algo.setItemText(11, QCoreApplication.translate("TrainWindow", u"FCN", None))

        self.labelLoss.setText(QCoreApplication.translate("TrainWindow", u"Loss Function:", None))
        self.combo_loss.setItemText(0, QCoreApplication.translate("TrainWindow", u"CrossEntropyLoss", None))
        self.combo_loss.setItemText(1, QCoreApplication.translate("TrainWindow", u"CE_DiceLoss", None))
        self.combo_loss.setItemText(2, QCoreApplication.translate("TrainWindow", u"FocalLoss", None))
        self.combo_loss.setItemText(3, QCoreApplication.translate("TrainWindow", u"LovaszSoftmax", None))

        self.labelEpochs.setText(QCoreApplication.translate("TrainWindow", u"Epoch:", None))
        self.labelPolicy.setText(QCoreApplication.translate("TrainWindow", u"Label Rendering Policy:", None))
        self.line_label_filter.setPlaceholderText(QCoreApplication.translate("TrainWindow", u"Enter labels (e.g., cat;dog;person)", None))
        self.line_policy.setPlaceholderText(QCoreApplication.translate("TrainWindow", u"Enter policies (e.g., Closure;Follow;Follow)", None))
        ___qtablewidgetitem = self.table_policy.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("TrainWindow", u"Label", None));
        ___qtablewidgetitem1 = self.table_policy.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("TrainWindow", u"Policy", None));
        self.btn_check.setText(QCoreApplication.translate("TrainWindow", u"Check Before Train", None))
        self.btn_start_train.setText(QCoreApplication.translate("TrainWindow", u"Start Training", None))
        self.btn_stop.setText(QCoreApplication.translate("TrainWindow", u"Stop Training", None))
        self.groupLog.setTitle(QCoreApplication.translate("TrainWindow", u"Log", None))
    # retranslateUi

        self.table_policy.horizontalHeader().setStretchLastSection(True)