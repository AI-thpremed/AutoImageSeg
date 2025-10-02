import os
import json
import cv2
import numpy as np
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QApplication, QFileDialog, QMessageBox, QTableWidgetItem, QProgressDialog
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QStandardItemModel, QStandardItem
from datetime import datetime
from ui_eval import Ui_EvalForm
from prepare_masks import prepare_masks_for_eval
from config_manager import ConfigManager


class EvalWindow(QWidget, Ui_EvalForm):
    def __init__(self,config: ConfigManager):
        super().__init__()
        self.setupUi(self)
        self.config = config

        # Bind and select folder button
        self.btnSelectJson.clicked.connect(lambda: self._select_dir(self.lineTrainJson))
        self.btnSelectMask1.clicked.connect(lambda: self._select_dir(self.lineMask1))
        self.btnSelectMask2.clicked.connect(lambda: self._select_dir(self.lineMask2))

        self.btnLabelme2Mask.clicked.connect(self.on_labelme_to_mask)
        self.btnEval.clicked.connect(self.on_mask_eval)


    def _select_dir(self, line):
        folder = QFileDialog.getExistingDirectory(self, "Select")
        if folder:
            line.setText(folder)

    def _refresh_policy_table(self, labels: list, policies: list):
        self.tablePolicy.setRowCount(len(labels))
        for row, (lab, pol) in enumerate(zip(labels, policies)):
            self.tablePolicy.setItem(row, 0, QTableWidgetItem(lab))
            self.tablePolicy.setItem(row, 1, QTableWidgetItem(pol))





    def on_labelme_to_mask(self):
        train_json_dir = self.lineTrainJson.text().strip()
        label_filter = self.lineLabelFilter.text().strip()
        policies = self.linePolicy.text().strip()

        if not train_json_dir:
            QMessageBox.warning(self, "Error", "JSON path can not be empty")
            return

        if not label_filter or not policies:
            QMessageBox.warning(self, "Error", "Label and Policy can not be empty")
            return

        labels_to_use = [l.strip() for l in label_filter.split(";") if l.strip()]
        policies_items = [p.strip() for p in policies.split(';') if p.strip()]

        if len(labels_to_use) != len(policies_items):
            QMessageBox.warning(self, "Error", "The number of labels and policies does not match.")
            return

        root_results = Path(__file__).parent / 'results'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        res_dir = root_results / f'eval_{timestamp}'
        res_dir.mkdir(exist_ok=True)

        cfg = {
            'train_json': train_json_dir,
            'results': str(res_dir),
            'labels_to_use': label_filter,
            'policies': policies
        }
        self._refresh_policy_table(labels_to_use, policies_items)

        prepare_masks_for_eval(cfg)

        QMessageBox.information(self, "Finished", f"Mask images are saved into {res_dir}")



    def on_mask_eval(self):
        mask1_dir = self.lineMask1.text().strip()
        mask2_dir = self.lineMask2.text().strip()

        if not mask1_dir or not mask2_dir:
            QMessageBox.warning(self, "Error", "The paths for the two mask directories cannot be empty.")
            return

        mask1_dir = Path(mask1_dir)
        mask2_dir = Path(mask2_dir)

        if not mask1_dir.is_dir() or not mask2_dir.is_dir():
            QMessageBox.warning(self, "Error", "mask path invalid")
            return

        mask1_files = list(mask1_dir.glob("*.png"))
        mask2_files = list(mask2_dir.glob("*.png"))

        if not mask1_files or not mask2_files:
            QMessageBox.warning(self, "Error", "mask path has no .png images")
            return

        # Check if the number of files and file names correspond one-to-one
        mask1_names = {f.name for f in mask1_files}
        mask2_names = {f.name for f in mask2_files}

        if mask1_names != mask2_names:
            QMessageBox.warning(self, "Error",
                                "The number of files or filenames in the two mask directories do not match.")
            return

        results_per_image = []
        results_summary = {}

        for mask1_file in mask1_files:
            mask2_file = mask2_dir / mask1_file.name

            mask1 = cv2.imread(str(mask1_file), cv2.IMREAD_GRAYSCALE)
            mask2 = cv2.imread(str(mask2_file), cv2.IMREAD_GRAYSCALE)

            if mask1 is None or mask2 is None:
                QMessageBox.warning(self, "Error", f"unable to read mask path: {mask1_file} or {mask2_file}")
                return

            # 计算 IoU 和 Dice
            unique_values = np.unique(mask1)
            for val in unique_values:
                if val == 0:
                    continue
                mask1_bin = (mask1 == val).astype(np.uint8)
                mask2_bin = (mask2 == val).astype(np.uint8)

                intersection = np.sum(mask1_bin * mask2_bin)
                union = np.sum(mask1_bin) + np.sum(mask2_bin) - intersection
                iou = intersection / union if union != 0 else 0

                dice = 2 * intersection / (np.sum(mask1_bin) + np.sum(mask2_bin)) if (np.sum(mask1_bin) + np.sum(
                    mask2_bin)) != 0 else 0

                # 保存每个图片的每个标签的结果
                results_per_image.append((mask1_file.name, val, iou, dice))

                # 累加每个标签的结果
                if val not in results_summary:
                    results_summary[val] = {'iou_sum': 0, 'dice_sum': 0, 'count': 0}
                results_summary[val]['iou_sum'] += iou
                results_summary[val]['dice_sum'] += dice
                results_summary[val]['count'] += 1

        for val in results_summary:
            results_summary[val]['iou_avg'] = results_summary[val]['iou_sum'] / results_summary[val]['count']
            results_summary[val]['dice_avg'] = results_summary[val]['dice_sum'] / results_summary[val]['count']

        root_results = Path(__file__).parent / 'results'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        res_dir = root_results / f'eval_project_{timestamp}'
        res_dir.mkdir(exist_ok=True)

        per_image_file = res_dir / 'evaluation_results_per_image.txt'
        with open(per_image_file, 'w') as f:
            f.write("image,label,IoU,Dice\n")
            for file, val, iou, dice in results_per_image:
                f.write(f"{file},{val},{iou:.3f},{dice:.3f}\n")

        # Save the average IoU and Dice for each tag
        summary_file = res_dir / 'evaluation_results_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("label,IoU_avg,Dice_avg\n")
            for val in results_summary:
                f.write(f"{val},{results_summary[val]['iou_avg']:.3f},{results_summary[val]['dice_avg']:.3f}\n")

        QMessageBox.information(self, "Evaluation Complete", f"The evaluation results have been saved to {res_dir}")
