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

import os
import json
import cv2
import numpy as np
from typing import Dict, List, Any
from pathlib import Path
from PySide6.QtWidgets import QMessageBox, QApplication
from PIL import Image
import shutil


def shapes_to_label_mask(json_path: str,
                         label2id: Dict[str, int],
                         policy_map: Dict[str, str],
                         img_h: int,
                         img_w: int) -> np.ndarray:
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    for shape in data.get('shapes', []):
        label = shape['label']
        if label not in label2id:
            continue

        policy = policy_map.get(label, 'Closure')
        shape_type = shape.get('shape_type', 'polygon')
        pts = np.array(shape['points'], dtype=np.float32)
        color = label2id[label]

        if policy == 'Closure':
            # 圆 -> 直接画实心
            if shape_type == 'circle':
                cx, cy = pts[0]
                px, py = pts[1]
                radius = int(np.round(np.linalg.norm([px - cx, py - cy])))
                cv2.circle(mask, (int(cx), int(cy)), radius, color, -1)
                continue

            # 其余全部强制成多边形
            if len(pts) < 3:
                continue
            pts_i32 = pts.astype(np.int32)
            if not np.array_equal(pts_i32[0], pts_i32[-1]):
                pts_i32 = np.vstack([pts_i32, pts_i32[0]])
            if cv2.contourArea(pts_i32) >= 5:
                cv2.fillPoly(mask, [pts_i32], color=color)

        elif policy == 'Follow':
            pts_i32 = pts.astype(np.int32)

            if shape_type == 'linestrip' and len(pts_i32) >= 2:
                cv2.polylines(mask, [pts_i32], isClosed=False, color=color, thickness=2)

            elif shape_type == 'polygon' and len(pts_i32) >= 3:
                if not np.array_equal(pts_i32[0], pts_i32[-1]):
                    pts_i32 = np.vstack([pts_i32, pts_i32[0]])
                if cv2.contourArea(pts_i32) >= 5:
                    cv2.fillPoly(mask, [pts_i32], color=color)

            elif shape_type == 'rectangle' and len(pts_i32) == 2:
                x1, y1, x2, y2 = map(int, [*pts_i32[0], *pts_i32[1]])
                cv2.rectangle(mask, (x1, y1), (x2, y2), color, -1)

            elif shape_type == 'circle' and len(pts_i32) == 2:
                cx, cy = pts_i32[0]
                px, py = pts_i32[1]
                radius = int(np.round(np.linalg.norm([px - cx, py - cy])))
                cv2.circle(mask, (cx, cy), radius, color, -1)

            # point 忽略
    return mask



def prepare_masks_and_mapping(cfg: Dict[str, Any]) -> None:
    """
    cfg :
        train_json, test_json, cache
    """
    train_json_dir = cfg['train_json']
    test_json_dir  = cfg['test_json']
    res_dir      = cfg.get('results', 'results')
    cache_dir=cfg.get('cache', 'cache')



    labels_raw = cfg.get("labels_to_use", "")
    labels_to_use = [l.strip() for l in labels_raw.split(";") if l.strip()]

    policies_str = cfg["policies"]
    policies = [p.strip() for p in policies_str.split(';') if p.strip()]
    policy_map = dict(zip(labels_to_use, policies))

    os.makedirs(res_dir, exist_ok=True)

    mapping_file = os.path.join(res_dir, 'label_mapping.json')

    if labels_to_use:
        label_set = set(labels_to_use)
    else:
        label_set = set()
        for root in [train_json_dir, test_json_dir]:
            for js in Path(root).rglob('*.json'):
                with open(js, encoding='utf-8') as f:
                    data = json.load(f)
                for s in data.get('shapes', []):
                    if s['shape_type'] != 'point':
                        label_set.add(s['label'])


    label_set = {lbl for lbl in label_set if lbl in labels_to_use or not labels_to_use}

    label2id = {lbl: idx + 1 for idx, lbl in enumerate(label_set)}
    label2id['_background_'] = 0


    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    print('[prepare] label -> id:', label2id)

    mask_dir = os.path.join(cache_dir, 'train_masks')
    os.makedirs(mask_dir, exist_ok=True)
    for js in Path(train_json_dir).rglob('*.json'):
        data = json.load(open(js, encoding='utf-8'))
        h, w = data['imageHeight'], data['imageWidth']
        mask = shapes_to_label_mask(str(js), label2id, policy_map, h, w)
        out_path = os.path.join(mask_dir, js.stem + '.png')
        cv2.imwrite(out_path, mask)

    mask_dir = os.path.join(cache_dir, 'test_masks')
    os.makedirs(mask_dir, exist_ok=True)
    for js in Path(test_json_dir).rglob('*.json'):
        data = json.load(open(js, encoding='utf-8'))
        h, w = data['imageHeight'], data['imageWidth']
        mask = shapes_to_label_mask(str(js), label2id,  policy_map,h, w)
        out_path = os.path.join(mask_dir, js.stem + '.png')
        cv2.imwrite(out_path, mask)

    print('[prepare] masks & mapping ready in', cache_dir)


def prepare_masks_and_mapping_mask(cfg: Dict[str, Any]) -> None:
    try:
        train_mask_dir = cfg['train_mask']
        test_mask_dir = cfg['test_mask']
        line_json = cfg['line_json']
        res_dir = cfg.get('results', 'results')
        cache_dir = cfg.get('cache', 'cache')

        os.makedirs(res_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)

        with open(line_json, 'r') as f:
            label2id = json.load(f)

        with open(os.path.join(res_dir, 'label_mapping.json'), 'w') as f:
            json.dump(label2id, f, indent=4)
        print('[prepare] label -> id:', label2id)

        train_masks_dir = os.path.join(cache_dir, 'train_masks')
        test_masks_dir = os.path.join(cache_dir, 'test_masks')
        os.makedirs(train_masks_dir, exist_ok=True)
        os.makedirs(test_masks_dir, exist_ok=True)

        for mask_dir, target_dir in [(train_mask_dir, train_masks_dir), (test_mask_dir, test_masks_dir)]:
            for mask_file in os.listdir(mask_dir):
                mask_path = os.path.join(mask_dir, mask_file)
                target_path = os.path.join(target_dir, mask_file)

                with Image.open(mask_path) as img:
                    if img.mode != 'L':
                        raise ValueError(f"Mask image {mask_file} is not a grayscale image.")

                    unique_pixels = set(img.getdata())
                    for pixel in unique_pixels:
                        if pixel not in label2id.values():
                            raise ValueError(f"Pixel value {pixel} in mask {mask_file} is not in label2id mapping.")

                shutil.copy(mask_path, target_path)

        for label, id in label2id.items():
            if id != 0:
                found = False
                for mask_dir in [train_masks_dir, test_masks_dir]:
                    for mask_file in os.listdir(mask_dir):
                        with Image.open(os.path.join(mask_dir, mask_file)) as img:
                            if id in img.getdata():
                                found = True
                                break
                        if found:
                            break
                if not found:
                    raise ValueError(f"Label {label} (id {id}) is in label2id but not found in any mask images.")

        print('[prepare] masks & mapping ready in', cache_dir)

    except Exception as e:
        QMessageBox.critical(None, "Error", f"An error occurred: {e}")
        raise e


def prepare_masks_for_eval(cfg: Dict[str, Any]) -> None:
    """
    cfg :
        train_json, test_json, cache
    """
    train_json_dir = cfg['train_json']
    res_dir      = cfg.get('results', 'results')
    labels_raw = cfg.get("labels_to_use", "")
    policies_str = cfg["policies"]

    labels_to_use = [l.strip() for l in labels_raw.split(";") if l.strip()]

    policies = [p.strip() for p in policies_str.split(';') if p.strip()]


    policy_map = dict(zip(labels_to_use, policies))

    os.makedirs(res_dir, exist_ok=True)

    mapping_file = os.path.join(res_dir, 'label_mapping.json')

    label_set = set(labels_to_use)

    label2id = {lbl: idx + 1 for idx, lbl in enumerate(label_set)}
    label2id['_background_'] = 0
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)

    print('[prepare] label -> id:', label2id)

    # ---------- build train mask ----------
    mask_dir = os.path.join(res_dir, 'train_masks')
    os.makedirs(mask_dir, exist_ok=True)
    for js in Path(train_json_dir).rglob('*.json'):
        data = json.load(open(js, encoding='utf-8'))
        h, w = data['imageHeight'], data['imageWidth']
        mask = shapes_to_label_mask(str(js), label2id, policy_map, h, w)
        out_path = os.path.join(mask_dir, js.stem + '.png')
        cv2.imwrite(out_path, mask)
    print('[prepare] masks & mapping ready in', res_dir)