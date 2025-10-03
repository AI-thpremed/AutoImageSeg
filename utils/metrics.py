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

import torch

def compute_metrics(pred, mask, num_classes, ignore_index=None):
    """
    pred   : (B,C,H,W) softmax logits
    mask   : (B,H,W)   long
    returns dict: {
        'iou_cls':[c], 'dice_cls':[c],
        'miou_with_bg':float, 'miou_no_bg':float,
        'mdice_with_bg':float,'mdice_no_bg':float
    }
    """
    pred = torch.argmax(pred, dim=1).flatten()   # (N,)
    mask = mask.flatten()

    if ignore_index is not None:
        keep = mask != ignore_index
        pred, mask = pred[keep], mask[keep]

    ious, dices = [], []
    for cls in range(num_classes):
        pred_cls = pred == cls
        mask_cls = mask == cls
        inter = (pred_cls & mask_cls).sum().float()
        union = (pred_cls | mask_cls).sum().float()
        iou = (inter / (union + 1e-7)).item()
        dice = (2 * inter / (pred_cls.sum() + mask_cls.sum() + 1e-7)).item()
        ious.append(iou)
        dices.append(dice)

    metrics = {
        'iou_cls': ious,
        'dice_cls': dices,
        'miou_with_bg': torch.tensor(ious).mean().item(),
        'miou_no_bg': torch.tensor(ious[1:]).mean().item() if num_classes > 1 else None,
        'mdice_with_bg': torch.tensor(dices).mean().item(),
        'mdice_no_bg': torch.tensor(dices[1:]).mean().item() if num_classes > 1 else None,
    }
    return metrics