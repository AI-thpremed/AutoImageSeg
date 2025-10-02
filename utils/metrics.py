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