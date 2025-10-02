# infer_worker.py
import os, sys, json, datetime, torch
import json
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from queue import Queue
from tqdm import tqdm
from models.unet import U_Net
from models.mobileunet import MobileUNet
from models.fastscnn import FastSCNN
from models.unext import UNext
from models.attunet import AttU_Net
from models.nestedunet import NestedUNet
from models.unetresnet import UNetResnet
from models.fcn import FCN
from models.linknet import LinkNet
from utils.dataset import SegDataset
from config_manager import ConfigManager




# -------------------- log --------------------
class TqdmToLog:
    def __init__(self, log_func):
        self.log_func = log_func

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.log_func(line.rstrip())

    def flush(self):
        pass



# -------------------- General Utilities --------------------
def imread_rgb(p):
    """
    Uniformly use PIL to read images, ensuring the channel order is RGB.

    :param p: Path to the image file.
    :return: Image as a NumPy array with RGB channel order.
    """
    from PIL import Image
    img = Image.open(p).convert('RGB')
    return np.array(img)


def infer_worker(cfg: dict, log_q: Queue):
    """
    cfg 字段:
        model_dir : str  训练产出目录（含 best.pth / training_config.json / label_mapping.json）
        img_dir   : str  待推理图片目录
        mode      : "infer_only" | "infer_eval"   本函数只实现前者
    """
    config_manager = ConfigManager(config_path="config.json")

    config = config_manager.config

    image_size = config["common"].get("image_size", 512)

    log_file = None

    def log(msg, to_ui=True):
        """
        to_ui=True 时才往 UI 队列发；
        所有日志都会写文件与控制台。
        """
        if msg is None:
            log_q.put(None)
            if log_file:
                log_file.close()
            return

        if to_ui:
            log_q.put(msg)

        if log_file:
            log_file.write(msg + '\n')
            log_file.flush()

        tqdm.write(msg)

    root_cache   = Path(__file__).parent / 'cache'
    root_results = Path(__file__).parent / 'results'
    root_cache.mkdir(exist_ok=True)
    root_results.mkdir(exist_ok=True)

    timestamp   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    res_dir     = root_results / f'infer_project_{timestamp}'
    res_dir.mkdir(exist_ok=True)

    cfg_save_path = res_dir / 'inference_config.json'
    with open(cfg_save_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)

    log_path = res_dir / 'inference.log'
    log_file = open(log_path, 'a', encoding='utf-8')



    try:
        log("=== start ===")

        train_cfg_path = Path(cfg['model_dir']) / 'training_config.json'
        with open(train_cfg_path, encoding='utf-8') as f:
            train_cfg = json.load(f)

        algo = train_cfg['algo']

        # Directly reuse label-mappingjson during training
        map_path = Path(cfg['model_dir']) / 'label_mapping.json'
        with open(map_path) as f:
            label2id = json.load(f)  # {"background":0,"cat":1,"dog":2"}

        destination_path = res_dir / 'label_mapping.json'  #
        with open(destination_path, 'w', encoding='utf-8') as f:
            json.dump(label2id, f, indent=4, ensure_ascii=False)

        id2label = {v: k for k, v in label2id.items()}
        num_classes = len(label2id)
        class_names = [id2label[i] for i in range(num_classes)]
        log(f"Label id map load success：{label2id}")


        model_map = {'UNet': U_Net, 'MobileUNet': MobileUNet,
                     'FastSCNN': FastSCNN, 'UNext': UNext, 'AttU_Net': AttU_Net,
                     'NestedUNet': NestedUNet, 'UNetResnet': UNetResnet,
                     'FCN': FCN, 'LinkNet': LinkNet}
        ModelCls = model_map[algo]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model  = ModelCls(in_ch=3, out_ch=num_classes).to(device)

        best_path = Path(cfg['model_dir']) / 'best.pth'
        ckpt = torch.load(best_path, map_location='cpu')
        model.load_state_dict(ckpt)
        model.eval()
        log("Model load success")


        dataset = SegDataset(img_dir=cfg['img_dir'], mask_dir=None, config=config, img_size=image_size, val=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        out_root = res_dir / 'inference_results'
        out_root.mkdir(exist_ok=True)


        total = len(dataset)
        report_every = max(1, total // 10)



        with torch.no_grad():
            for idx, (image, _,original_sizes) in enumerate(tqdm(dataloader, file=TqdmToLog(log))):

                image = image.to(device)  # 确保 image 是一个张量
                logits = model(image)  # [1,C,H,W]
                pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)  # [H,W]
                original_height = original_sizes[0].item()  # 将 Tensor 转换为 Python 的原生整数
                original_width = original_sizes[1].item()  # 将 Tensor 转换为 Python 的原生整数

                pred_resized = cv2.resize(pred, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

                img_path = dataset.ids[idx]
                mask_path = out_root / f"{img_path}.png"
                cv2.imwrite(str(mask_path), pred_resized)


                if (idx + 1) % report_every == 0 or idx == total - 1:
                    progress = (idx + 1) / total * 100
                    log(f"Inference progress {idx + 1}/{total} ({progress:.0f}%)")

        log("=== Inference completed ===")
        log(f"Results have been saved to: {out_root}")

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        log(f"Inference error: {e}\n{tb}", to_ui=True)
        log(None)

