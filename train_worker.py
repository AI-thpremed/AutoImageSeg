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

# train_worker.py
import os, sys, json, datetime, torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import pandas as pd
from utils.dataset import SegDataset
# models
from models.unet import U_Net
from models.mobileunet import MobileUNet
from models.fastscnn import FastSCNN
from models.unext import UNext
from models.attunet import AttU_Net
from models.nestedunet import NestedUNet
from models.unetresnet import UNetResnet
from models.fcn import FCN
from models.linknet import LinkNet
from prepare_masks import prepare_masks_and_mapping
from utils.metrics import compute_metrics
from utils.losses import CE_DiceLoss, LovaszSoftmax, FocalLoss   # 你的文件
from config_manager import ConfigManager
import random
import numpy as np
import torch.backends.cudnn as cudnn

class TqdmToLog:
    def __init__(self, log_func):
        self.log_func = log_func

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.log_func(line.rstrip())

    def flush(self):
        pass


def train_worker(cfg: dict, log_q):


    config_manager = ConfigManager(config_path="config.json")

    config = config_manager.config  #  config_manager.config

    image_size = config["common"].get("image_size", 256)

    # random seed
    seed = config["common"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    log_file = None

    def log(msg, level="INFO"):
        if msg is None:
            log_q.put(None)
            if log_file:
                log_file.close()
            return

        if level == "INFO":
            log_q.put(msg)

        if log_file:
            log_file.write(msg + '\n')
            log_file.flush()

        tqdm.write(msg)  # 控制台可见

    root_cache   = Path(__file__).parent / 'cache'
    root_results = Path(__file__).parent / 'results'
    root_cache.mkdir(exist_ok=True)
    root_results.mkdir(exist_ok=True)

    timestamp   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    project_dir = root_cache   / f'train_project_{timestamp}'
    res_dir     = root_results / f'train_project_{timestamp}'
    project_dir.mkdir(exist_ok=True)
    res_dir.mkdir(exist_ok=True)

    cfg['cache']   = str(project_dir)
    cfg['results'] = str(res_dir)


    cfg_save_path = res_dir / 'training_config.json'
    with open(cfg_save_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)

    log_file_path = res_dir / 'training.log'
    log_file = open(log_file_path, 'a', encoding='utf-8')




    train_loss_df = pd.DataFrame(columns=['epoch', 'train_loss'])
    val_loss_df = pd.DataFrame(columns=['epoch', 'val_loss'])

    try:
        prepare_masks_and_mapping(cfg)



        mapping_path = res_dir / 'label_mapping.json'
        with open(mapping_path) as f:
            label_map  = json.load(f)
        class_names  = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
        num_classes  = len(class_names)



        train_ds = SegDataset(cfg['train_img'], project_dir / 'train_masks',config,image_size,val=False)
        val_ds   = SegDataset(cfg['test_img'],  project_dir / 'test_masks',config,image_size,val=True)
        train_dl = DataLoader(train_ds, batch_size=config["training"]["dataloader"]["batch_size"], shuffle=config["training"]["dataloader"]["shuffle"],  num_workers=0)
        val_dl   = DataLoader(val_ds,   batch_size=config["training"]["dataloader"]["batch_size"], shuffle=False, num_workers=0)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_map = {'UNet': U_Net, 'MobileUNet': MobileUNet,
                     'FastSCNN': FastSCNN, 'UNext': UNext,'AttU_Net': AttU_Net,
                     'NestedUNet': NestedUNet,'UNetResnet': UNetResnet,
                     'FCN': FCN,'LinkNet': LinkNet}

        ModelCls  = model_map[cfg['algo']]
        model     = ModelCls(in_ch=3, out_ch=num_classes).to(device)

        loss_map = {
            'CrossEntropyLoss': torch.nn.CrossEntropyLoss(),
            'CE_DiceLoss': CE_DiceLoss(),
            'FocalLoss': FocalLoss(gamma=2),
            'LovaszSoftmax': LovaszSoftmax()
        }
        loss_fn = loss_map[cfg['loss']]
        lr = config["training"].get("lr", 1e-4)
        optimizer = Adam(model.parameters(), lr=lr)
        evemetric = config["training"].get("evemetric", "iou")  # default as  "iou"
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
        earlystop = config["training"].get("earlystop", 0)


        if not isinstance(earlystop, int):
            try:
                earlystop = int(earlystop)
            except (ValueError, TypeError):
                earlystop = 0
                print(f"Warning: earlystop value '{earlystop}' is not a valid integer. Using default value 0.")

        if earlystop>cfg['epochs'] or earlystop<0:
            earlystop=0

        best_miou_no_bg = 0.0
        best_mdice_no_bg = 0.0
        early_stop_counter = 0

        for epoch in range(cfg['epochs']):
            model.train()

            running_loss = 0.0
            total_batches = len(train_dl)

            pbar = tqdm(train_dl,
                        desc=f'Epoch {epoch+1}/{cfg["epochs"]} [TRAIN]',
                        file=sys.stdout)

            for batch_idx, (img, msk,_) in enumerate(pbar):
                img, msk = img.to(device), msk.to(device)
                pred = model(img)
                loss = loss_fn(pred, msk)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix(loss=f'{loss.item():.4f}')

                # ✅ 每 10% 或最后一步，向 UI 汇报
                if (batch_idx + 1) % max(1, total_batches // 10) == 0 or batch_idx == total_batches - 1:
                    progress = (batch_idx + 1) / total_batches * 100
                    log(f'Train Epoch {epoch+1}/{cfg["epochs"]} | {batch_idx+1}/{total_batches} batches ({progress:.0f}%) | loss {loss.item():.4f}')

            scheduler.step()
            avg_train_loss = running_loss / total_batches
            log(f'Epoch {epoch+1:03d} | train_loss {avg_train_loss:.4f}')
            # train_loss_df = train_loss_df.append({'epoch': epoch + 1, 'train_loss': avg_train_loss}, ignore_index=True)
            new_row = pd.DataFrame({'epoch': [epoch + 1], 'train_loss': [avg_train_loss]})
            train_loss_df = pd.concat([train_loss_df, new_row], ignore_index=True)

            model.eval()
            val_loss, count = 0, 0
            metrics_sum = {k: [0]*num_classes if k.endswith('_cls') else 0
                           for k in ['iou_cls','dice_cls','miou_with_bg',
                                     'miou_no_bg','mdice_with_bg','mdice_no_bg']}
            total_val_batches = len(val_dl)

            pbar = tqdm(val_dl,
                        desc=f'Epoch {epoch+1}/{cfg["epochs"]} [VAL]',
                        file=sys.stdout)

            with torch.no_grad():
                for batch_idx, (img, msk,_) in enumerate(pbar):
                    img, msk = img.to(device), msk.to(device)
                    pred = model(img)
                    val_loss += loss_fn(pred, msk).item()
                    count += 1
                    batch_met = compute_metrics(pred, msk, num_classes)
                    for k in ['iou_cls', 'dice_cls']:
                        metrics_sum[k] = [a+b for a,b in zip(metrics_sum[k], batch_met[k])]
                    for k in ['miou_with_bg','miou_no_bg','mdice_with_bg','mdice_no_bg']:
                        if batch_met[k] is not None:
                            metrics_sum[k] += batch_met[k]

                    pbar.set_postfix(loss=f'{val_loss/count:.4f}')

                    # ✅ every 10% report to UI
                    if (batch_idx + 1) % max(1, total_val_batches // 10) == 0 or batch_idx == total_val_batches - 1:
                        progress = (batch_idx + 1) / total_val_batches * 100
                        log(f'Val Epoch {epoch+1} | {batch_idx+1}/{total_val_batches} batches ({progress:.0f}%)')

            val_loss /= count
            for k in ['iou_cls', 'dice_cls']:
                metrics_sum[k] = [v/count for v in metrics_sum[k]]
            for k in ['miou_with_bg','miou_no_bg','mdice_with_bg','mdice_no_bg']:
                if isinstance(metrics_sum[k], (int, float)):
                    metrics_sum[k] /= count

            log_str = f"Epoch {epoch+1:03d} | val_loss {val_loss:.4f}\n"
            # val_loss_df = val_loss_df.append({'epoch': epoch + 1, 'val_loss': val_loss}, ignore_index=True)
            new_row = pd.DataFrame({'epoch': [epoch + 1], 'val_loss': [val_loss]})
            val_loss_df = pd.concat([val_loss_df, new_row], ignore_index=True)

            for cls, iou, dice in zip(class_names,
                                      metrics_sum['iou_cls'],
                                      metrics_sum['dice_cls']):
                log_str += f"  {cls:<10} IoU={iou:.4f} Dice={dice:.4f}\n"
            log_str += (f"  Mean(w/ bg) IoU={metrics_sum['miou_with_bg']:.4f} "
                        f"Dice={metrics_sum['mdice_with_bg']:.4f}\n")
            if metrics_sum['miou_no_bg'] is not None:
                log_str += (f"  Mean(no bg) IoU={metrics_sum['miou_no_bg']:.4f} "
                            f"Dice={metrics_sum['mdice_no_bg']:.4f}\n")
            log(log_str)



            if evemetric=="iou":
                miou_no_bg = metrics_sum['miou_no_bg'] or 0
                if miou_no_bg > best_miou_no_bg:
                    best_miou_no_bg = miou_no_bg
                    best_path = res_dir / 'best.pth'
                    torch.save(model.state_dict(), best_path)
                    log(f"  √ Saved best model (mIoU_no_bg={best_miou_no_bg:.4f})")
                    early_stop_counter = 0

            else:
                mdice_no_bg = metrics_sum['mdice_no_bg'] or 0
                if mdice_no_bg > best_mdice_no_bg:
                    best_mdice_no_bg = mdice_no_bg
                    best_path = res_dir / 'best.pth'
                    torch.save(model.state_dict(), best_path)
                    log(f"  √ Saved best model (mdice_no_bg={best_mdice_no_bg:.4f})")

                    early_stop_counter = 0


            early_stop_counter += 1

            if earlystop > 0 and early_stop_counter >= earlystop:
                log(f"Early stopping triggered after {earlystop} epochs without improvement.")
                break

        train_loss_csv_path = res_dir / 'train_loss.csv'
        val_loss_csv_path = res_dir / 'val_loss.csv'

        train_loss_df.to_csv(train_loss_csv_path, index=False)
        val_loss_df.to_csv(val_loss_csv_path, index=False)

        log("Finished training")
    except Exception as e:
        log(f"Training error:{e}")
        raise
    finally:
        log(None)
