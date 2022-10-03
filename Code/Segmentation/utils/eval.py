import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.val_metrics import val_dice_coeff



def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                mask_pred = net(imgs)
            
            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred[-1], true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred[-1])
                pred = (pred > 0.5).float()
                tot += val_dice_coeff(pred, true_masks).item()
            pbar.update()

    net.train()
    return tot / n_val

def eval_net_w(net, loader, device):
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch',
              leave=False) as pbar:
        for batch in loader:
            imgs = batch["image"]
            mask_full = batch["mask_full"]
            mask_portal = batch["mask_portal"]

            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_full = mask_full.to(device=device, dtype=mask_type)
            mask_portal = mask_portal.to(device=device, dtype=mask_type)
            with torch.no_grad():
                mask_full_pred, mask_portal_pred = net(imgs)

            pred1 = torch.sigmoid(mask_full_pred)
            pred2 = torch.sigmoid(mask_portal_pred)
            pred1 = (pred1 > 0.5).float()
            pred2 = (pred2 > 0.5).float()

            tot += val_dice_coeff(pred1, mask_full).item()
            tot += val_dice_coeff(pred2, mask_portal).item()

            pbar.update()

    net.train()
    return tot / n_val