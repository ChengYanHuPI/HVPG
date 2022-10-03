import argparse
import logging
import os
import sys
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from utils.eval import eval_net
from unet import UNet, UNet_2, UNet_3, UNet_4
from utils.loss import FocalTversky_BCELoss, FocalTverskyLoss
from prefetch_generator import BackgroundGenerator
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

from torchsummary import summary
# choose the gpu id
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device("cuda")
# the training set & the training label

# model saving path
file_name = os.path.basename(__file__).split('.')[0]
dir_checkpoint = f"checkpoints/{file_name}/"


def train_net(net, device, epochs, batch_size, lr, val_percent, save_cp, n_classes, multi_sv):
    dataset = BasicDataset(dir_img, dir_mask, n_classes, transform=True)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)


    # Change log name during training
    writer = SummaryWriter(comment=f"_{file_name}")
    global_step = 0

    logging.info(f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    """)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 'min' if net.n_classes > 1 else 'max', patience=5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # Fixed step decay
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = FocalTversky_BCELoss().cuda()
        # criterion = FocalTverskyLoss().cuda()
        # criterion = nn.BCEWithLogitsLoss().cuda()

    for epoch in range(epochs):
        net.train()
        with tqdm(total=n_train, desc=f"Epoch {epoch + 1}/{epochs}", unit="img") as pbar:
            for batch in train_loader:
                imgs = batch["image"]
                true_masks = batch["mask"]

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                # Forward Propagation
                masks_pred = net(imgs)
                # Calculating Loss
                if multi_sv == 1:
                    loss1 = criterion(masks_pred[0], true_masks)
                    writer.add_scalar("Train/loss1", loss1.item(), global_step)
                    pbar.set_postfix({"loss1 (batch)": loss1.item()})
                    #Clear the previously stored gradients
                    optimizer.zero_grad()
                    loss1.backward()
                if multi_sv == 2:
                    loss1 = criterion(masks_pred[0], true_masks)
                    loss2 = criterion(masks_pred[1], true_masks)
                    writer.add_scalar("Train/loss1", loss1.item(), global_step)
                    writer.add_scalar("Train/loss2", loss2.item(), global_step)
                    pbar.set_postfix({"loss1 (batch)": loss1.item(), "loss2 (batch)": loss2.item()})
                    optimizer.zero_grad()
                    loss1.backward(retain_graph=True)
                    loss2.backward()
                if multi_sv == 3:
                    loss1 = criterion(masks_pred[0], true_masks)
                    loss2 = criterion(masks_pred[1], true_masks)
                    loss3 = criterion(masks_pred[2], true_masks)
                    writer.add_scalar("Train/loss1", loss1.item(), global_step)
                    writer.add_scalar("Train/loss2", loss2.item(), global_step)
                    writer.add_scalar("Train/loss3", loss3.item(), global_step)
                    pbar.set_postfix({
                        "loss1 (batch)": loss1.item(),
                        "loss2 (batch)": loss2.item(),
                        "loss3 (batch)": loss3.item()
                    })
                    optimizer.zero_grad()
                    loss1.backward(retain_graph=True)
                    loss2.backward(retain_graph=True)
                    loss3.backward()
                if multi_sv == 4:
                    loss1 = criterion(masks_pred[0], true_masks)
                    loss2 = criterion(masks_pred[1], true_masks)
                    loss3 = criterion(masks_pred[2], true_masks)
                    loss4 = criterion(masks_pred[3], true_masks)
                    writer.add_scalar("Train/loss1", loss1.item(), global_step)
                    writer.add_scalar("Train/loss2", loss2.item(), global_step)
                    writer.add_scalar("Train/loss3", loss3.item(), global_step)
                    writer.add_scalar("Train/loss4", loss4.item(), global_step)
                    pbar.set_postfix({
                        "loss1 (batch)": loss1.item(),
                        "loss2 (batch)": loss2.item(),
                        "loss3 (batch)": loss3.item(),
                        "loss4 (batch)": loss4.item()
                    })
                    optimizer.zero_grad()
                    loss1.backward(retain_graph=True)
                    loss2.backward(retain_graph=True)
                    loss3.backward(retain_graph=True)
                    loss4.backward()
                # Gradient cropping
                nn.utils.clip_grad_value_(net.parameters(), 0.01)
                # Applying gradients to parameter updates
                optimizer.step()
                pbar.update(imgs.shape[0])
                global_step += 1

                if global_step % (n_train // (4 * batch_size)) == 0:
                    val_score = eval_net(net, val_loader, device)
                    writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch + 1)
                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/val', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/val', val_score, global_step)
        scheduler.step()
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info("Created checkpoint directory")
            except OSError:
                pass
            torch.save(net.state_dict(), dir_checkpoint + f"CP_epoch{epoch + 1}_{val_score}.pth")
            logging.info(f"Checkpoint {epoch + 1} saved !")

    writer.close()


def weight_init(m):
    for m in m.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight.data)


def get_args():
    parser = argparse.ArgumentParser(description='Train the Model on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i',
                        '--input_path',
                        type=str,
                        default='/',
                        help='input np.array data path',
                        dest='input_path')
    parser.add_argument('-ia',
                        '--label_path',
                        type=str,
                        default='/',
                        help='input np.array label path',
                        dest='label_path')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=300, help='Number of epochs', dest='epochs')
    parser.add_argument('-b',
                        '--batch-size',
                        metavar='B',
                        type=int,
                        nargs='?',
                        default=32,
                        help='Batch size',
                        dest='batchsize')
    parser.add_argument('-l',
                        '--learning-rate',
                        metavar='LR',
                        type=float,
                        nargs='?',
                        default=0.0001,
                        help='Learning rate',
                        dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default='', help='Load model from a .pth file')
    parser.add_argument('-v',
                        '--validation',
                        dest='val',
                        type=float,
                        default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-m',
                        '--model_type',
                        dest='model_type',
                        type=str,
                        default=UNet_4,
                        help='the training model type')
    parser.add_argument('-ca', '--channels', dest='channels', type=int, default=5, help='the input slices(channels)')
    parser.add_argument('-cs', '--classes', dest='classes', type=int, default=1, help='the classes the model outputs')
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.info(f"Using device {device}")
    args = get_args()
    dir_img = args.input_path
    dir_mask = args.label_path
    # choose the model type
    net = args.model_type(n_channels=args.channels, n_classes=args.classes, bilinear=False)
    multi_sv = int(str(args.model_type).split('_')[-1][:-2])
    # the parameters initialization
    net.apply(weight_init)
    net.to(device=device)

    # record the info
    logging.info(f"Network:\n"
                 f"\t{net.n_channels} input channels\n"
                 f"\t{net.n_classes} output channels (classes)")

    # load the model to finetune
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f"Model loaded from {args.load}")

    # start to train the model
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  save_cp=True,
                  val_percent=args.val / 100,
                  n_classes=args.classes,
                  multi_sv=multi_sv)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
