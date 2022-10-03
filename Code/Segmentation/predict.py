import argparse
import logging
import os
import numpy as np
import time
import torch
import torch.nn.functional as F
from torchvision import transforms
from unet import UNet, UNet_2, UNet_3, UNet_4
from utils.dataset import BasicDataset
import nibabel as nib
from preprocess import PreProcessing
from mitk import npy_regionprops_denoise, nii_resample

# print(torch.__version__)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device {device}')


class ModelPredict(PreProcessing):
    def __init__(self,
                 model_type,
                 classes,
                 model,
                 ornt,
                 spacing,
                 shape,
                 data_type,
                 threshold=0.1,
                 channels=5,
                 *args,
                 **kwargs):
        super().__init__(ornt, spacing, shape, data_type, *args, **kwargs)
        net = model_type(n_channels=channels, n_classes=classes, bilinear=False)
        net.to(device=device)
        net.load_state_dict(torch.load(model, map_location=device), strict=False)
        self.net = net
        self.threshold = threshold

    def predict_image_slice(self, full_img):
        net = self.net
        out_threshold = self.threshold
        net.eval()
        img = torch.from_numpy(BasicDataset.preprocess(full_img))
        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            output = net(img)
            if net.n_classes > 1:
                probs = F.softmax(output[-1], dim=1)
                probs = torch.argmax(probs, dim=1)
                mask = torch.squeeze(probs).cpu().numpy()
            else:
                probs = torch.sigmoid(output[-1])
                probs = probs.squeeze(0)
                tf = transforms.Compose(
                    [transforms.ToPILImage(),
                     transforms.Resize(full_img.shape[0]),
                     transforms.ToTensor()])
                probs = tf(probs.cpu())
                full_mask = probs.squeeze().cpu().numpy()
                mask = (full_mask > out_threshold)
        return mask

    def predict_image_all(self, images_nii):
        affine = images_nii.affine
        images = images_nii.get_fdata()
        save_img = np.zeros_like(images)
        len = images.shape[2]
        step = 2
        for j in range(0, len):
            if j < step:
                img = images[:, :, 0:2 * step + 1]
            if (j + step + 1) <= len and (j - step) >= 0:
                img = images[:, :, j - step:j + step + 1]
            if (j + step + 1) > len:
                img = images[:, :, len - 2 * step - 1:len]
            mask = self.predict_image_slice(full_img=img)
            save_img[:, :, j] = mask
        save_img = npy_regionprops_denoise(save_img, 1)
        mask_nii = nib.Nifti1Image(save_img, affine)
        return mask_nii

    def run(self, image_nii):
        preprocessed_nii = self.processing_image(image_nii)
        mask_nii = self.predict_image_all(preprocessed_nii)
        return mask_nii


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m',
                        '--model',
                        dest='model_path',
                        default=
                        '/home3/HWGroup/wushu/8007/8007_code/checkpoints/train_postcava_v1.3/CP_epoch110.pth'
                        )
    parser.add_argument('-t',
                        '--mask_threshold',
                        dest='threshold',
                        type=float,
                        default=0.9,
                        help="Minimum probability value to consider a mask pixel white")
    parser.add_argument('-mt',
                        '--model_type',
                        dest='model_type',
                        type=str,
                        default=UNet_2,
                        help='the training model type')
    parser.add_argument('-ca', '--channels', dest='channels', type=int, default=5, help='the input slices(channels)')
    parser.add_argument('-cs', '--classes', dest='classes', type=int, default=1, help='the classes the model outputs')
    parser.add_argument('-dt', '--data_type', dest='data_type', type=str, default='dce', help='the input data type')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    ornt = ["L", "A", "S"]
    # spacing = [1, 1, 1]
    spacing = [0.875,0.875,0.875]
    shape = [400, 400]
    data_type = 'dce'

    img_nii_dir = '/home3/HWGroup/8007/8007_data/v1.3/sort/vessel/testing/water/'
    pred_dir = '/home3/HWGroup/8007/8007_data/v1.3/pred/pred_testing_postcava/'

    for file in os.listdir(img_nii_dir):
        print(file)
        start_time = time.time()
        img_nii = nib.load(os.path.join(img_nii_dir, file))
        liver_model = ModelPredict(args.model_type, args.classes, args.model_path, ornt, spacing, shape, data_type,
                                   args.threshold, args.channels)
        liver_predict = liver_model.run(img_nii)
        liver_predict = nii_resample(liver_predict, img_nii, interpolation='nearest')
        if not os.path.isdir(pred_dir):
            os.mkdir(pred_dir)
        nib.save(liver_predict, os.path.join(pred_dir, file))
        end_time = time.time()
        print('total time: ', (end_time - start_time))
