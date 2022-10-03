from os.path import splitext
from os import listdir
import pickle
import numpy as np
from glob import glob
from numpy.lib.type_check import imag
import torch
from torch.utils.data import Dataset
import logging
# from PIL import Image
import nibabel as nib
import torchvision
from torchvision import transforms as tfs
from torchvision.transforms import transforms
import torchvision.transforms.functional as F
import random
# from mitk.image_processing.geometric_transformation.nii_resize import _nii_rotation
import PIL
# import cv2


class BasicDataset(Dataset):
    def __init__(self,
                 imgs_dir,
                 masks_dir,
                 n_classes,
                 transform=None,
                 mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.n_classes = n_classes
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.ids = [
            splitext(file)[0] for file in listdir(imgs_dir)
            # if not file.startswith('.') and 'HK' in file
        ]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img):
        img_nd = np.array(pil_img)
        img_trans = img_nd.transpose((2, 0, 1))
        return img_trans

    def make_mask_couinaud(self, img):
        maskk = np.zeros((self.n_classes, img.shape[0], img.shape[1]))
        for i in range(self.n_classes):
            maskk[i, :, :] = (np.logical_or(maskk[i, :, :],
                                            (img == i))).astype(int)
        return maskk

    # def rotate_bound(image, angle):
    #     # grab the dimensions of the image and then determine the
    #     # center
    #     (h, w) = image.shape[:2]
    #     (cX, cY) = (w // 2, h // 2)

    #     # grab the rotation matrix (applying the negative of the
    #     # angle to rotate clockwise), then grab the sine and cosine
    #     # (i.e., the rotation components of the matrix)
    #     M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    #     cos = np.abs(M[0, 0])
    #     sin = np.abs(M[0, 1])

    #     # compute the new bounding dimensions of the image
    #     nW = (h * sin) + (w * cos)
    #     nH = (h * cos) + (w * sin)

    #     # adjust the rotation matrix to take into account translation
    #     M[0, 2] += (nW / 2) - cX
    #     M[1, 2] += (nH / 2) - cY

    #     # perform the actual rotation and return the image
    #     return cv2.warpAffine(image, M, (nW, nH))

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'

        mask = np.load(mask_file[0], allow_pickle=True)
        img = np.load(img_file[0], allow_pickle=True)

        img[img < 0] = 0
        
        if self.n_classes == 1:
            mask = np.expand_dims(mask, axis=-1)
            mask = mask.transpose((2, 0, 1))

        # if len(img.shape) == 2:
        #     img = np.expand_dims(img, axis=-1)
        
        img = img.transpose((2, 0, 1))
        # Convert to tensor
        img = torch.from_numpy(img).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)

        if self.transform:
            # Perform random data augmentation
            # if torch.rand(1) < 0.5:
            #     img = F.vflip(img)
            # if torch.rand(1) < 0.5:
            #     mask = F.hflip(mask)
            if self.n_classes == 1:
                if random.random() > 0.5:
                    angle = random.randint(-10, 10)
                    img = F.rotate(img, angle)
                    mask = F.rotate(mask, angle)
            else:
                mask = mask.unsqueeze(0)
                angle = random.randint(-10,10)
                img = F.rotate(img, angle)
                mask = F.rotate(mask, angle)
                mask = torch.squeeze(mask)
                
        return {'image': img, 'mask': mask}


# class BasicDataset_w(Dataset):
#     """
#     WNet_2
#     """
#     def __init__(self,
#                  imgs_dir,
#                  masks_dir,
#                  n_classes,
#                  scale=1,
#                  mask_suffix=''):
#         self.imgs_dir = imgs_dir
#         self.masks_dir = masks_dir
#         self.n_classes = n_classes
#         self.scale = scale
#         self.mask_suffix = mask_suffix
#         assert 0 < scale <= 1, 'Scale must be between 0 and 1'

#         self.ids = [
#             splitext(file)[0] for file in listdir(imgs_dir)
#             if not file.startswith('.')
#         ]
#         logging.info(f'Creating dataset with {len(self.ids)} examples')

#     def __len__(self):
#         return len(self.ids)

#     @classmethod
#     def preprocess(cls, pil_img, scale):
#         img_nd = np.array(pil_img)
#         img_trans = img_nd.transpose((2, 0, 1))
#         return img_trans

#     def make_mask_couinaud(self, img):
#         maskk = np.zeros((img.shape[0], img.shape[1], self.n_classes))
#         for i in range(self.n_classes):
#             maskk[:, :, i] = (np.logical_or(maskk[:, :, i],
#                                             (img == i))).astype(int)
#         return maskk

#     def __getitem__(self, i):
#         idx = self.ids[i]

#         mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
#         img_file = glob(self.imgs_dir + idx + '.*')

#         mask = np.load(mask_file[0])
#         # mask = np.expand_dims(mask, axis=0)

#         img = np.load(img_file[0])

#         # img = np.expand_dims(img, axis=0)

#         # mask = mask.transpose((2, 0, 1))
#         img = img.transpose((2, 0, 1))

#         mask_full = np.zeros_like(mask)
#         mask_portal = np.zeros_like(mask)

#         mask_full[mask >= 1] = 1

#         mask_portal[mask == 2] = 1
#         # mask_portal[mask == 1] = 0

#         return {
#             'image': torch.from_numpy(img).type(torch.FloatTensor),
#             'mask_full': torch.from_numpy(mask_full).type(torch.FloatTensor),
#             'mask_portal':
#             torch.from_numpy(mask_portal).type(torch.FloatTensor)
#         }


# class CarvanaDataset(BasicDataset):
#     def __init__(self, imgs_dir, masks_dir, scale=1):
#         super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
