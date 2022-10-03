import argparse
import logging
import os
import time
from nibabel.nifti1 import Nifti1Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import pydicom
from mitk.image_conversion.dcm2nii import dcm2nii
from mitk import DcmDataSet, nii2dcmseg,dcmseg2nii1
import nibabel as nib

def check_origin(path_new, path_ori):
    ori_nii = nib.load(path_ori)
    new_nii = nib.load(path_new)
    if new_nii.header['qoffset_z'] != ori_nii.header['qoffset_z']:
        new_nii.header['qoffset_z'] = ori_nii.header['qoffset_z']
        new_nii.affine[:] = ori_nii.affine[:]
        nib.save(new_nii, path_new)
    else:
        pass

#dcm2nii
def func_dcm2nii(path_dcm,path_save):
    dcm = DcmDataSet(path_dcm)
    lenth = dcm.get_volume_indexs_lenth()
    print(f'\033[1;30mvolume_indexs_lenth: {lenth}\033[0m')
    if lenth == 4:
        volume_no = 3
    elif lenth == 1:
        volume_no = 0
    else:
        raise ValueError(f'Volume_indexs_lenth {lenth} should be 1 or 4!')
    nii = dcm2nii(path_dcm, volume_no)
    nib.save(nii, path_save)

#dcmseg2nii
def get_a_dicom_slice(dcm_folder: str, task = 'None'):
    """
    Read diocm information from DICOM folder
    Args:
        dcm_folder: DICOM folder.
    Returns:
        ds_slice: Return diocm information.
    """
    files = os.listdir(dcm_folder)
    for i in range(len(files)):
        dcm_path = os.path.join(dcm_folder, files[i])
        ds_slice = pydicom.dcmread(dcm_path)
        if hasattr(ds_slice, 'SpacingBetweenSlices'):
            return ds_slice
        elif 'fatty' in task or 'visualization' in task:
            return ds_slice
        else:
            continue


