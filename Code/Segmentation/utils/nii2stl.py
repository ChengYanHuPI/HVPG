import vtk
import numpy as np
import nibabel as nib
import sys
from multiprocessing import Pool
import os
import cv2
import SimpleITK as sitk


def liver_to_couinaud8(nii_couinaud):
    """
    Conversion of liver segment 9 to liver segment 8
    Args:
        nii_couinaud: Input liver segmentation 9 segment nib data.
        path_couinaud8：Liver segmentation 8 segment preservation pathway
    Returns:
        affine: affine matrix.
        data_couinaud8：Liver segmentation 8 segment npy data
        new_couinaud8： Liver segmentation 8 segment nib file
    """
    data_couinaud = nii_couinaud.get_data()
    if len(data_couinaud.shape) == 4:
        for i in range(data_couinaud.shape[3]):
            if data_couinaud[:, :, :, i].any():
                data_couinaud = data_couinaud[:, :, :, i]

    data_couinaud8 = np.zeros_like(data_couinaud)
    for i in range(1, 9):
        if i < 4 or i == 5 or i == 6:
            data_couinaud8[data_couinaud == i] = i
        if i == 4:
            data_couinaud8[np.logical_or(data_couinaud == i,
                                         data_couinaud == 7)] = i
        if i > 6:
            data_couinaud8[data_couinaud == i + 1] = i
    return data_couinaud8


def nii2stl(nifti_file_path: str):
    """
    nifti2stl
    Args:
        nifti_file_path:

    Returns:
        None:
    """
    # Determine the file type
    file_type = None
    if '.nii.gz' in nifti_file_path:
        file_type = '.nii.gz'
    elif '.nii' in nifti_file_path:
        file_type = '.nii'
    else:
        raise TypeError('Wrong file format!')
    image_info = sitk.ReadImage(nifti_file_path)
    dims = image_info.GetSize()
    images = sitk.GetArrayFromImage(image_info)
    old_origin = image_info.GetOrigin()
    direction = image_info.GetDirection()
    spacing = image_info.GetSpacing()
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(nifti_file_path)
    reader.Update()
    # Modify information
    change = vtk.vtkImageChangeInformation()
    change.SetOutputOrigin(old_origin)
    change.SetInputData(reader.GetOutput())
    change.Update()
    image_data = change.GetOutput()
    origin = image_data.GetOrigin()
    matrix = vtk.vtkMatrix4x4()
    if origin != old_origin:
        matrix.SetElement(0, 0, direction[0])
        matrix.SetElement(0, 1, direction[1])
        matrix.SetElement(0, 2, direction[2])
        matrix.SetElement(0, 3, old_origin[0])
        matrix.SetElement(1, 0, direction[3])
        matrix.SetElement(1, 1, direction[4])
        matrix.SetElement(1, 2, direction[5])
        matrix.SetElement(1, 3, old_origin[1])
        matrix.SetElement(2, 0, direction[6])
        matrix.SetElement(2, 1, direction[7])
        matrix.SetElement(2, 2, direction[8])
        matrix.SetElement(2, 3, old_origin[2])
    else:
        matrix.SetElement(0, 0, direction[0])
        matrix.SetElement(0, 1, direction[1])
        matrix.SetElement(0, 2, direction[2])
        matrix.SetElement(0, 3, 0)
        matrix.SetElement(1, 0, direction[3])
        matrix.SetElement(1, 1, direction[4])
        matrix.SetElement(1, 2, direction[5])
        matrix.SetElement(1, 3, 0)
        matrix.SetElement(2, 0, direction[6])
        matrix.SetElement(2, 1, direction[7])
        matrix.SetElement(2, 2, direction[8])
        matrix.SetElement(2, 3, 0)
    marching_cubes = vtk.vtkImageMarchingCubes()
    marching_cubes.SetInputData(image_data)
    marching_cubes.SetValue(0, 0.6)
    marching_cubes.Update()
    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)
    transform.PostMultiply()

    trans_filter = vtk.vtkTransformPolyDataFilter()
    trans_filter.SetInputData(marching_cubes.GetOutput())
    trans_filter.SetTransform(transform)
    trans_filter.Update()

    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(trans_filter.GetOutput())
    clean.Update()

    surface = clean.GetOutput()
    save_path = nifti_file_path.replace(file_type, '.stl')
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(save_path)
    writer.SetFileTypeToBinary()
    writer.SetInputData(surface)
    writer.Write()


def nii2stl_process(nifti_file_list: list,
                    stl_save_list: list = [],
                    process_num: int = 1):
    """
    Multi-process approach to format conversion
    Args:
        nifti_file_list: List of nii files
        stl_save_list: stl file storage list
        process_num: Number of processes, default is 1
    Returns:
        None
    """
    # The number of processes should not be too high
    if process_num > 8:
        process_num = 8
    elif process_num <= 0:
        process_num = 1
    # Open process
    pool_1 = Pool(processes=process_num)
    for ind, file_path in enumerate(nifti_file_list):
        if len(nifti_file_list) == len(stl_save_list):
            pool_1.apply_async(nii2stl,
                               args=(
                                   nifti_file_list[ind],
                                   stl_save_list[ind],
                               ))
        else:
            pool_1.apply_async(nii2stl, args=(nifti_file_list[ind]))
    pool_1.close()
    # Wait for process execution to complete
    pool_1.join()


def erode_nii(path, path_save):
    nii = nib.load(path)
    data = nii.get_data()
    affine = nii.affine
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    data_ = cv2.erode(data, kernel)
    new_nii = nib.Nifti1Image(data_, affine)
    nib.save(new_nii, path_save)


if __name__ == '__main__':

    ### Batch nii2stl ###
    for name in os.listdir('/home3/HWGroup/8007/8007_data/pufy/Processed/'):
        path = os.path.join('/home3/HWGroup/8007/8007_data/pufy/Processed/', name)
        files = os.listdir(path)
        for file in files:
            if file.endswith('nii.gz'):
                print(name, file)
                sub_file = os.path.join(path, file)
                nii2stl(sub_file)