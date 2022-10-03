#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import vtk
import SimpleITK as sitk


def nifti2stl(nifti_file_path: str):
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
    save_path = nifti_file_path.replace(file_type, '_2.stl')
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(save_path)
    writer.SetFileTypeToBinary()
    writer.SetInputData(surface)
    writer.Write()


if __name__ == '__main__':
    import os, time

    nifti_file_dir = '/home3/HWGroup/wushu/8007/8007_data/pufy/Processed/01190718V003/'
    for file in os.listdir(nifti_file_dir):
        if '.nii.gz' not in file:
            continue
        nifti_file_path = os.path.join(nifti_file_dir, file)
        save_path = nifti_file_path.replace('.nii.gz', '.stl')
        t0 = time.clock()
        nifti2stl(nifti_file_path)
        print(time.clock() - t0)
