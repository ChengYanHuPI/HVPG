import itk
import numpy as np
import vtk
import os
import nibabel as nib
import time
import shutil
from skimage import morphology
from scipy import ndimage
from threading import Thread
import cv2


def stl2vtk(stl_file_path: str) -> str:
    """
    Convert stl files to vtk files
    Args:
        stl_file_path:

    Returns:

    """
    # Convert stl files to vtk files
    stl_reader = vtk.vtkSTLReader()
    stl_reader.SetFileName(stl_file_path)
    stl_reader.Update()
    # Store as vtk file
    vtk_file_path = stl_file_path.replace('.stl', '_tmp.vtk')
    vtk_writer = vtk.vtkPolyDataWriter()
    vtk_writer.SetFileName(vtk_file_path)
    vtk_writer.SetInputData(stl_reader.GetOutput())
    vtk_writer.SetFileVersion(42)
    # vtk_writer.SetFileTypeToBinary()
    vtk_writer.Write()

    return vtk_file_path


def fill_image_bounds(nifti_file_path: str):
    """
    Fill image border pixel values
    Args:
        nifti_file_path:

    Returns:
        None:
    """
    image_info = nib.load(nifti_file_path)
    image_data = image_info.get_data()
    # Filling six sides of an image
    empty_image = np.zeros_like(image_data)
    dims = image_data.shape
    slice_range = 3
    for i in range(slice_range):
        empty_image[:, :, i] = image_data[:, :, i + 1]
        empty_image[:, :, dims[2] - 1 - i] = image_data[:, :, dims[2] - 2 - i]
        empty_image[:, i, :] = image_data[:, i + 1, :]
        empty_image[:, dims[1] - 1 - i, :] = image_data[:, dims[1] - 2 - i, :]
        empty_image[i, :, :] = image_data[i + 1, :, :]
        empty_image[dims[0] - 1 - i, :, :] = image_data[dims[0] - 2 - i, :, :]
    # Morphological closed-open operations
    new_image_data = morphology.opening(morphology.closing(empty_image, morphology.ball(2)), morphology.ball(2))
    for i in range(slice_range - 1):
        image_data[:, :, i] = new_image_data[:, :, i]
        image_data[:, :, dims[2] - 1 - i] = new_image_data[:, :, dims[2] - 1 - i]
        image_data[:, i, :] = new_image_data[:, i, :]
        image_data[:, dims[1] - 1 - i, :] = new_image_data[:, dims[1] - 1 - i, :]
        image_data[i, :, :] = new_image_data[i, :, :]
        image_data[dims[0] - 1 - i, :, :] = new_image_data[dims[0] - 1 - i, :, :]

    nib.save(nib.Nifti1Image(image_data, image_info.affine, image_info.header), nifti_file_path)


def stl2nifti(stl_file_path: str, ref_image_path: str, nifti_file_save_path: str = None):
    """
    stl files are converted to nifti files and stored
    Args:
        stl_file_path: stl file path
        ref_image_path: Reference image storage path
        nifti_file_save_path: nifti image storage path, by default stored to the same directory as the stl file

    Returns:
        None:
    """
    t_start = time.clock()
    if not os.path.exists(stl_file_path):
        raise FileNotFoundError('Model file does not exist! {0}'.format(stl_file_path))
    if not os.path.exists(ref_image_path):
        raise FileNotFoundError('Reference image does not exist! {0}'.format(ref_image_path))
    # File type determination
    if stl_file_path.endswith('.stl'):
        stl_file_path = stl2vtk(stl_file_path)
    elif stl_file_path.endswith('.vtk'):
        vtk_file_path = stl_file_path.replace('.vtk', '_tmp.vtk')
        Thread(target=shutil.copy, args=[stl_file_path, vtk_file_path]).start()
        stl_file_path = vtk_file_path
    else:
        raise TypeError('Model file {} type error, currently only support .vtk, .stl type file!'.format(stl_file_path))

    # Storage Path
    if not nifti_file_save_path:
        nifti_file_save_path = stl_file_path.replace('_tmp.vtk', '.nii.gz')

    InputPixelType = itk.SS
    OutputPixelType = itk.SS
    MeshPixelType = itk.SS
    Dimension = 3
    OutputImageType = itk.Image[OutputPixelType, Dimension]
    MeshType = itk.Mesh[MeshPixelType, Dimension]
    InputImageType = itk.Image[InputPixelType, Dimension]
    # Rewrite the direction matrix (to avoid non-orthogonal errors)
    image_info = nib.load(ref_image_path)
    q_form = image_info.get_qform()
    image_info.set_qform(q_form)
    s_form = image_info.get_sform()
    image_info.set_sform(s_form)
    nib.save(image_info, ref_image_path)
    # itk load file
    try:
        images = itk.imread(ref_image_path, itk.SS)
        # Load model file and reference image file
        mesh_reader = itk.meshread(stl_file_path, itk.SS)
        cast_filter = itk.CastImageFilter[InputImageType, OutputImageType]
        cast = cast_filter.New()
        cast.SetInput(images)
        mesh_label = itk.TriangleMeshToBinaryImageFilter[MeshType, OutputImageType]
        mesh_label_map = mesh_label.New()
        mesh_label_map.SetInput(mesh_reader)
        mesh_label_map.SetInfoImage(cast.GetOutput())
        mesh_label_map.SetInsideValue(1)
        mesh_label_map.Update()
        itk.imwrite(mesh_label_map.GetOutput(), nifti_file_save_path)
        fill_image_bounds(nifti_file_save_path)
    except:
        raise RuntimeError('Model file to label file conversion failed!')
    if os.path.exists(stl_file_path):
        os.remove(stl_file_path)
    print('successed,', time.clock() - t_start)


if __name__ == '__main__':

    stl_file = '/home3/HWGroup/abdominal_fat/vatsatseg-master/data_zsnfm/1.3.46.670589.11.78173.5.0.12904.2021102414260573082/skeleton.stl'
    image_file = '/home3/HWGroup/abdominal_fat/vatsatseg-master/data_zsnfm/1.3.46.670589.11.78173.5.0.12904.2021102414260573082/FatImaging_W.nii.gz'
    stl2nifti(stl_file, image_file)


