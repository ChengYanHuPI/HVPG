import vtk, time
from vtk.util.numpy_support import numpy_to_vtk
import nibabel as nib
import numpy as np
from multiprocessing import Pool
from mitk import DcmDataSet, dcm2nii
import os


def stl2image(stl_file_path: str,
              nifti_file_path: str,
              nifti_save_path: str = None):
    '''
    Convert annotated stl models to voxel
    Args:
        stl_file_path (str):stl model file path
        nifti_file_path (str):The raw nifti data corresponding to the stl model, used to extract the file header information
        nifti_save_path (str):The path to store the data after conversion, the default is the model file name
    
    '''

    # If the input is directly nifti, the above code can be commented out
    read = vtk.vtkNIFTIImageReader()
    read.SetFileName(nifti_file_path)
    read.Update()
    image_info = nib.load(nifti_file_path)
    # Get Origin
    affine = image_info.affine
    origin = list(affine[:, 3][:3])
    for i in range(3):
        if affine[i, i] < 0:
            origin[i] = -1 * origin[i]
    # Get spacing
    spacing = image_info.header.get_zooms()
    im_header = image_info.header
    dims = image_info.get_data().shape
    # Read model files
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_file_path)
    reader.Update()
    surface = reader.GetOutput()
    # Convert
    whiteImage = vtk.vtkImageData()
    whiteImage.SetSpacing(spacing)
    whiteImage.SetDimensions(dims)
    whiteImage.SetExtent(0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)
    whiteImage.SetOrigin(origin)
    whiteImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
    whiteImage.GetPointData().SetScalars(
        numpy_to_vtk(np.array([1] * dims[0] * dims[1] * dims[2])))
    # fill the image with foreground voxel
    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(surface)
    pol2stenc.SetOutputOrigin(origin)
    pol2stenc.SetOutputSpacing(spacing)
    pol2stenc.SetOutputWholeExtent(whiteImage.GetExtent())
    pol2stenc.Update()
    #cut the corresponding white image and set the background:
    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(whiteImage)
    imgstenc.SetStencilData(pol2stenc.GetOutput())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(0)
    imgstenc.Update()
    # dilate one pixel
    dilate = vtk.vtkImageContinuousDilate3D()
    dilate.SetKernelSize(
        2, 2,
        2)  #If KernelSize of an axis is 1, no processing is done on that axis.
    dilate.SetInputData(imgstenc.GetOutput())
    dilate.Update()
    # save image data
    if not nifti_save_path:
        nifti_save_path = stl_file_path[:-3] + 'nii.gz'
    writer = vtk.vtkNIFTIImageWriter()
    writer.SetFileName(nifti_save_path)
    writer.SetInputData(dilate.GetOutput())
    writer.SetNIFTIHeader(read.GetNIFTIHeader())
    writer.Write()

    image_info = nib.load(nifti_save_path)
    nib.save(nib.Nifti1Image(image_info.get_data(), affine, im_header),
             nifti_save_path)


def stl2image_process(nifti_path: str, stl_path: str, process_num: int = 1):
    '''
    Multi-process batch conversion
    model_file_list (list):List of model file paths
    nifti_file_list (list):Used to get the corresponding file header information
    nifti_save_list (list):The file name of the model after conversion, default is the file name of the stl model
    process_num (int): Number of processes, default is 1
    '''
    # The number of processes should not be too high
    if process_num > 8:
        process_num = 8
    elif process_num <= 0:
        process_num = 1
    # Open process
    pool_1 = Pool(processes=process_num)
    model_file_list = os.listdir(stl_path)
    for file in model_file_list:
        stl_file = os.path.join(stl_path, file)
        pool_1.apply_async(stl2image, args=(stl_file, nifti_path))
    pool_1.close()
    # Wait for process execution to complete
    pool_1.join()


if __name__ == '__main__':
    image_file_path_ = '/home3/HWGroup/wushu/8007/8007_data/data_v1.4/CT/01200516V009/CT/chest/CleanNII/Series0203_lung.nii.gz'
    stl_file_path_ = '/home3/HWGroup/wushu/8007/8007_data/data_v1.4/AO&SVC_ROI/dkj/01200516V009/STL/'
    # save_path='/home3/HWGroup/licp/test_pipeline_couinaud/test_mimics/conv_A.nii.gz'
    t0 = time.clock()
    stl2image_process(image_file_path_, stl_file_path_)
    print(time.clock() - t0)
