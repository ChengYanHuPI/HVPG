import vtk
from vmtk import vmtkscripts
import itk
from skimage import morphology
import numpy as np
import nibabel as nib
import time
import os
from multiprocessing import Pool


def get_affine_matrix(file_path: str) -> dict:
    """
    Get image information--origin,dimension,spacing,affine_matrix
    Args:
        file_path:

    Returns:
        image_info(dict):
    """
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(file_path)
    reader.Update()
    transform = vtk.vtkTransform()
    if reader.GetQFormMatrix():
        transform.SetMatrix(reader.GetQFormMatrix())
    elif reader.GetSFormMatrix():
        transform.SetMatrix(reader.GetSFormMatrix())
    # RAS->LPS
    affine = [0] * 16
    matrix = transform.GetMatrix()
    matrix.DeepCopy(affine, matrix)
    affine = np.array(affine).reshape([4, 4])
    affine = affine * np.array([[-1, -1, -1, -1], [-1, -1, -1, -1], [1, 1, 1, 1],
                                [0, 0, 0, 1]])
    new_matrix = vtk.vtkMatrix4x4()
    for i in range(affine.shape[0]):
        for j in range(affine.shape[1]):
            new_matrix.SetElement(i, j, affine[i, j])
    dims = reader.GetOutput().GetDimensions()
    origin = reader.GetOutput().GetOrigin()
    spacing = reader.GetOutput().GetSpacing()
    image_info = {"image_data"   : reader.GetOutput(),
                  "dimension"    : dims,
                  "origin"       : origin,
                  "spacing"      : spacing,
                  "affine_matrix": new_matrix}
    return image_info


def get_endpoints(nifti_file_path: str, image_info: dict):
    """
    Calculation of all vessels
    Args:
        nifti_file_path:
        image_info:

    Returns:

    """
    info = nib.load(nifti_file_path)
    image_data = info.get_data()
    image_data = morphology.dilation(image_data, morphology.ball(2))
    # image_data = image_info['image_data']
    # dims = image_info['dimension']
    # npy_data = vtk_to_numpy(image_data.GetPointData().GetScalars()).reshape((dims[2], dims[1], dims[0]))
    skeleton = morphology.skeletonize_3d(image_data)
    # Get Endpoints
    coords = np.where(skeleton > 0)
    points = np.array((coords[0], coords[1], coords[2])).transpose()
    offset, end_points = 1, []
    for p in points:
        if np.sum(skeleton[p[0] - offset:p[0] + offset + 1, p[1] - offset:p[1] + offset + 1,
                  p[2] - offset:p[2] + offset + 1] > 0) == 2:
            end_points.append(p)
    # Convert to 3D coordinate space
    matrix = [0] * 16
    affine_matrix = image_info['affine_matrix']
    spacing = image_info['spacing']
    affine_matrix.DeepCopy(matrix, affine_matrix)
    matrix = np.array(matrix).reshape((4, 4))
    new_endpoints = []
    for p in end_points:
        point = [p[0] * spacing[0], p[1] * spacing[1], p[2] * spacing[2], 1]
        res = np.dot(matrix, np.array(point))
        new_endpoints.append(res[:3].tolist())

    return new_endpoints


def nifti2stl(nifti_file_path: str, thresh: float = 1, mesh_save_path: str = None):
    """
    Slice Data Generator
    Args:
        nifti_file_path:
       stl_save_path:

    Returns:
        surface:type vtk.vtkPolyData.
    """
    # Statistics time
    t_start = time.perf_counter()
    if not os.path.exists(nifti_file_path):
        raise FileNotFoundError(nifti_file_path + ' does not exist.')
    if not mesh_save_path:
        if '.nii.gz' in nifti_file_path:
            mesh_save_path = nifti_file_path.replace('.nii.gz', '.vtk')
        elif '.nii' in nifti_file_path:
            mesh_save_path = nifti_file_path.replace('.nii', '.vtk')
        else:
            raise TypeError(nifti_file_path + 'Wrong file type!')
    # Read files
    pixel_type = itk.UC
    dimensions = 3
    image_type = itk.Image[pixel_type, dimensions]
    reader = itk.ImageFileReader[image_type].New()
    reader.SetFileName(nifti_file_path)
    try:
        reader.Update()
    except:
        print('itk error reading nifti file, reset qform with sform and reload!')
        image_info = nib.load(nifti_file_path)
        qform = image_info.get_qform()
        image_info.set_qform(qform)
        sform = image_info.get_sform()
        image_info.set_sform(sform)
        nib.save(image_info, nifti_file_path)
        reader.Update()

    # Thresholding (itk thresholding is extremely time consuming)
    npy_data = itk.GetArrayFromImage(reader.GetOutput())
    npy_data[npy_data < thresh] = 0
    npy_data[npy_data > 0] = 255
    dims = npy_data.shape
    coords = np.where(npy_data > 0)
    points = np.array((coords[2], coords[1], coords[0])).transpose()
    offset, end_points = 1, []
    for p in points:
        if np.sum(npy_data[p[2] - offset:p[2] + offset + 1, p[1], p[0]] > 0) == 1:
            z = p[2] + 1 if p[2] + 1 < dims[0] else p[2] - 1
            npy_data[z, p[1], p[0]] = npy_data[p[2], p[1], p[0]]
    npy_data = morphology.dilation(npy_data, morphology.ball(2))
    image = itk.GetImageFromArray(npy_data)
    image.SetSpacing(reader.GetOutput().GetSpacing())
    image.SetOrigin(reader.GetOutput().GetOrigin())
    image.SetDirection(reader.GetOutput().GetDirection())
    image.CopyInformation(reader.GetOutput())
    # Extraction of slice data
    mesh_type = itk.Mesh[itk.D, dimensions]
    mesh_filter = itk.BinaryMask3DMeshSource[image_type, mesh_type].New()
    mesh_filter.SetInput(image)
    mesh_filter.SetObjectValue(255)
    writer = itk.MeshFileWriter[mesh_type].New()
    writer.SetFileName(mesh_save_path)
    writer.SetInput(mesh_filter.GetOutput())
    writer.SetFileTypeAsBINARY()
    writer.Update()
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(mesh_save_path)
    reader.Update()
    # Smooth grid
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(reader.GetOutput())
    clean.Update()
    # 2, WindowedSinc smoother
    smooth = vtk.vtkSmoothPolyDataFilter()
    smooth.SetInputData(clean.GetOutput())
    smooth.SetNumberOfIterations(20)
    smooth.SetRelaxationFactor(0.1)
    smooth.SetFeatureAngle(175)
    smooth.SetFeatureEdgeSmoothing(1)
    smooth.SetBoundarySmoothing(1)
    smooth.Update()
    normal = vtk.vtkPolyDataNormals()
    normal.SetInputData(smooth.GetOutput())
    normal.SetAutoOrientNormals(1)
    normal.SplittingOff()
    normal.ConsistencyOn()
    normal.ComputePointNormalsOn()
    normal.ComputeCellNormalsOn()
    normal.Update()

    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(mesh_save_path)
    writer.SetInputData(normal.GetOutput())
    writer.Write()

    print('successed,', time.perf_counter() - t_start)

    return normal.GetOutput()


def compute_centerlines(vessel_nifti_file: str, save_path: str = None, file_prefix=None):
    """
    Calculation of vascular centerline by voronoi diagram
    Args:
        vessel_nifti_file:
        save_path:
        file_prefix:
    Returns:

    """
    # Close warning
    warning = vtk.vtkObject()
    warning.GlobalWarningDisplayOff()
    inlets: list = None
    if not os.path.exists(vessel_nifti_file):
        raise FileExistsError('{}file does not exist!'.format(vessel_nifti_file))
    if not (vessel_nifti_file.endswith('.nii.gz') or vessel_nifti_file.endswith('.nii')):
        raise TypeError('{}wrong file type, only nifti format data is supported!')
    # Get affine matrix to convert all endpoints on voxels to 3D model
    image_info = get_affine_matrix(vessel_nifti_file)
    # Get all voxel endpoints
    endpoints = get_endpoints(vessel_nifti_file, image_info)
    # Convert nifti to mesh surface
    mesh_save_path = os.path.join(save_path, file_prefix + '.vtk')
    mesh_surface = nifti2stl(vessel_nifti_file, mesh_save_path=mesh_save_path)

    dist_kd_tree = vtk.vtkKdTree()
    dist_kd_tree.BuildLocatorFromPoints(mesh_surface.GetPoints())
    distances = []
    for p in endpoints:
        res = vtk.vtkIdList()
        dist_kd_tree.FindClosestNPoints(1, p, res)
        if res.GetNumberOfIds() < 1:
            continue
        p1 = mesh_surface.GetPoint(res.GetId(0))
        distances.append(np.linalg.norm(np.array(p) - np.array(p1)))
    if inlets:
        inlet = inlets
    else:
        # Portal vein requires separate entrance calculation
        if file_prefix == 'portal_vein':
            z_pos=list(np.array(endpoints)[:,2])
            max_ind = z_pos.index(min(z_pos))
        else:
            max_ind = distances.index(max(distances))
        # Centerline calculation using vmtk
        inlet = endpoints[max_ind]
    format_outlet = []
    for i in endpoints:
        format_outlet += i
    centerline = vmtkscripts.vmtkCenterlines()
    centerline.Surface = mesh_surface
    centerline.SeedSelectorName = "pointlist"
    centerline.SourcePoints = inlet
    centerline.TargetPoints = format_outlet
    centerline.RadiusArrayName = "MaximumInscribedSphereRadius"
    centerline.Execute()
    # Adjusting the centerline
    centerlines_modified = centerline.Centerlines
    centerlines_modified.BuildCells()
    nbofcells = centerlines_modified.GetNumberOfCells()
    for n in range(nbofcells):
        # Reads the number of dots in each cell
        pids = vtk.vtkIdList()
        centerlines_modified.GetCellPoints(n, pids)
        nbofpoints = pids.GetNumberOfIds()
        # Determine the points in the unit where errors may exist
        if nbofpoints <= 3:
            centerlines_modified.DeleteCell(n)
    centerlines_save_path = os.path.join(save_path, file_prefix + '_centerlines.vtk')
    centerlines_modified.RemoveDeletedCells()
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(centerlines_save_path)
    writer.SetInputData(centerlines_modified)
    writer.Write()

    return mesh_save_path, centerlines_save_path


def main(file_dir: str, save_dir: str = None):
    """
    Calculate centerline program entry
    Args:
        file_dir:
        save_dir:
    Returns:

    """
    if not save_dir:
        save_dir = os.path.join(file_dir, '../ProcessedData')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    p = Pool(processes=3)
    file_names = [['hepatic_vein', 'inferior_vena_venous'], ['portal_vein'], ['abdominal_aorta']]
    suffix = '.nii.gz'
    nifti_file_paths,res=[],[]
    for ind, files in enumerate(file_names):
        file_paths = []
        prefix = files[0]
        for file in files:
            file_path = os.path.join(file_dir, file + suffix)
            if os.path.exists(file_path):
                file_paths.append(file_path)
        if len(file_paths) > 1:
            tmp_merge_file = os.path.join(file_dir, 'hepatic_inferior_vein.nii.gz')
            nifti_file_paths.append(tmp_merge_file)
            info_0 = nib.load(file_paths[0])
            data_0 = info_0.get_data()
            for tmp_file in file_paths[1:]:
                data_1 = nib.load(tmp_file).get_data()
                data_0[data_1 > 0] = 1
            nib.save(nib.Nifti1Image(data_0, info_0.affine, info_0.header), tmp_merge_file)
            res.append(p.apply_async(compute_centerlines, args=(tmp_merge_file, save_dir, prefix,)))
        elif len(file_paths) == 1:
            res.append(p.apply_async(compute_centerlines, args=(file_paths[0], save_dir, prefix,)))
            nifti_file_paths.append(file_paths[0])
        else:
            if ind == 0:
                print('No file corresponding to hepatic vein found！')
            elif ind == 1:
                print('No file corresponding to the portal vein found！')
            else:
                print('No file corresponding to hepatic artery found！')
    p.close()
    p.join()
    file_path_list=[]
    for i,j in enumerate(res):
        file_path_list.append([nifti_file_paths[i], j.get()[0], j.get()[1]])

    return file_path_list


if __name__ == '__main__':
    files = 'D:\\source_code\\VGM_HVPG\\DemoData'
    main(files)
    # compute_centerlines(file_0)
