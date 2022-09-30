# -*- coding: UTF-8 -*-
"""
@File ：compute_params.py
@Author ：xiongminghua
@Email ：xiongminghua@zyheal.com
@Create ：2022-09-26 17:33 
@Modified : 2022-09-26 17:33 
@Note : 
"""
import os
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
from collections import Counter
from vmtk import vmtkscripts
import xlwt, xlrd
from xlutils.copy import copy


def compute_Radius2(mesh_surface: vtk.vtkPolyData, centerlines: vtk.vtkPolyData, cell_points: dict):
    """
    计算血管等效直径、最大直径以及似圆度
    Args:
        mesh_surface:
        cell_points:

    Returns:

    """
    radius_array = centerlines.GetPointData().GetArray('MaximumInscribedSphereRadius')
    radius_eq = {}
    roundness = {}
    plane = vtk.vtkPlane()
    cutter = vtk.vtkCutter()
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    cell_points_params = {}
    for c_id, c_points in cell_points.items():
        radius_params = {}
        if len(c_points) < 10:
            continue
        for ind in range(0,len(c_points),10):
            radius_max, radius_tmp, roundness_tmp = [], [], []
            if ind +5>= len(c_points):
                continue
            p0 = c_points[ind]
            p1 = c_points[ind +5]
            vector = np.array(p0) - np.array(p1)
            normal = vector / np.linalg.norm(vector)
            plane.SetOrigin(p0)
            plane.SetNormal(normal)
            cutter.SetCutFunction(plane)
            cutter.SetInputData(mesh_surface)
            cutter.GenerateTrianglesOn()
            cutter.Update()
            connectivity.SetInputData(cutter.GetOutput())
            connectivity.SetClosestPoint(p0)
            connectivity.SetExtractionModeToClosestPointRegion()
            connectivity.Update()
            delaunay = vtk.vtkDelaunay2D()
            delaunay.SetInputData(connectivity.GetOutput())
            delaunay.SetTolerance(0.00001)
            delaunay.Update()
            # 计算面积
            massprop = vtk.vtkMassProperties()
            massprop.SetInputData(delaunay.GetOutput())
            area = massprop.GetSurfaceArea()
            # 最小半径
            r = radius_array.GetTuple1(centerlines.FindPoint(p0))
            if area >= 5 * vtk.vtkMath.Pi() * r * r:
                continue
            # 计算血管截面周长
            perimeter = 0
            nbofps = connectivity.GetOutput().GetNumberOfPoints()
            perimeter_points = []
            for p in range(nbofps):
                tp = connectivity.GetOutput().GetPoint(p)
                perimeter_points.append(tp)
            tp0 = perimeter_points[0]
            max_radius_tmp = []
            while (perimeter_points):
                tp0_ext = np.tile(np.array(tp0), (len(perimeter_points), 1))
                diff = tp0_ext - np.array(perimeter_points)
                dst_sum = np.sum(diff * diff, axis=1).tolist()
                min_ind = dst_sum.index(min(dst_sum))
                max_radius_tmp.append(np.sqrt(max(dst_sum)))
                perimeter += np.linalg.norm(np.array(tp0) - np.array(perimeter_points[min_ind]))
                tp0 = perimeter_points[min_ind]
                del perimeter_points[min_ind]
            radius_max.append(max(max_radius_tmp))
            radius_tmp.append(np.sqrt(area / vtk.vtkMath.Pi()))
            rd = 4 * vtk.vtkMath.Pi() * area / (perimeter * perimeter)
            if rd > 1:
                continue
            roundness_tmp.append(rd)
            radius_params[c_points[ind]] = [np.nanmean(radius_max), 2*np.nanmean(radius_tmp), 2*r, np.nanmean(roundness_tmp)]
        cell_points_params[c_id] = radius_params

    return cell_points_params


def create_xls(file_path: str):
    style = xlwt.XFStyle()
    align = xlwt.Alignment()
    align.horz = 0x02  # 设置水平居中
    align.vert = 0x01  # 设置垂直居中
    style.alignment = align
    # 生成params的汇总文件头
    workbook = xlwt.Workbook(encoding='ascii')
    worksheet = workbook.add_sheet('sheet')
    worksheet.write(0, 0, '', style)
    worksheet.write(1, 0, 'ID', style)
    # 合并文件的顺序
    keys = ['血管参数']
    for ind, key in enumerate(keys):
        worksheet.write_merge(0, 0, ind * 19 + 1, (ind + 1) * 19, key, style)
        # 添加第二行标题
        worksheet.write(1, ind * 19 + 1, '容积(ml)', style)
        worksheet.write(1, ind * 19 + 2, '总长度(mm)', style)
        worksheet.write(1, ind * 19 + 3, '主干长度(mm)', style)
        worksheet.write(1, ind * 19 + 4, '分支长度均值(mm)', style)
        worksheet.write(1, ind * 19 + 5, '血管主干扭曲度均值', style)
        worksheet.write(1, ind * 19 + 6, '血管分支扭曲度均值', style)
        worksheet.write(1, ind * 19 + 7, '血管主干曲率均值', style)
        worksheet.write(1, ind * 19 + 8, '血管分支曲率均值', style)
        worksheet.write(1, ind * 19 + 9, '血管主干直径最大值均值(mm)', style)
        worksheet.write(1, ind * 19 + 10, '血管分支直径最大值均值(mm)', style)
        worksheet.write(1, ind * 19 + 11, '血管主干等效直径均值(mm)', style)
        worksheet.write(1, ind * 19 + 12, '血管分支等效直径均值(mm)', style)
        worksheet.write(1, ind * 19 + 13, '血管主干最小直径均值(mm)', style)
        worksheet.write(1, ind * 19 + 14, '血管分支最小直径均值(mm)', style)
        worksheet.write(1, ind * 19 + 15, '血管主干截面似圆度', style)
        worksheet.write(1, ind * 19 + 16, '血管分支截面似圆度', style)
        worksheet.write(1, ind * 19 + 17, '端节点数', style)
        worksheet.write(1, ind * 19 + 18, '分支节点数', style)
        worksheet.write(1, ind * 19 + 19, '分支数', style)
    workbook.save(file_path)
    # print('')
    return workbook, worksheet


def insert_data(file_path: str, patient_id: str, params):
    pass
    style = xlwt.XFStyle()
    align = xlwt.Alignment()
    align.horz = 0x02  # 设置水平居中
    align.vert = 0x01  # 设置垂直居中
    style.alignment = align
    rxls = xlrd.open_workbook(file_path)
    rows = rxls.sheets()[0].nrows  # 读取excel文件
    workbook = copy(rxls)  # 将xlrd的对象转化为xlwt的对象
    worksheet = workbook.get_sheet(0)  # 获取要操作的sheet
    worksheet.write(rows, 0, patient_id, style)
    for param_ind, val in enumerate(params):
        worksheet.write(rows, param_ind + 1, val, style)
    workbook.save(file_path)


def compute_params(nifti_file: str, mesh_file: str, centerlines_file: str):
    """
    参数计算函数接口
    Args:
        nifti_file:
        mesh_file:
        centerlines_file:

    Returns:

    """
    warning = vtk.vtkObject()
    warning.GlobalWarningDisplayOff()
    # 血管容积
    image_reader = vtk.vtkNIFTIImageReader()
    image_reader.SetFileName(nifti_file)
    image_reader.Update()
    image_data = image_reader.GetOutput()
    spacing = image_data.GetSpacing()
    unit_vol = spacing[0] * spacing[1] * spacing[2] * 0.001
    npy_data = vtk_to_numpy(image_data.GetPointData().GetScalars())
    vol = np.sum(npy_data > 0) * unit_vol
    # 其他的参数
    centerline_reader = vtk.vtkPolyDataReader()
    centerline_reader.SetFileName(centerlines_file)
    centerline_reader.Update()
    centerlines = centerline_reader.GetOutput()
    centerlines_group = {}
    all_points = []
    for cell_id in range(centerlines.GetNumberOfCells()):
        cell_points = []
        cell_idlist = vtk.vtkIdList()
        centerlines.GetCellPoints(cell_id, cell_idlist)
        nbofpids = cell_idlist.GetNumberOfIds()
        for p_id in range(nbofpids):
            points = centerlines.GetPoint(cell_idlist.GetId(p_id))
            if p_id > 0:
                points_0 = centerlines.GetPoint(cell_idlist.GetId(p_id - 1))
                if points_0 != points:
                    cell_points.append(points)
                    all_points.append(points)
        centerlines_group[cell_id] = cell_points
    # 统计所有点出现的次数
    label_points = Counter(all_points)
    centerlines_points_group = {}
    for key, c_points in centerlines_group.items():
        label_cell_point = []
        for point in c_points:
            label_cell_point.append(label_points[point])
        centerlines_points_group[key] = label_cell_point
    # 血管长度
    cell_length, euclid_length, cell_label = {}, {}, {}
    for key, label_cell in centerlines_points_group.items():
        cell_dist, euclid_dist, labels = [], [], []
        start_id, _id, dist = 0, 0, 0
        for ind, val in enumerate(label_cell):
            if val == label_cell[_id]:
                p0 = centerlines_group[key][_id]
                p1 = centerlines_group[key][ind]
                dist += np.linalg.norm(np.array(p0) - np.array(p1))
                _id = ind
                if ind == len(label_cell) - 1:
                    labels.append(label_cell[start_id])
                    cell_dist.append(dist)
                    p0 = centerlines_group[key][start_id]
                    p1 = centerlines_group[key][_id]
                    euclid_dist.append(np.linalg.norm(np.array(p0) - np.array(p1)))
                    dist, start_id, _id = 0, ind, ind
            else:
                labels.append(label_cell[start_id])
                cell_dist.append(dist)
                p0 = centerlines_group[key][start_id]
                p1 = centerlines_group[key][_id]
                euclid_dist.append(np.linalg.norm(np.array(p0) - np.array(p1)))
                dist, start_id, _id = 0, ind, ind
        cell_length[key] = cell_dist
        euclid_length[key] = euclid_dist
        cell_label[key] = labels
    # 总长度
    total_length = 0
    for key, vals in cell_length.items():
        for i, val in enumerate(vals):
            total_length += val / cell_label[key][i]
    # 主干长度
    main_branch_length = cell_length[0][0]
    # 分支长度
    mean_branch_length = (total_length - main_branch_length) / len(cell_length.values())

    # 端节点数
    terminal_nodes = centerlines.GetNumberOfCells() + 1
    # 分支数
    branches = 0
    for key, vals in cell_label.items():
        for i, val in enumerate(vals):
            if val > 1:
                branches += 1 / val
            else:
                branches += 1
    # 分支节点数量
    branch_nodes = int(branches) + 1 - terminal_nodes
    # 主干扭曲度
    main_branch_tuotorsity = (cell_length[0][0] / euclid_length[0][0]) - 1
    # 分支扭曲度平均值
    branches_tuotorsity = []
    for key, vals in cell_length.items():
        for i, val in enumerate(vals[1:]):
            branches_tuotorsity.append((val / euclid_length[key][i + 1]) - 1)
    mean_branches_tuotorsity = np.nanmean(branches_tuotorsity)
    # 计算曲率
    curvature = vmtkscripts.vmtkCenterlineGeometry()
    curvature.Centerlines = centerlines
    curvature.CurvatureArrayName = 'Curvature'
    curvature.Execute()
    curvature_array = curvature.Centerlines.GetPointData().GetScalars('Curvature')
    # 主干曲率\分支曲率
    main_branch_curvature, branch_curvatures = [], []
    for cell_id in range(curvature.Centerlines.GetNumberOfCells()):
        cell_idlist = vtk.vtkIdList()
        curvature.Centerlines.GetCellPoints(cell_id, cell_idlist)
        nbofpids = cell_idlist.GetNumberOfIds()
        for p_id in range(nbofpids):
            pos = curvature.Centerlines.GetPoint(cell_idlist.GetId(p_id))
            curv = curvature_array.GetTuple1(cell_idlist.GetId(p_id))
            if label_points[pos] == cell_label[cell_id][0]:
                main_branch_curvature.append(curv)
            else:
                branch_curvatures.append(curv)
    main_curvature = np.nanmedian(main_branch_curvature)
    branches_curvature = np.nanmedian(branch_curvatures)
    # 计算似圆度
    mesh_reader = vtk.vtkPolyDataReader()
    mesh_reader.SetFileName(mesh_file)
    mesh_reader.Update()
    mesh_surface = mesh_reader.GetOutput()
    radius_params = compute_Radius2(mesh_surface, centerlines, centerlines_group)
    # 主干最大直径、等效直径、最小直径以及似圆度
    # 分支最大直径、等效直径、最小直径以及似圆度
    main_branch_param, branches_params = [], []
    for cell_id,vals in radius_params.items():
        for pos,val in vals.items():
            if cell_label[cell_id][0]== label_points[pos]:
                main_branch_param.append(val)
            else:
                branches_params.append(val)
    median_main_branch=np.median(main_branch_param, axis=0)
    median_branch_param= np.median(branches_params, axis=0)
    # 写入参数
    xls_file_dir = os.path.join(nifti_file, '../../Features/')
    if not os.path.exists(xls_file_dir):
        os.makedirs(xls_file_dir)
    xls_file=os.path.join(xls_file_dir,'vessel_params.xls')
    if not os.path.exists(xls_file):
        create_xls(xls_file)
    params = [vol, total_length,main_branch_length, mean_branch_length, main_branch_tuotorsity,
              mean_branches_tuotorsity, main_curvature, branches_curvature,
              median_main_branch[0], median_branch_param[0], median_main_branch[1], median_branch_param[1],
              median_main_branch[2], median_branch_param[2],
              median_main_branch[3], median_branch_param[3],
              terminal_nodes,branch_nodes,branches]
    # 获取文件
    patient_id=os.path.split(nifti_file)[-1].split('.')[0]
    insert_data(xls_file, patient_id, params)


if __name__ == '__main__':
    nifti_file = 'D:/source_code/VGM_HVPG/DemoData/abdominal_aorta.nii.gz'
    mesh_file = 'D:/source_code/VGM_HVPG/ProcessedData/abdominal_aorta.vtk'
    centerlines_file = 'D:/source_code/VGM_HVPG/ProcessedData/abdominal_aorta_centerlines.vtk'
    compute_params(nifti_file, mesh_file, centerlines_file)
