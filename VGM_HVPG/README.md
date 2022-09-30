# 2.5D UNet

Based on UNet and changed the dimension to 2.5D which means input image shape is h x w x 5.

## Dependencies

Ubuntu 20.04.3, python 3.6, CUDA 11.0, anaconda (4.10.1),nibabel (3.2.1), SimpleITK (2.1.1), numpy (1.19.5), scikit-image (0.17.2), scipy (1.5.2), pytorch (1.7.1), tqdm(4.46.0), opencv-python(4.46.0.66), itk(5.2.0), tensorboard(2.5.0)

This is my configuration, I am not sure about the compatability of other versions

## Setup

```bash
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 -c pytorch
pip install nibabel==3.2.1
pip install tensorboard==2.5.0
# private repository 
pip install git+ssh://git@e.coding.net/xymedimg/passer-workers/medical-imaging-toolkit.git@v0.0.3#egg=mitk
# Others can be installed by pip
```

## Data

##### prepare data

1. Annotate your data and convert to nifity format files (.nii/.nii.gz).

   example(hepatic vein):

   <img src="./DemoData/img1.png" style="zoom:50%;" />

2. Put the original image and the corresponding labels into two folders to prepare for preprocessing and training.

##### Preprocess data

Preprocess data into numpy files required by the network: modify the corresponding path parameters in the `preprocess.py` and run it by your environment.

You can train better network models by modifying these preprocessing parameters:

 `*spacing*, *shape*, *data_type*(can be modified by yourself)`

## Running

- Training

  run `train.py`

  You need check and set the parameters: CUDA_VISIBLE_DEVICES, dir_checkpoint, input_path, label_path, batchsize, lr, model_type, channels, classes...

- Testing

  run `predict.py`

  You need check and set the parameters: CUDA_VISIBLE_DEVICES, model_path, threshold, model_type, channels, classes, data_type, ornt, spacing, shape(according to your preprocess parameters), img_nii_dir, pred_dir...

## Results

The following is one of my predicting results.

<img src="./DemoData/img2.png" style="zoom:50%;" />

<img src="./DemoData/img3.png" style="zoom:75%;" />

# Extraction of Hepatic Vascular Parameters
  ## 1. Installation
    (a) Relying on third-party library
        The development environment of this module: Windows10, anaconda, python3.6.9. The used third-party libraries: vmtk, vtk, itk, scikit-image, nibabel, xlwt, xlrd, xlutils.

        Note: Nibabel is used to rewrite head information of nifti files. (Itk will report an error when reading some files, indicating that the non-orthogonal file cannot be read). However, xlwt, xlrd, and xlutils are used to read, write, and copy excel files. Itk, vtk read nifti files, reading and writing 3-D model files(*vtk) ; vmtk is used to calculate vascular centerline. 

    (b) Installation Sequence:
        First, install anaconda, since vtmk library needs to be installed successfully in anaconda environment.  After installation is complete, create a new environment and switch to it. In the anaconda prompt, command "conda create -n vmtk python 3.6.9" and "conda active vmtk". Then, install the vmtk library and command " conda install -c vmtk vtk itk vmtk" to install vmtk, vtk, and itk. Finally, install other third-party libraries normally. 
        Suggestions: switch to domestic mirror source.

  ## 2. Application Method
    (a) Use the source code
        In the anaconda prompt, command "conda activate vmtk", to switch to the vmtk environment and enter the directory of the source code. Then, command "python compute_vessel_params.py../../DemoData" and wait until completion.  Generated 3-D model files and the corresponding centerline files will be stored in the directory "ProcessedData". Parameters will be stored in the directory "Features".

  ## 3. Results
    (a) Model files
        Generated 3-D files include model files of vessels and the corresponding centerline files. After running, they will be stored in the directory of "ProcessedData".
    (b) Calculation of  parameter index
        Calculated parameter indexes will be stored in excel file in the directory "Features".
        For instance, ![](assets/README-99dd089b.png)


  ## 4. Extraction process of hepatic vascular parameters 
    * Segmentation of hepatic vasculars:  The neural network is used to predict hepatic vasculars. The predicted hepatic vascular data will be stored as nifti format.
    * Calculation of vascular centerlines: The nifti data of  predicted vascular centerlines generates 3-D models (*.vtk file). Then, according to 3-D models, calculate vascular centerlines
    * Calculation of vascular parameter indexes:  Based on vascular centerlines and nfti data, calculate vascular parameters, including vascular volume, vascular length, vascular torsion resistance, vascular curvature, vascular diameter, vascular section roundness, vascular node number, etc.
    * All parameters are stored in an excel file.

  ## 5. Vascular parameter index:
    *Vascular volume(ml): vascular total volume
    *Vascular length(mm): including vascular total length, the length of the main vessel and the length of the branch vessel.
    *Vascular torsion resistance: including mean torsion resistance of main vessel and mean torsion resistance of branch vessel
    *Vascular curvature (1/mm): including mean curvature of main vessel and mean curvature of branch vessel 
    *Vascular diameter (mm): including maximum diameter, equivalent diameter, minimum diameter of vessel
