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
