# 肝脏血管参数提取
  ## 1、安装
    (a)、依赖第三方库
      该模块的开发环境为：Windows10、anaconda、python3.6.9。其中需要用到的第三方库有：vmtk、vtk、itk、scikit-image、nibabel、xlwt、xlrd、xlutils。
      注意：其中nibabel用来重写nifti文件头信息（itk读取部分文件会报错，提示无法读取非正交文件），而xlwt，xlrd，xlutils用来对excel文件进行读写和拷贝。itk，vtk读取nifti文件并读写三维模型文件（*.vtk）；vmtk库用来计算血管中心线

    (b)、安装顺序
      由于vmtk库需要在anaconda环境下才能安装成功，故需要先安装anaconda，安装完成后创建一个新的环境并切换到该环境下，在anaconda prompt中执行“conda create -n vmtk python=3.6.9”，“conda activate vmtk”，最后安装vmtk库，执行“conda install -c vmtk vtk itk vmtk” 安装vmtk以及vtk、itk；然后正常安装其他的第三方库。建议更换为国内镜像源。

  ## 2、使用方法
    (a)、使用源码
      在anaconda prompt中执行“conda activate vmtk”切换到vmtk环境下，并进入源码的目录，执行“python compute_vessel_params.py ../../DemoData”,等待运行结束即可。生成的三维模型文件和对应的中心线文件将会存储到目录“ProcessedData”中，参数结果将会存储到目录“Features”中

  ## 3、结果
    (a)、模型文件
      生成的三维模型文件包括：血管的模型文件以及对应的中心线文件，运行结束后将会存储到目录“ProcessedData”。
    (b)、计算的参数指标
      计算的参数指标将会以excel的方式存储于“Features”中，例如：
  ![](assets/README-99dd089b.png)


  ## 4、肝脏血管参数提取流程
    * 肝脏血管分割：采用神经网络对肝脏血管进行预测，并将预测的血管数据存储为nifti格式的数据
    * 计算血管中心线：将预测的血管nifti数据生成三维模型（*.vtk文件），然后根据三维模型计算出血管的中心线。
    * 计算血管参数指标：根据血管的中心线和nifti数据计算血管的参数，包括血管的容积、血管长度、血管扭曲度、血管曲率、血管管径、血管截面似圆度、血管节点数等。
    * 将所有的参数存储到excel文件中。
  ## 5、血管参数指标
    * 血管容积(ml)：血管的总容积
    * 血管长度(mm)：包括血管总长度、血管主干长度以及血管分支长度 ；
    * 血管扭曲度：包括血管主干扭曲度均值、血管分支扭曲度均值
    * 血管曲率(1/mm)：包括血管主干曲率均值和血管分支曲率均值
    * 血管管径(mm)：包括血管最大管径、等效管径以及最小管径
