## 绝缘子缺陷检测

### 1 项目背景

随着国家西电东送战略的提出，特高压输电和高压输电变得尤为重要，输电线路的巡检是保证输电正常的重要一环。智能化电网和的快速发展，巡检无人机、机器人正在逐步代替传统人工巡检。绝缘子是输电线路的重要组成部分，是唯一的电气绝缘件和重要的结构支撑件，绝缘子性能及其配置的合理性直接影响线路的安全稳定运行。绝缘子破损的快速定位排除对保证高压输电线路的正常运行和避免大范围停电具有十分重要的意义。

### 2 算法

#### 2.1 PP-YOLO

PP-YOLO是PaddleDetection优化和改进的YOLOv3的模型，其精度(COCO数据集mAP)和推理速度均优于YOLOv4模型，在COCO test-dev2017数据集上精度达到45.9%，在单卡V100上FP32推理速度为72.9 FPS, V100上开启TensorRT下FP16推理速度为155.6 FPS。该项目使用百度自研算法PP-YOLO v2。

![](D:\wdz\NIUVS\算法.png)

#### 2.2模型库

[PP-YOLO模型库](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/configs/ppyolo/README_cn.md#模型库) 可以根据自己的具体需求进行模型的下载

### 3 项目内容

该项目内容主要包括环境的配置，数据的准备，配置文件的修改，模型训练，模型评估，模型推理测试等。

#### 3.1 环境配置

安装Paddledetection的依赖,这里使用的版本为**paddlepaddle-gpu2.2.2.post101**

安装PaddleDetection依赖

```python
pip install -r /home/C/wangdz/PaddleDetection2/PaddleDetection_my/requirements.txt
```

安装其他依赖

```python
from PIL import Image   

import matplotlib.pyplot as plt

import numpy as np 

import os  

import random
```

编译安装paddledet

```python
cd work/PaddleDetection 

python setup.py install
```

测试安装是否成功，最后输出OK，说明编译安装成功

```python
python ppdet/modeling/tests/test_architectures.py
```

#### 3.2 准备数据集

所使用的数据集存放在insulator文件夹中。

划分数据集

```python
python make_dataset.py
```

最后得到的数据集目录如下

```python
dataset
├── Annotations
│   └── .xml
├── JEPGImages
│   └── .jpg
├── label_list
├── train.txt
└── val.txt
```

随机查看文字检测数据集图片，如果运行一次没有显示图片，再运行一次即可

```python
train = '/home/C/wangdz/PaddleDetection2/PaddleDetection_my/insulator/JPEGImages'  
#从指定目录中选取一张图片
def get_one_image(train):        
    #image_array = get_one_image(train)
    plt.figure()     
    files = os.listdir(train)    
    n = len(files)    
    ind = np.random.randint(0,n)    
    img_dir = os.path.join(train,files[ind])      
    image = Image.open(img_dir)      
    plt.imshow(image)    
    plt.show()    
    image = image.resize([508, 508])   
get_one_image(train)  
```

#### 3.3 修改配置文件

找到我们训练需要的配置文件，以ppyolov2其中一个的配置文件为例，configs/ppyolo/ppyolov2_r50vd_dcn_voc.yml

然后根据实际的需求进行修改

1. ../datasets/voc.yml 主要说明了训练数据和验证数据的路径。例如：数据集格式，分类数和训练集路径、验证集路径等。
2. ../runtime.yml主要说明了公共的运行参数。例如：是否使用GPU，模型保存路径等。
3. ./*base*/ppyolov2_r50vd_dcn.yml 主要说明模型、和主干网络的情况。例如backbone，neck，head，loss，前后处理等。
4. ../*base*/optimizer_365e.yml 主要说明了学习率和优化器的配置。例如学习率和学习率策略，优化器类型等。
5. ./*base*/ppyolov2_reader.yml 主要说明数据读取器配置。例如batch size，并发加载子进程数，数据预处理等。

新版的PaddleDetection，可以对配置文件集中修改，不需要每个配置文件都挨个进行更改。后续调优过程中可以去每个配置文件再进行具体调整。 所以只需要修改配置文件configs/ppyolo/ppyolov2_r50vd_dcn_voc.yml和和配置文件../datasets/voc.yml

#### 3.4 开始训练

**--eval** 代表边训练边评估，每次评估后还会评出最佳mAP模型保存到best_model文件夹下，如果验证集很大，测试将会比较耗时，建议调整configs/runtime.yml 文件中的 snapshot_epoch配置以减少评估次数，或训练完成后再进行评估。

**--use_vdl=True** 开启可视化

**--vdl_log_dir="./output"** 将生成的日志放在output文件夹下

开启训练

```python
python tools/train.py -c configs/ppyolo/ppyolov2_r50vd_dcn_voc.yml --eval --use_vdl=True --vdl_log_dir="./output"
```

如果中间训练中断，可以进行断点训练

```python
python tools/train.py -c configs/ppyolo/ppyolov2_r50vd_dcn_voc.yml -r /output/ppyolov2_r50vd_dcn_voc/10000
```

#### 3.5 模型评估

由于我们边训练边评估，已经保存好了最优模型:

**位置**：PaddleDetection_my/output/model_1/ppyolov2_r50vd_dcn_voc/

**.pdmodel** 是训练使用的完整模型program描述，区别于推理模型，训练模型program包含完整的网络，包括前向网络，反向网络和优化器，而推理模型program仅包含前向网络。

**.pdparams** 是训练网络的参数dict，key为变量名，value为Tensor array数值

***.pdopt**是训练优化器的参数，结构与*.pdparams一致。

评估模型

```python
python -u tools/eval.py -c configs/ppyolo/ppyolov2_r50vd_dcn_voc.yml  \ -o weights=/home/C/wangdz/PaddleDetection2/PaddleDetection_my/output/model_1/ppyolov2_r50vd_dcn_voc/best_model.pdparams
```

#### 3.6 模型测试

使用训练好的模型进行测试，也可以选择验证集中的其他图片进行测试。

使用图片进行推理测试，这里亦可以通过-c直接指定一些参数

```python
python tools/infer.py -c configs/ppyolo/ppyolov2_r50vd_dcn_voc.yml \  -o weights=/home/C/wangdz/PaddleDetection2/PaddleDetection_my/output/model_1/ppyolov2_r50vd_dcn_voc/best_model.pdparams \  --infer_img=/home/C/wangdz/PaddleDetection2/PaddleDetection_my/insulator/JPEGImages/image_3897.jpg  
```

也可以通过--infer_dir指定目录并推理目录下所有图像。

对测试结果图片进行可视化

```python
## 显示原图
img_path= "/home/C/wangdz/PaddleDetection2/PaddleDetection_my/output/image_3897.jpg" 
img = Image.open(img_path) 
plt.figure("test_img", figsize=(100,100)) 
plt.imshow(img) 
plt.show()
```

