## 本项目基于[yolo3](https://github.com/pjreddie/darknet.git) 与[crnn](https://github.com/meijieru/crnn.pytorch.git)  实现中文自然场景文字检测及识别

## 由于本项目融合多个框架，维护较困难，为了更好的部署应用，后续将只围绕[darknet](https://github.com/pjreddie/darknet.git)框架进行开发，不在支持keras、tensorflow、pytorch。
###  新的项目地址：https://github.com/chineseocr/darknet-ocr.git 

# 实现功能
- [x]  文字方向检测 0、90、180、270度检测（支持dnn/tensorflow） 
- [x]  支持(darknet/opencv dnn /keras)文字检测,支持darknet/keras训练
- [x]  不定长OCR训练(英文、中英文) crnn\dense ocr 识别及训练 ,新增pytorch转keras模型代码(tools/pytorch_to_keras.py)
- [x]  支持darknet 转keras, keras转darknet, pytorch 转keras模型
- [x]  身份证/火车票结构化数据识别 
- [x]  新增CNN+ctc模型，支持DNN模块调用OCR，单行图像平均时间为0.02秒以下     
- [ ]  CPU版本加速    
- [ ]  支持基于用户字典OCR识别    
- [ ]  新增语言模型修正OCR识别结果  
- [ ]  支持树莓派实时识别方案  
 

## 环境部署

GPU部署 参考:setup.md     
CPU部署 参考:setup-cpu.md   


### 下载编译darknet(如果直接运用opencv dnn或者keras yolo3 可忽略darknet的编译)  
```
git clone https://github.com/pjreddie/darknet.git 
mv darknet chineseocr/
##编译对GPU、cudnn的支持 修改 Makefile
#GPU=1
#CUDNN=1
#OPENCV=0
#OPENMP=0
make 
```

修改 darknet/python/darknet.py line 48    
root = '/root/'##chineseocr所在目录     
lib = CDLL(root+"chineseocr/darknet/libdarknet.so", RTLD_GLOBAL)    


## 下载模型文件   
模型文件地址:
* [baidu pan](https://pan.baidu.com/s/1gTW9gwJR6hlwTuyB6nCkzQ)

复制文件夹中的所有文件到models目录
   
## 模型转换
pytorch ocr 转keras ocr     
``` Bash
python tools/pytorch_to_keras.py  -weights_path models/ocr-dense.pth -output_path models/ocr-dense-keras.h5
```
darknet 转keras     
``` Bash
python tools/darknet_to_keras.py -cfg_path models/text.cfg -weights_path models/text.weights -output_path models/text.h5
```
keras 转darknet      
``` Bash
python tools/keras_to_darknet.py -cfg_path models/text.cfg -weights_path models/text.h5 -output_path models/text.weights
```

## 编译语言模型(可选)
``` Bash
git clone --recursive https://github.com/parlance/ctcdecode.git   
cd ctcdecode   
pip install .  
```
## 下载语言模型 (可选)
``` Bash
wget https://deepspeech.bj.bcebos.com/zh_lm/zh_giga.no_cna_cmn.prune01244.klm
mv zh_giga.no_cna_cmn.prune01244.klm chineseocr/models/
```
## 模型选择  
``` Bash
参考config.py文件
```  

## 构建docker镜像 
``` Bash
##下载Anaconda3 python 环境安装包（https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh） 放置在chineseocr目录下   
##建立镜像   
docker build -t chineseocr .   
##启动服务   
docker run -d -p 8080:8080 chineseocr /root/anaconda3/bin/python app.py

```

## web服务启动
``` Bash
cd chineseocr## 进入chineseocr目录
python app.py 8080 ##8080端口号，可以设置任意端口
```

## 访问服务
http://127.0.0.1:8080/ocr

<img width="500" height="300" src="https://github.com/chineseocr/chineseocr/blob/master/test/demo.png"/>



## 识别结果展示

<img width="500" height="300" src="https://github.com/chineseocr/chineseocr/blob/master/test/train-demo.png"/>
<img width="500" height="300" src="https://github.com/chineseocr/chineseocr/blob/master/test/idcard-demo.png"/>
<img width="500" height="300" src="https://github.com/chineseocr/chineseocr/blob/master/test/img-demo.png"/>
<img width="500" height="300" src="https://github.com/chineseocr/chineseocr/blob/master/test/line-demo.png"/>



## 参考
1. yolo3 https://github.com/pjreddie/darknet.git   
2. crnn  https://github.com/meijieru/crnn.pytorch.git              
3. ctpn  https://github.com/eragonruan/text-detection-ctpn    
4. CTPN  https://github.com/tianzhi0549/CTPN       
5. keras yolo3 https://github.com/qqwweee/keras-yolo3.git    
6. darknet keras 模型转换参考 参考：https://www.cnblogs.com/shouhuxianjian/p/10567201.html  
7. 语言模型实现 https://github.com/lukhy/masr


## Note in anaconda env

```
conda install cudatoolkit nvcc_linux-64 -c conda-forge
```

ADD CFLAGS and LDFLAGS to Makefile in darknet:

```
# Add this to ARCH, for 2080 Ti whose capability is 7.5
ARCH= -gencode arch=compute_75,code=[sm_75,compute_75]

# fix in anaconda env
CFLAGS+= -isystem /data/xiaowentao/.anaconda3/include
LDFLAGS+= -L/data/xiaowentao/.anaconda3/lib -Wl,-rpath,/data/xiaowentao/.anaconda3/lib -Wl,-rpath-link,/data/xiaowentao/.anaconda3/lib
NVCCFLAGS= --compiler-options -fPIC -isystem /data/xiaowentao/.anaconda3/include

# $(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@
$(NVCC) $(ARCH) $(COMMON) $(NVCCFLAGS) -c $< -o $@
```

If you encounter the error:

```
nvcc fatal   : A single input file is required for a non-link phase when an outputfile is specified
```

Maybe you can check is there is any non-ascii character... e.g. '-' instead of '–'.

Note that darknet failed with 'CUDA Error: invalid device symbol' when I used CUDA 10.1 installed from anaconda and 2080 Ti.
Downgrade CUDA to 10.0 fixed this error.

```
conda install cudatoolkit==10.0.130 -c anaconda
# provide nvcc
conda install cudatoolkit-dev==10.0 -c conda-forge
conda install opencv -c conda-forge
# ensure pkg-config --libs opencv work correctly
conda install pkgconfig -c anaconda
```

For installing swig decoder:

```
conda install boost -c anaconda && bash setup.sh && pip install .
```
