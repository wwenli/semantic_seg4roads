# Semantic Segmentation for Roads Using DeepLab V3

## Note: For NabLab code tester, please refer to <a href='deeplab/g3doc/tester.md'>this</a><br> to perform code testing on the local machine.

In this project, I have explored current state-of-the-art methods for semantic segmentation used for road objects, such as road surface, sidewalk, lane markings, and tactile patches. 

Papers about current popular semantic segmentation methods include: 

1. [FCN](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
2. [SegNet](https://arxiv.org/pdf/1511.00561.pdf)
3. [Dilated Convolutions](http://vladlen.info/papers/dilated-convolutions.pdf)
4. [DeepLab v1](https://arxiv.org/pdf/1412.7062.pdf)
5. [DeepLab v2](https://arxiv.org/pdf/1606.00915.pdf)
6. [RefineNet](https://arxiv.org/pdf/1611.06612.pdf)
7. [PSPNet](https://arxiv.org/pdf/1612.01105.pdf)
8. [Large Kernel Matters](https://arxiv.org/pdf/1703.02719.pdf)
9. [DeepLab v3](https://arxiv.org/pdf/1706.05587.pdf)

The performance of each method is documented [here](./PERFORMANCE.md). 
By comparison, I have chosen DeepLab v3 as the main method used for road 
semantic segmentation because of 1. its state-of-the-arts performance 
and 2. its open-source availability. 

The original [DeepLab v2 source code](https://bitbucket.org/aquariusjay/deeplab-public-ver2) 
is built on top of [Caffe](http://caffe.berkeleyvision.org/), which has 
weak compatibility with other packages and frameworks limited to specific versions. Luckily, I found an 
(re-)implementation of DeepLab v2 in TensorFlow [here](https://github.com/DrSleep/tensorflow-deeplab-resnet), which 
is much easier to compile and used for customized training. Recently in March 2018, Google has released the official 
tensorflow implemented Deeplab v3 under tensorflow/models/research.

The original DeepLab v3 is trained and tested on the [Pascal VOC2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/). 
Google has trained and tested Deeplab v3 on the [Pascal VOC2012 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/), 
[Cityscapes dataset](https://www.cityscapes-dataset.com/downloads/), and [ADE20K dataset](http://groups.csail.mit.edu/vision/datasets/ADE20K/).
For the purpose of this project, the [Mapillary Vistas Dataset](https://blog.mapillary.com/product/2017/05/03/mapillary-vistas-dataset.html) 
is used for training and testing with the specific goal of semantic segmentation for roads. A comparison of the current 
publicly available semantic segmentation datasets/benchmarks is documented [here](./DATASET.md). 


## 1 Installation

### 1.1 Dependencies

DeepLab depends on the following libraries:

*   Numpy
*   Pillow 1.0
*   tf Slim (which is included in the "tensorflow/models/research/" checkout)
*   Matplotlib
*   Tensorflow

For detailed steps to install Tensorflow, follow the [Tensorflow installation
instructions](https://www.tensorflow.org/install/). A typical user can install
Tensorflow using one of the following commands:

```bash
# For CPU
pip install tensorflow
# For GPU
pip install tensorflow-gpu
```

The remaining libraries can be installed on Ubuntu 14.06 using via apt-get:

```bash
sudo apt-get install python-pil python-numpy
sudo pip install matplotlib
```

### 1.2 Add Libraries to PYTHONPATH

When running locally, the current and slim directories
should be appended to PYTHONPATH. This can be done by running the following from
tensorflow/models/research/:

```bash
# From current directory
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

Note: This command needs to run from every new terminal you start. If you wish
to avoid running this manually, you can add it as a new line to the end of your
~/.bashrc file.

### 1.3 Testing the Installation

You can test if you have successfully installed the Tensorflow DeepLab by
running the following commands:

Quick test by running model_test.py:

```bash
# From current directory
python deeplab/model_test.py
```

Quick running the whole code on the PASCAL VOC 2012 dataset:

```bash
# From deeplab/
sh local_test.sh

```

## 2 Running DeepLab on Mapillary Semantic Segmentation Dataset

To utilize DeepLab on our project, I have trained 4 models and compared their results.

The 4 models are: 
* Run DeepLab on the original Mapillary Dataset with 66 target classes
* Run DeepLab on the original Mapillary Dataset with 2 target classes (lane marking and other)
* Run DeepLab on the downsized Mapillary Dataset with 66 classes
* Run DeepLab on the downsized Mapillary Dataset with 2 target classes (lane marking and other)

### 2.1 Download Dataset and Convert to TFRecord

First, please run the following commands to fetch the mapillary dataset:

```bash
# From current directory
mkdir deeplab/datasets/mapillary
cp /media/sharedHDD/mapillary_dataset.tar.gz deeplab/datasets/mapillary
tar -xzf deeplab/datasets/mapillary/mapillary_dataset.tar.gz deeplab/datasets/mapillary/
```

We have prepared the script (under the folder `deeplab/datasets`) to
convert mapillary semantic segmentation dataset to TFRecord.

#### 2.1.1 To prepare data to train DeepLab on the original Mapillary Dataset, please run:

```bash
# From the deeplab/datasets directory.
bash convert_mapillary.sh
```

The converted dataset will be saved at
./deeplab/datasets/mapillary/tfrecord

#### 2.1.2 To prepare data to train DeepLab on the original Mapillary Dataset with 2 target classes, please run:

```bash
# From the deeplab/datasets directory.
python convert_2classes.py
bash convert_mapillary_2classes.sh
```

The converted dataset will be saved at
./deeplab/datasets/mapillary/tfrecord_lane_marking_general

#### 2.1.3 To prepare data to train DeepLab on the downsized Mapillary Dataset, please run:

```bash
# From the deeplab/datasets directory.
bash convert_mapillary_downsize.sh
```

The converted dataset will be saved at
./deeplab/datasets/mapillary/tfrecord_512_384

#### 2.1.3 To prepare data to train DeepLab on the downsized Mapillary Dataset with 2 target classes, please run:

```bash
# From the deeplab/datasets directory.
python convert_downsize_2classes.py
bash convert_mapillary_downsize_2classes.sh
```

The converted dataset will be saved at
./deeplab/datasets/mapillary/tfrecord_512_384_lane_marking_general

### 2.2 Recommended Directory Structure for Training and Evaluation

```
+ datasets
   - build_data.py
   - build_mapillary_data.py
   + mapillary 
      + tfrecord_512_384
      + exp
         + train_on_train_set
            + train
            + eval
            + vis
      + mapillary_dataset
         + training
            + images_512_384
            + labels_512_384
            + instances_512_384
         + validation
            + images_512_384
            + labels_512_384
            + instances_512_384
```

where the folder `train_on_what_model` stores the train/eval/vis events and
results (when training DeepLab on the mapillary train set).

### 2.3 Running the train/eval/vis jobs and export the trained deeplab model

We have prepared the script (under the current directory) to
do the train/eval/vis/export jobs all at once. 
A local training job using `xception_65` can be run with the following command:

#### 2.3.1 To run deeplab on the original Mapillary dataset:

First make sure in `deeplab/datasets/segmentation_dataset.py` line 111, `num_classes` is set to 66, then:

```bash
# From current directory
sh train_mapillary_raw.sh
```

#### 2.3.2 To run deeplab on the original Mapillary dataset with 2 classes:

First make sure in `deeplab/datasets/segmentation_dataset.py` line 111, `num_classes` is set to 2, then:

```bash
# From current directory
sh train_mapillary_2classes.sh
```

#### 2.3.3 To run deeplab on the downsized Mapillary dataset:

First make sure in `deeplab/datasets/segmentation_dataset.py` line 111, `num_classes` is set to 66, then:

```bash
# From current directory
sh train_mapillary_downsize.sh
```

#### 2.3.4 To run deeplab on the downsized Mapillary dataset with 2 classes:

First make sure in `deeplab/datasets/segmentation_dataset.py` line 111, `num_classes` is set to 2, then:

```bash
# From current directory
sh train_mapillary_2classes_downsize.sh
```

### 2.4 Running Tensorboard

Progress for training and evaluation jobs can be inspected using Tensorboard. If
using the recommended directory structure, Tensorboard can be run using the
following command:

```bash
tensorboard --logdir=${PATH_TO_LOG_DIRECTORY}
```

where `${PATH_TO_LOG_DIRECTORY}` points to the directory that contains the train
directorie (e.g., the folder `train_on_what_model` in the above example). Please
note it may take Tensorboard a couple minutes to populate with data.

### 2.5 Export trained deeplab model to frozen inference graph

After model training finishes, you could export it to a frozen TensorFlow
inference graph proto. Your trained model checkpoint usually includes the
following files:

*   model.ckpt-${CHECKPOINT_NUMBER}.data-00000-of-00001,
*   model.ckpt-${CHECKPOINT_NUMBER}.index
*   model.ckpt-${CHECKPOINT_NUMBER}.meta

The exported model is located at `deeplab/datasets/mapillary/exp/train_on_what_model/export`

### 2.6 Performing inference

I have provided a script called `inference_mapillary.py` that can fetch trained and saved model and perform inference on 
any arbitrary image. 

If you are using the 66 classes model, please make sure in `inference_mapillary.py`, line 147-156 are enabled and line 158-160 
are commented out. Please provide the model location at line 168 and your own image location at line 184, then run:

```bash
python inference_mapillary.py
```

## 3 Training results 

In the following subsections, I documented the training result of the 4 models. To compare their performance, 
I have trained them using the same parameters and for the same number of steps (100,000 to be specific). 
It takes approximately 10 hours to finish training 100,000 steps on a GeForce 1080 Ti GPU. 

### 3.1 <a href='deeplab/g3doc/train_on_downsize.md'>Training on down-sized Mapillary dataset.</a><br>
### 3.2 <a href='deeplab/g3doc/train_on_downsize_2classes.md'>Training on down-sized Mapillary dataset with 2 classes.</a><br>
### 3.3 <a href='deeplab/g3doc/train_on_raw.md'>Training on original Mapillary dataset.</a><br>
### 3.4 <a href='deeplab/g3doc/train_on_2classes.md'>Training on original Mapillary dataset with 2 classes.</a><br>


## References

1.  **Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs**<br />
    Liang-Chieh Chen+, George Papandreou+, Iasonas Kokkinos, Kevin Murphy, Alan L. Yuille (+ equal
    contribution). <br />
    [[link]](https://arxiv.org/abs/1412.7062). In ICLR, 2015.

2.  **DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,**
    **Atrous Convolution, and Fully Connected CRFs** <br />
    Liang-Chieh Chen+, George Papandreou+, Iasonas Kokkinos, Kevin Murphy, and Alan L Yuille (+ equal
    contribution). <br />
    [[link]](http://arxiv.org/abs/1606.00915). TPAMI 2017.

3.  **Rethinking Atrous Convolution for Semantic Image Segmentation**<br />
    Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam.<br />
    [[link]](http://arxiv.org/abs/1706.05587). arXiv: 1706.05587, 2017.

4.  **Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation**<br />
    Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam. arXiv: 1802.02611.<br />
    [[link]](https://arxiv.org/abs/1802.02611). arXiv: 1802.02611, 2018.

5.  **ParseNet: Looking Wider to See Better**<br />
    Wei Liu, Andrew Rabinovich, Alexander C Berg<br />
    [[link]](https://arxiv.org/abs/1506.04579). arXiv:1506.04579, 2015.

6.  **Pyramid Scene Parsing Network**<br />
    Hengshuang Zhao, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, Jiaya Jia<br />
    [[link]](https://arxiv.org/abs/1612.01105). In CVPR, 2017.

7.  **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate shift**<br />
    Sergey Ioffe, Christian Szegedy <br />
    [[link]](https://arxiv.org/abs/1502.03167). In ICML, 2015.

8.  **Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation**<br />
    Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen<br />
    [[link]](https://arxiv.org/abs/1801.04381). arXiv:1801.04381, 2018.

9.  **Xception: Deep Learning with Depthwise Separable Convolutions**<br />
    François Chollet<br />
    [[link]](https://arxiv.org/abs/1610.02357). In CVPR, 2017.

10. **Deformable Convolutional Networks -- COCO Detection and Segmentation Challenge 2017 Entry**<br />
    Haozhi Qi, Zheng Zhang, Bin Xiao, Han Hu, Bowen Cheng, Yichen Wei, Jifeng Dai<br />
    [[link]](http://presentations.cocodataset.org/COCO17-Detect-MSRA.pdf). ICCV COCO Challenge
    Workshop, 2017.

11. **Tensorflow: Large-Scale Machine Learning on Heterogeneous Distributed Systems**<br />
    M. Abadi, A. Agarwal, et al. <br />
    [[link]](https://arxiv.org/abs/1603.04467). arXiv:1603.04467, 2016.

12. **The Pascal Visual Object Classes Challenge – A Retrospective,** <br />
    Mark Everingham, S. M. Ali Eslami, Luc Van Gool, Christopher K. I. Williams, John
    Winn, and Andrew Zisserma. <br />
    [[link]](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/). IJCV, 2014.

13. **The Cityscapes Dataset for Semantic Urban Scene Understanding**<br />
    Cordts, Marius, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, Bernt Schiele. <br />
    [[link]](https://www.cityscapes-dataset.com/). In CVPR, 2016.
