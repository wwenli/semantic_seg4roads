### 1 Training Progress

![](img/loss_downsize.png)

The model is trained on the downsized Mapillary dataset (every image is downsized into 384 by 512 pixels for faster training). 

The model is trained for 100,000 steps. It takes around 10 hours to finish the training on a GeForce 1080 Ti GPU.

The learning progress is shown above. We can see that the majority of the learning is done during the first 500 steps. 
After that, the model is trying to learn/correct fine details. The training loss fluctuates between 0.00 and 0.15.

### 2 Training results on validation set
Original Image             |  Training Result on 66 Classes | Ground Truth
:-------------------------:|:--------------:|:----------------:
![](img/000000_image.png)  |  ![](img/train_on_downsize/000000_prediction.png) | ![](img/Ar4n_0npVlDM9b5w3ymV-Q.png) 
![](img/001956_image.png)  |  ![](img/train_on_downsize/001956_prediction.png) | ![](img/w-XEZhFtU0qMVSM0yZcpmg.png) 
![](img/001963_image.png)  |  ![](img/train_on_downsize/001963_prediction.png) | ![](img/V39DAks5M0-w3FM08m1fyw.png) 
![](img/001978_image.png)  |  ![](img/train_on_downsize/001978_prediction.png) | ![](img/TjAVp3hnSQUKhcWji_bWTw.png) 
![](img/001991_image.png)  |  ![](img/train_on_downsize/001991_prediction.png) | ![](img/wRWPuTrp-_Ve55VYvuRSew.png) 

The above images are sampled from the validation set. From the comparison between the training result and the ground truth, 
we can see that the model is limited to recognize fine details. However, the model can successfully classify most of the pixels 
and have improved significantly from [this](https://github.com/Transportation-Inspection/semantic_segmentation).

### 3 Inference on KITTI Sampled Images
![](img/train_on_downsize/000022_10_inf.png)
![](img/train_on_downsize/000117_10_inf.png)
![](img/train_on_downsize/000071_10_inf.png)

The above 3 images show how this model performs on any arbitrary images. The 3 images are sampled
from the KITTI dataset. We can see that our model can intelligently recognize the lane marking mostly. However, the correctness
is still limited.