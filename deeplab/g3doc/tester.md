First of all, please take some time and read through the previous documentation. HOWEVER, PLEASE DO 
NOT RUN ANY CODE/COMMAND LISTED IN THAT DOCUMENTATION! On the local machine, most of the data pre-processing 
results are stored. Some of them takes 10+ hours to convert, and would be meaningless to redo them. 

# 1 Environment 

Please login to NavLab and run:
```bash
cd /media/sharedHDD/semantic_segmentation
```

Within this directory, you would find: 
```
+semantic_segmentation
    +deeplab
        +core
        +datasets
            +mapillary
                +exp
                    +train_on_2classes
                    +train_on_downsize
                    +train_on_downsize_2classes
                    +train_on_raw
                +init_models
                +tfrecord
                +tfrecord_512_384
                +tfrecord_512_384_lane_marking_general
                +tfrecord_lane_marking_general
        +g3doc
        +utils
    +slim
```

Please note that the data pre-processing results are stored as tfrecord with their specific tasks. 
The 4 models are stored under `exp`. 

# 2 Code Testing
The testing would be performed on the downsized Mapillary dataset with 66 target classes. 
I have provided a script in `semantic_segmentation/local_test.sh` that performs training, evaluation, 
visualization, and inference. Please make sure you are in `semantic_segmentation/` and run:
```bash
sh local_test.sh
```
You should first see the model training for 30 steps, followed by evaluation and visualization process. 
It ends by popping up an inference image similar to the ones included in my documentation. 