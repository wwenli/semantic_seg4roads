# Datasets and Benchmarks for Semantic Segmentation Task

| Dataset/ Benchmark | Stored Location | Complexity | Volume | Resolution | Response time for Dataset Request|
| --- | --- | --- | --- | --- | --- |
|[Mapillary Vistas Dataset](http://blog.mapillary.com/product/2017/05/03/mapillary-vistas-dataset.html) | /media/sharedHDD/mapillary_dataset|66 object categories  (Including crosswalk-zebra, marking general…)|25,000 densely annotated street level images|diverse (sample:3984 x 2988)| 8 days|
|[SYNTHIA Dataset](http://synthia-dataset.net/) | N.A. |SYNTHIA-RAND-CITYSCAPES contains lane-marking |9,000|1280 × 760 |immediately |
|[Cityscapes](https://www.cityscapes-dataset.com/dataset-overview/) [benchmark](https://www.cityscapes-dataset.com/benchmarks/#scene-labeling-task)(urban street scenes) | /media/sharedHDD/cityscapes | [30 classes](https://www.cityscapes-dataset.com/dataset-overview/#class-definitions) (actually 19 are used) (road, vegetation, sidewalk...)|[5000 fine annotated images](https://www.cityscapes-dataset.com/examples/#fine-annotations) & [20000 coarse annotated images](https://www.cityscapes-dataset.com/examples/#coarse-annotations)|2048 x 1024| 2 hours |
|[Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) [benchmark](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?cls=mean&challengeid=11&compid=6&submid=12345) | /media/sharedHDD/VOC/VOCdevkit/VOC2012| [20 object categories and one background](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/segexamples/index.html) (mainly for vehicles, person and animals, `NO road vegetation sidewalk`)|Augmented version: 10582 for training and 1449 for validation|
|[KITTI:Road/Lane Detection Evaluation 2013](http://www.cvlibs.net/datasets/kitti/eval_road.php)| N.A. | ONLY road| 289 training images|1242 × 375|immediately|

