# TinyImageNet Benchmarks:
In this repo, I have benchmarked various computer vision architectures on Tiny ImageNet dataset.

### TinyImageNet:
This dataset consists of 200 classes from original ImageNet dataset. Each class is having 500 train images, 50 validation images. So 1,00,000 images for training and 10,000 images for validation.
Download link for the dataset: https://tiny-imagenet.herokuapp.com/

### Get Started:
![](https://github.com/meet-minimalist/TinyImageNet-Benchmarks/blob/master/misc_utils/get_started.png)

### Benchmark Results:

**Sr. No.**|**Model**|**Train Top-1 Acc**|**Train Top-5 Acc**|**Test Top-1 Acc**|**Test Top-5 Acc**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
1|VGG16|57.45|80.577|38.75|63.33
2|VGG16\_bn|71.597|91.825|48.15|73.26
3|ResNet18\_wo\_residual|32.611|59.604|28.7|55.19
4|ResNet34\_wo\_residual|12.734|33.818|12.67|33.46
5|ResNet18|54.039|80.179|41.45|67.96
6|ResNet34|54.601|80.423|43.11|69.45
7|ResNet50 (Cancelled)|--|--|--|-- 
8|ResNext50 (Cancelled)|--|--|--|--
9|ResNet18\_v2|55.325|81.294|41.65|68.37
10|ResNet34\_v2|53.534|79.917|42.09|69.07
11|Inception\_v1|34.399|62.234|22.59|46.36
12|Inception\_v2|51.345|78.046|38.41|64.77
13|Inception\_v3|44.529|69.91|35.76|59.83
14|Inception\_v4|55.42|81.526|31.07|55.5
15|Inception\_ResNet\_v2|36.87|63.467|27.02|51.12
16|Xception|74.397|93.696|36.12|60.83
17|MobileNet\_v1|22.824|48.799|21.86|46.57
18|MobileNet\_v2|43.48|71.792|33.13|59.46
19|MobileNet\_v3\_small|38.502|66.87|29.52|55.08
20|MobileNet\_v3\_large|36.321|64.099|27.73|53.1
21|ShuffleNet (Cancelled)|--|--|--|-- 
22|SqueezeNet|11.024|24.786|10.39|24.2
23|NASNet|34.684|61.895|30.39|56.02
24|MNASNet|34.463|61.932|30.96|57.6
25|EfficientNet|43.359|70.631|36.9|64.52

### Accuracy Plots:
- VGG16 and VGG16_batch_norm
![](https://github.com/meet-minimalist/TinyImageNet-Benchmarks/blob/master/misc_utils/plots/Vgg16.png)
- ResNet without residuals and ResNet with residual connections
![](https://github.com/meet-minimalist/TinyImageNet-Benchmarks/blob/master/misc_utils/plots/ResNets_w_wo_residual.png)
- ResNet_v1 and ResNet_v2
![](https://github.com/meet-minimalist/TinyImageNet-Benchmarks/blob/master/misc_utils/plots/ResNets.png)
- Various Inception architectures
![](https://github.com/meet-minimalist/TinyImageNet-Benchmarks/blob/master/misc_utils/plots/Inceptions.png)
- Various Lightweight models
![](https://github.com/meet-minimalist/TinyImageNet-Benchmarks/blob/master/misc_utils/plots/Light_weight.png)
- Comparision of All models
![](https://github.com/meet-minimalist/TinyImageNet-Benchmarks/blob/master/misc_utils/plots/all_models.png)


### Implementation detail:
- All the models have been trained using Adam Optimizer with the batch size of 256.
- The learning rate used in all the models is cyclic learning rate. For each model a dry run for 2000 steps have been made to identify the min and max LR for cyclic learning rate.
- Each model have been trained for 200 epochs
- First dry_run for cyclic learning rate will be performed. From that min max LR will be decided. Then the same will be updated in a particular model config file. Then, the training of that model will be carry out. At the end of training, with the help of best saved checkpoint, the evaluation of model on training and test set will be done. All this can be done with the help of run_classifier.py file only.
- The tfrecord have been created in following way.
    - First run the script "./misc_utils/create_train_test_ann.py" to generate txt file for training and test set annotations. This will be used for tfrecord generation.
    - Now uncomment line 63-64 in "tfrecords_helper.py" and run file to generate tfrecords for TinyImageNet or any other dataset at given location.
- Note : Here, the test set means validation set which have been provided from TinyImageNet site, which has labels in it. The actual test set from TinyImageNet site doesn't contain any labels, so we have used validation set as a test set during training.


### File structre:
- data_aug
    - Folder containing data augmentation library
- misc_utils
    - Folder for plotting the final accuracies based on ""./paper/Benchmark Results.xls" file.
- models
    - Folder having implementation of all the individual models
- paper
    - Folder contains pdf files of papers for various models 
- TinyImageNetClassifier.py
    - Contains model dry_run, training and evaluation scripts
- config.py
    - Contains general configuration for training
- tfrecords_helper.py
    - Contains input data pipeline along with augmentation. 
- run_classifier.py
    - for model training initial point.

### References:
- Data augmentation library taken from : https://github.com/Paperspace/DataAugmentationForObjectDetection
- Paper references
    - VGG16 [https://arxiv.org/pdf/1409.1556]
    - ResNet [https://arxiv.org/abs/1512.03385]
    - ResNet_v2 [https://arxiv.org/abs/1603.05027]
    - Inception_v1 [https://arxiv.org/abs/1409.4842]
    - Inception_v2, Inception_v3 [https://arxiv.org/abs/1512.00567]
    - Inception_v4, Inception_ResNet_v2 [https://arxiv.org/abs/1602.07261]
    - Xception [https://arxiv.org/abs/1610.02357]
    - MobileNet_v1 [https://arxiv.org/abs/1704.04861]
    - MobileNet_v2 [https://arxiv.org/abs/1801.04381]
    - MobileNet_v3 [https://arxiv.org/abs/1905.02244]
    - SqueezeNet [https://arxiv.org/abs/1602.07360]
    - NASNet [https://arxiv.org/abs/1707.07012]
    - MNASNet [https://arxiv.org/abs/1807.11626]
    - EfficientNet [https://arxiv.org/abs/1905.11946]
    - Shufflenet [https://arxiv.org/abs/1707.01083]
    - ResNext [https://arxiv.org/abs/1611.05431]

### Todos
 - [x] Train each model and update the accuracies

