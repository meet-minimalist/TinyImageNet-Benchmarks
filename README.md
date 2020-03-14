# TinyImageNet Benchmarks
In this repo, I have benchmarked various computer vision architectures on Tiny ImageNet dataset.

# TinyImageNet
This dataset consists of 200 classes from original ImageNet dataset. Each class is having 500 train images, 50 validation images. So 1,00,000 images for training and 10,000 images for validation.
Download link for the dataset: https://tiny-imagenet.herokuapp.com/

# Benchmark Results

| Sr. No. |        Model        | Implemented | Top-1 Acc (Train) | Top-5 Acc (Train) | Top-1 Acc (Test) | Top-5 Acc (Test) |
|---------|---------------------|-------------|-------------------|-------------------|------------------|------------------|
|       1 | VGG16               | Y           |                   |                   |                  |                  |
|       2 | VGG16_bn            | Y           |                   |                   |                  |                  |
|       3 | ResNet18            | Y           |                   |                   |                  |                  |
|       4 | ResNet34            | Y           |                   |                   |                  |                  |
|       5 | ResNet50            | Cancelled   |                   |                   |                  |                  |
|       6 | ResNext50           | Cancelled   |                   |                   |                  |                  |
|       7 | ResNet18_v2         | Y           |                   |                   |                  |                  |
|       8 | ResNet34_v2         | Y           |                   |                   |                  |                  |
|       9 | Inception_v1        | Y           |                   |                   |                  |                  |
|      10 | Inception_v2        | Y           |                   |                   |                  |                  |
|      11 | Inception_v3        | Y           |                   |                   |                  |                  |
|      12 | Inception_v4        | Y           |                   |                   |                  |                  |
|      13 | Inception_ResNet_v2 | Y           |                   |                   |                  |                  |
|      14 | Xception            | Y           |                   |                   |                  |                  |
|      15 | MobileNet_v1        | Y           |                   |                   |                  |                  |
|      16 | MobileNet_v2        | Y           |                   |                   |                  |                  |
|      17 | MobileNet_v3        | Y           |                   |                   |                  |                  |
|      18 | ShuffleNet          | Cancelled   |                   |                   |                  |                  |
|      19 | SqueezeNet          | Y           |                   |                   |                  |                  |
|      20 | NASNet              | Y           |                   |                   |                  |                  |
|      21 | MNASNet             | Y           |                   |                   |                  |                  |
|      22 | EfficientNet        | Y           |                   |                   |                  |                  |



### Todos
 - [-] Train each model and update the accuracies
