## Cifar10 CNN

###### About cifar10 dataset
* consist 60000 images
* classes : 10
* shape of image 32x32x3

###### Requirements
* python 3.x
* Flask==1.1.1
* tensorflow==2.0.0

###### Model Architecture
* 3 convolutional layer, 3 max pool layer, 2 dense layer
* stride : 1
* padding : 1
* activation function : relu
* optimazer : adam

**1st convolutional layer**
* input : 32x32x3
* filter : 3x3x3x10 
* output : 32x32x10

**1st MaxPool layer**
* stride : 2
* input : 32x32x10
* output : 16x16x10

**2nd convolutional layer**
* input : 16x16x10
* filter : 3x3x10x20 
* output : 16x16x20

**2nd MaxPool layer**
* stride : 2
* input : 16x16x20
* output : 8x8x20

**3rd convolutional layer**
* input : 8x8x20
* filter : 3x3x20x40 
* output : 8x8x40

**3rd MaxPool layer**
* stride : 2
* input : 8x8x40
* output : 4x4x40

**1st Hidden layer**
* input : 640
* output : 280

**2nd Hidden layer**
* input : 28
* output : 80
*10 Target

........



