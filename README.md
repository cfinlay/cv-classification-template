This is a template repository for running experiments in computer vision classification problems.
Included are many of the standard architectures, including [ResNet](https://arxiv.org/abs/1512.03385), [ResNeXt](https://arxiv.org/abs/1611.05431), [VGG](https://arxiv.org/abs/1409.1556)
and [AllCNN](https://arxiv.org/pdf/1412.6806.pdf).
The main starting point is `experiment_template/`,  which contains a barebones script for training a model
on one of the standard image classification datasets. You can copy this folder and modify it as necessary to suit your needs.

The folder `utils/` has some useful tools for dissecting a trained model (including a plotting script), as
well as a dispatcher tool for running many experiments sequentially.
