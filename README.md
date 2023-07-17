# Research: Improving Bounding Box and Label Predictions in LiDAR Images

This project aims to present improvements to existing methods for predicting bounding boxes and labels of objects in LiDAR images. These enhancements are crucial for real-world applications such as autonomous driving, by improving how autonomous cars perceive and predict objects in a given frame. Our research employs a mix of strategies, such as saliency maps for data augmentation, implementing current deep learning architectures, and incorporating Focal Loss for better minority class prediction. We focus on improving upon the KITI dataset.

## Table of Contents
1. [Introduction](#introduction)
2. [Background](#background)
3. [Motivation](#motivation)
4. [Approach](#approach)

## Introduction <a name="introduction"></a>

Our project aims to improve the bounding box and classification performance of object detection models. We experiment with a variety of models, data augmentation algorithms, and loss functions. We use ensemble learners, YoloV5, and a novel data augmentation technique incorporating saliency map algorithms. Furthermore, we handle class imbalance with the implementation of Focal Loss.

## Background <a name="background"></a>

The research area of image classification in LiDAR and RBG image spaces is filled with various approaches. We discuss a few, including PointNet, Point-RCNN, VoxelNet, SECOND, PointPillars, and PV-RCNN. Our work also focuses on the issue of class imbalance in the current SOTA architectures and highlights the importance of dealing with it effectively.

## Motivation <a name="motivation"></a>

The exploration of this area is primarily for its applications to self-driving cars. Our aim is to improve the robustness of object detection using SOTA deep learning research. Our results in ensemble learning can yield higher confidence and more accurate predictions. This can translate to significant differences in existing models, where small improvements can enhance self-driving cars.

## Approach <a name="approach"></a>

Our approach utilizes a 3D point cloud LiDAR based object detection model called SECOND. We add value by implementing model ensemble, saliency map augmentation, and focal loss. We also make an effort to ensemble different 3D object detection models like SECOND with PointPillars and SECOND with PV-RCNN. We use soft non-maximal suppression to combine the 3D bounding box predictions from both models. Focal loss was introduced to deal with the issue of imbalanced data. Lastly, we employed a saliency map algorithm to augment LiDAR images before they are sent to models for training and testing.

## Challenges and Future Work

We encountered issues in analyzing and processing large datasets due to time and compute power constraints. However, this has led us to focus on improving the results of the KITI dataset. Future efforts will continue to experiment with hyper-parameter tuning and further training iterations. We also aim to incorporate a model that combines both LiDAR pointclouds and images for enhanced bounding box predictions.
