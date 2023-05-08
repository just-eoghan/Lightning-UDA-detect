---
title: 'Lightning-UDA-Detect: Easily run unsupervised domain adaptation object detection'
tags:
  - Python
  - computer vision
  - unsupervised domain adaptation
  - object detection
authors:
  - name: Eoghan Mulcahy
    corresponding: true
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: John Nelson
    corresponding: false # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Pepijn Van de Ven
    corresponding: false # (This is how to denote the corresponding author)
    affiliation: "1, 2"
affiliations:
 - name: Electronic & Computer Engineering, University of Limerick, Limerick, V94T9PX, Ireland
   index: 1
 - name: Health Research Institute, Universtiy of Limerick, Limerick, V94T9PX, Ireland
   index: 2
date: 28 April 2023
bibliography: paper.bib
---

# Summary

Here, we present a library for easily running unsupervised domain adaptation (UDA) object detection. It is designed to be easy to install and includes automated provenance through configuration files and built-in experiment tracking. UDA can be used to transfer knowledge from a source domain with annotated data to a target domain with unlabelled data only. This is particulary useful in real-world application based settings as annotation tasks are widely acknowledged to be arduous and time-consuming. Within recent years, several algorithms were developed to utilize UDA specifically for object detection. However, implementations are difficult to install and lack user friendliness. Lightning-UDA-Detect is designed to allow researchers to easily interact with and extend several popular UDA architechtures specifically for object detection tasks.

# Statement of need

The deep learning object detection architechture Faster-RCNN has been extended to use UDA over the past number of years. The seminal paper for the task of unsupervised domain adaptive object detection "Domain Adaptive Faster R-CNN for Object Detection in the Wild" (DA) [@Chen2018] extended Faster-RCNN to use multi-level domain classifiers which enforce the learning of domain invarient features between a labelled source domain and an unlabelled target domain. This model was further extended to incorporate object scaling into the domain classifiers in "Scale-Aware Domain Adaptive Faster R-CNN" (SADA) [@Chen_2021] which improved performance. "Masked Image Consistency for Context-Enhanced Domain Adaptation" (MIC) [@hoyer2023mic] achieves current state of the art performance by extending the previous methods and adding an extra module to learn spatial context relations of the target domain. Each of these implementations are important to the community of researchers who work with unsupervised domain adapataion for object detection.

Each of the official implementations for DA [@krumo], SADA [@chen] and MIC [@hoyer] are built on maskrcnn-benchmark [@massa2018mrcnn] which has many compilation issues and has also been deperecated. This makes the installation and running of these important implementations difficult. 

Lightning-UDA-Detect integrates with the stable and popular package torchvision [@torchvision2016] which is part of the Pytorch [@NEURIPS2019_9015] project. This reduces the dependancies of Lightning-UDA-Detect and removes the need to compile from source. Lightning-UDA-Detect unifies three important architechtures of unsupervised domain adaptation based detection in an easy to install and run package. This reduces the barrier to entry for working with UDA based detection models.


# Features & Functionality

![Mean Average Precision \@50.\label{fig:map50}](lit-uda-map50.pdf)

The MAP@50 validation graph during training for each model can be seen in \autoref{fig:map50}.

![Max Mean Avearge Precision \@50. \label{fig:maxMap50}](map50-bar-plot.pdf)

The max MAP@50 scores for each model can be seen in \autoref{fig:maxMap50}.


# References