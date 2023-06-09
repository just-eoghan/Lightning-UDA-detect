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
 - name: Health Research Institute, University of Limerick, Limerick, V94T9PX, Ireland
   index: 2
date: 28 April 2023
bibliography: paper.bib
---

# Summary

Here, we present the *Lightning-UDA-Detect* library, designed to easily run unsupervised domain adaptation (UDA) based object detection. *Lightning-UDA-Detect* is structured for straightforward installation and includes automated provenance through configuration files and built-in experiment tracking. UDA can transfer knowledge from a source domain with annotated data to a target domain with unlabeled data only. UDA is particularly useful in real-world application-based settings, as annotation tasks are arduous and time-consuming. Several algorithms have been developed to utilize UDA specifically for object detection in recent years. However, implementations are difficult to install and lack user-friendliness. *Lightning-UDA-Detect* allows researchers to interact easily with and extend several popular UDA architectures specifically for object detection tasks. We present evidence that our library is proven to achieve the officially reported performance of several UDA architectures under the mAP@50 metric, the standard metric for measuring object detector performance.

# Statement of need

The seminal paper for the task of unsupervised domain adaptive object detection "Domain Adaptive Faster R-CNN for Object Detection in the Wild" (DA) [@Chen2018] extended Faster-RCNN [@NIPS2015_14bfa6bb] to use multi-level domain classifiers which enforce the learning of domain invariant features between a labeled source domain and an unlabeled target domain. The DA model was further extended by [@Chen_2021] to incorporate object scaling into the domain classifiers in "Scale-Aware Domain Adaptive Faster R-CNN" (SADA) which improved performance by 2.7 mAP@50. Finally, SADA was extended by [@hoyer2023mic] to create "Masked Image Consistency for Context-Enhanced Domain Adaptation" (MIC), which achieves current state of the art performance by extending the previous methods and adding an extra module to learn spatial context relations of the target domain. MIC improves on SADA by 3.6 mAP@50. Each of these implementations is important to the community of researchers who work with unsupervised domain adaptation for object detection.

The official implementations for DA [@krumo], SADA [@chen] and MIC [@hoyer] are all built on maskrcnn-benchmark [@massa2018mrcnn] which has been deprecated and poses many compilation issues to the average user. Hence, installing and running these implementations is difficult, which impedes research and further advancements in UDA. 

*Lightning-UDA-Detect* integrates with the stable and popular package torchvision [@torchvision2016], part of the Pytorch [@NEURIPS2019_9015] project. 
Integrating with torchvision reduces the dependencies of *Lightning-UDA-Detect* and removes the need to compile from source. *Lightning-UDA-Detect* unifies three important architectures of unsupervised domain adaptation detection in an easy-to-run, installable and performance tested package. *Lightning-UDA-Detect* reduces the barrier to entry for working with UDA-based detection models.

# Features & Functionality

- *Lightning-UDA-Detect* uses Hydra [@Yadan2019Hydra] configuration files. Hydra allows for straightforward modification of experiment variables through hierarchical configuration by composition. This approach ensures provenance by creating a human-readable record of the experiment parameters which are separate and independent of the source code.

- *Lightning-UDA-Detect* has built-in online logging with *Weights \& Biases* [@wandb]. When running experiments, *Lightning-UDA-Detect* writes configuration values to an immutable run-file. Throughout runs, all relevant metrics e.g. training loss, validation accuracy and GPU power usage, are stored in the online logger. This run history provides proof and ease of reproducibility of results.

- *Lightning-UDA-Detect* integrates with Pytorch-Lightning [@W.Falconetal2022] a deep learning framework that facilitates automated usage of best practices. It also makes code more readable and understandable by abstracting model engineering code into standard functions e.g. `training_steps`, `validation_steps` and `process_data` .

![Mean Average Precision \@50.\label{fig:map50}](lit-uda-map50.pdf)

\autoref{fig:map50} outlines the mAP@50 validation graph during training for each model. Mean average precision (mAP) is determined by calculating the average precision (AP), which is defined by how well a predicted bounding box aligns with the ground truth bounding box considering at least a 50% overlap threshold, each of these AP scores is then averaged to calculate the overall mAP@50. We used torchmetrics [@torchmetrics] to calculate our mAP values. We ran all models on a single NVIDIA GeForce RTX 2080 SUPER GPU. We trained each model for 60,000 steps, using the same seed for every run. 

<!-- The mAP@50 validation graph during training for each model can be seen in \autoref{fig:map50}. Mean average precision (mAP) is determined by calculating the average precision (AP) which is defined by how well a predicted bounding box aligns with the ground truth bounding box considering at least a 50% overlap threshold, each of these AP scores are then averaged to calculate the overall mAP@50. We used torchmetrics [@torchmetrics] to calculate our mAP values. All models were run on one NVIDIA GeForce RTX 2080 SUPER GPU. Models were trained for 60,000 steps with the same seed for each run. -->

![Max Mean Average Precision \@50. \label{fig:maxMap50}](map50-bar-plot.pdf)

\autoref{fig:maxMap50} shows the max mAP@50 scores for each model . Our DA model achieved a score of 41.85 mAP@50 on the Cityscapes dataset, which is 0.55 higher than the officially reported value of 41.30. Our SADA model scored 44.35 mAP@50, 0.35 higher than the officially reported value of 44.00 mAP@50. Finally, our MIC model achieved a mAP@50 score of 48.14 which is 0.54 higher than the official value of 47.60. Our results show that our re-implementations can be trusted to perform correctly for each model.

<!-- The max mAP@50 scores for each model can be seen in \autoref{fig:maxMap50}. Our DA model achieved a score of 41.85 mAP@50 on the *Cityscapes* dataset which is 0.55 higher than the officially reported value of 41.30.  Our SADA model achieved a score of 44.35 mAP@50 which is 0.35 higher than the officially reported value of 44.00 mAP@50. Finally our MIC model achieved a mAP@50 score of 48.14 which is 0.54 higher than the official value of 47.60. Our results show our re-implementations can be trusted to perform correctly for each of the models. -->

# Further Development

In future other UDA architectures such as 02net [@gong2022improving] and ILLUME [@khindkar2022miss], could be added to the package, along with new yet-to-be-released architectures. Contributions to the package are warmly welcome; please open an issue to discuss new features.

# References