---
title: 'Lightning-UDA-Detect: Easily run unsupervised domain adaptation object detection'
tags:
  - Python
  - computer vision
  - unsupervised domain adaptation
  - object detection
authors:
  - name: Eoghan Mulcahy
    equal-contrib: true
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

Here, we present a library for easily running unsupervised domain adaptation based object detection. It is made easy to install and includes automated provenance through configuration files and built-in experiment tracking. Unsupervised domain adaptation (UDA) can be used to transfer knowledge from a source domain with annotated data to a target domain with unlabelled data only. This is particulary useful in real-world application based settings as annotation tasks are widely acknowledged to be arduous and time-consuming. Within recent years, several algorithms were developed to utilize UDA specifically for object detection. However, implementations are difficult to install and lack user friendliness and provenance. Lightning-UDA-Detect is designed to allow researches to easily interact with several popular UDA architechtures for object detection tasks.

# Statement of need

Todo

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

Todo

# References