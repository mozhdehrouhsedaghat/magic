# MAGIC: Mask-Guided Image Synthesis by Inverting a Quasi-Robust Classifier

This repository contains code for the following paper:

+ Rouhsedaghat M, Monajatipoor M, Kuo CC, Masi I. [MAGIC: Mask-Guided Image Synthesis by Inverting a Quasi-Robust Classifier](https://arxiv.org/abs/2209.11549). AAAI23

MAGIC allows a diverse set of image synthesis tasks following the semantic of objects and scenes requiring a single image, its binary segmentation source mask, and
a target mask. In each pair, the left image is the input, and the right one is the manipulated image, guided by the mask shown on top. a) position control and copy/move manipulation; b) shape control on object (non repetitive); c) shape control on  scene (repetitive) images.

For each input, we fix the mask and start the synthesis from three different starting points while observing the boundaries specified by the target mask and generating realistic images, MAGIC keeps specificity and generates diverse results.

This repo is in its initial stage, welcome bug reports to rouhseda@usc.edu

Thanks~
