# Does anatomical contextual information improve 3D U-net-based brain tumor segmentation?

This repository contains the code used to obtain and organize the anatomical contextual information used to train a U-Net segmentation model (using the nnU-Net framework) to explore the potential improvements in brain tumor segmentation when using contextual information. The results showed similar performance across models, with median Dice scores around 90%, and no statistically significant improvements in overall segmentation accuracy, training time, or domain generalization when contextual information was added.

[Journal](https://doi.org/10.3390/diagnostics11071159) | [Cite](#reference)

**Abstract**

Effective, robust, and automatic tools for brain tumor segmentation are needed for the extraction of information useful in treatment planning. Recently, convolutional neural networks have shown remarkable performance in the identification of tumor regions in magnetic resonance (MR) images. Context-aware artificial intelligence is an emerging concept for the development of deep learning applications for computer-aided medical image analysis. A large portion of the current research is devoted to the development of new network architectures to improve segmentation accuracy by using context-aware mechanisms. In this work, it is investigated whether or not the addition of contextual information from the brain anatomy in the form of white matter (WM), gray matter (GM), and cerebrospinal fluid (CSF) masks and probability maps improves U-Net-based brain tumor segmentation. The BraTS2020 dataset was used to train and test two standard 3D U-Net (nnU-Net) models that, in addition to the conventional MR image modalities, used the anatomical contextual information as extra channels in the form of binary masks (CIM) or probability maps (CIP). For comparison, a baseline model (BLM) that only used the conventional MR image modalities was also trained. The impact of adding contextual information was investigated in terms of overall segmentation accuracy, model training time, domain generalization, and compensation for fewer MR modalities available for each subject. Median (mean) Dice scores of 90.2 (81.9), 90.2 (81.9), and 90.0 (82.1) were obtained on the official BraTS2020 validation dataset (125 subjects) for BLM, CIM, and CIP, respectively. Results show that there is no statistically significant difference when comparing Dice scores between the baseline model and the contextual information models (p > 0.05), even when comparing performances for high and low-grade tumors independently. In a few low-grade cases where improvement was seen, the number of false positives was reduced. Moreover, no improvements were found when considering model training time or domain generalization. Only in the case of compensation for fewer MR modalities available for each subject did the addition of anatomical contextual information significantly improve (p < 0.05) the segmentation of the whole tumor. In conclusion, there is no overall significant improvement in segmentation performance when using anatomical contextual information in the form of either binary WM, GM, and CSF masks or probability maps as extra channels.

## Table of Contents
- [Setup](#setup)
- [Dataset](#dataset)
- [Code Structure](#code-structure)
- [Citation](#reference)
- [License](#license)

## Setup
Model training and evaluation are run using the nnU-Net framework. Find the details for Python environment setup and how to run model training and evaluation at https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1. 
The anatomical contextual information is obtained using functionalities from the [FSL - FMRIB Software Library](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/). See the [installation guides](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/index) for help on how to set FSL on your machine.

## Dataset
The dataset used in this project is the BraTS2020 dataset. You can find how to obtain it at https://www.med.upenn.edu/cbica/brats2020/data.html. 

## Code Structure
The code in this repository helps with contextual information extraction and its organization to fit the nnU-Net dataset requirements.

- `run_contextual_segmentation.sh`: Script that given the path to the BraTS2020 dataset, runt the FSL FAST algorithm for white matter, grey matter and cerebrospinal fluid brain region extraction.
- `brats_to_nnUNet_datastructure_context_info.py`: Script for the preparation of the data (contextual information and anatomical MR volumes) following the nnU-Net dataset specification for the project. 

## Citation
If you use this work, please cite:

```bibtex
@article{diagnostics11071159,
    AUTHOR = {Tampu, Iulian Emil and Haj-Hosseini, Neda and Eklund, Anders},
    TITLE = {Does Anatomical Contextual Information Improve 3D U-Net-Based Brain Tumor Segmentation?},
    JOURNAL = {Diagnostics},
    VOLUME = {11},
    YEAR = {2021},
    NUMBER = {7},
    ARTICLE-NUMBER = {1159},
    URL = {https://www.mdpi.com/2075-4418/11/7/1159},
    PubMedID = {34201964},
    ISSN = {2075-4418},
    DOI = {10.3390/diagnostics11071159}
}
```
## License
This work is licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/).



