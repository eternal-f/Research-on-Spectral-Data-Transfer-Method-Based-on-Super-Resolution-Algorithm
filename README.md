# Research on Spectral Data Transfer Method Based on Super-Resolution Algorithm  
This project explores the use of super-resolution algorithms to enhance the precision of spectral data obtained from Martian rovers (e.g., Chemcam and Supercam). By learning the mapping between spectral data from different instruments or environments, the proposed method enables effective spectral data transmission and compatibility improvement for downstream tasks such as material identification and compositional analysis.  
  
#  Project Overview  
Spectral data collected by different devices or under varying conditions often exhibit discrepancies that hinder their combined use. This project addresses the problem by applying deep learning-based super-resolution techniques to transfer spectral data from a source domain (e.g., Chemcam) to a target domain (e.g., Supercam), improving both resolution and consistency.  

## Key contributions:  
Built and compared multiple 1D super-resolution models tailored for spectral data.  
Explored both 1D and 2D (pseudo-grayscale) data representations.  
Proposed a Guided Attention mechanism to enhance performance on small-sample datasets.  
Achieved state-of-the-art PSNR scores (up to 79.49 on large datasets and 77.49 on small datasets).  

## Supported Super-Resolution Models  
The project includes implementations of the following models, each organized in its own subfolder under src/:  

**SRCNN**: Baseline CNN-based super-resolution    
**DRN**: Dual Regression Network  
**DRRN**: Deep Recursive Residual Network  
**GAIN**: Guided Attention Network   
**MSRN**: Multi-Scale Residual Network  
**RCAN**: Very Deep Residual Channel Attention Network  
**SRDenseNet**: Dense Network for Super-Resolution  
Each subfolder follows a consistent architecture:  
  
Main  
src/<model>/  
├── dataset.py -----   Data loading and preprocessing  
├── model.py ------    Model architecture definition  
├── prepare.py ---     Initialization of model and data  
├── train.py -----     Training loop  
├── test.py ------     Evaluation and visualization  
├── main.py ------     Entry point (placeholder)  
└── utils.py -----     Utility functions  

## Datasets  
The data/ directory contains spectral data used in this study:  
chemcam — input spectral data  
supercam — target spectral data (ground truth)  

## Key Results  
1D models significantly outperform 2D transformations for spectral data.  
The proposed Guided Attention model achieves the best performance:  
Large sample PSNR: 79.49  
Small sample PSNR: 77.49  
Outperforms traditional bicubic interpolation (PSNR ~63) and baseline CNNs (PSNR ~65–75).  

## Repository Structure
Main  
├── data ---------     Spectral datasets (Chemcam, Supercam)  
├── report ------      Project report (PDF)  
├── src ---------      Source code for all models  
│   ├── SRCNN/  
│   ├── DRN/  
│   ├── GAIN/  
│   └── ...  
└── README.md ------   Description  
