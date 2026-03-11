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
src/model_name/  
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
<img width="1748" height="876" alt="QQ20260311-144751" src="https://github.com/user-attachments/assets/a36094bc-98e1-4d27-b611-77c54a9e8464" />

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

# 基于超分辨率算法的光谱数据传递方法研究  
本项目探索利用超分辨率算法提升火星车获取的光谱数据（如Chemcam和Supercam）的精度。通过学习不同仪器或环境下光谱数据之间的映射关系，所提出的方法能够实现有效的光谱数据传输，并提升后续任务（如物质识别和成分分析）的兼容性。  

## 项目概述  
在不同设备或条件下收集的光谱数据通常存在差异，阻碍了它们的联合使用。本项目通过应用基于深度学习的超分辨率技术，将光谱数据从源域（例如Chemcam）迁移到目标域（例如Supercam），从而提高分辨率并增强数据一致性。  
  
## 主要工作：  
构建并比较了多种针对光谱数据的一维超分辨率模型。  
探索了一维和二维（伪灰度图）两种数据表示形式。  
提出了一种引导注意力机制，以提升在小样本数据集上的性能。  
取得了先进的峰值信噪比分数（在大型数据集上高达79.49，在小型数据集上高达77.49）。  

## 相关模型  
本项目包含以下模型的实现，每个模型都组织在src/下的独立子文件夹中：  
**SRCNN**：Baseline CNN-based super-resolution  
**DRN**：Dual Regression Network  
**DRRN**：Deep Recursive Residual Network  
**GAIN**：Guided Attention Network  
**MSRN**：Multi-Scale Residual Network  
**RCAN**：Very Deep Residual Channel Attention Network  
**SRDenseNet**：Dense Network for Super-Resolution  

每个子文件夹遵循一致的架构：  
Main  
src/model_name/    
├── dataset.py -----   数据加载与预处理    
├── model.py ------    模型架构定义    
├── prepare.py ---     模型与数据初始化    
├── train.py -----     训练  
├── test.py ------     测试  
├── main.py ------     占位  
└── utils.py -----     工具函数  

# 数据集  
data/目录包含本研究中使用的光谱数据：  
chemcam — 输入光谱数据  
supercam — 目标光谱数据（真实值）  

# 主要结果  
一维模型的性能显著优于二维转换模型。    
提出的引导注意力模型取得了最佳性能：  
大样本PSNR：79.49  
小样本PSNR：77.49  
性能优于传统的双三次插值（PSNR ~63）和SRCNN（PSNR ~65–75）。  
<img width="1748" height="876" alt="QQ20260311-144751" src="https://github.com/user-attachments/assets/570b0afb-2d09-4ddd-b70c-518321052b21" />

# 文件结构
Main  
├── data ---------    光谱数据集 (Chemcam, Supercam)  
├── report ------     项目报告 (PDF)  
├── src ---------     所有模型的源代码  
│   ├── SRCNN/  
│   ├── DRN/  
│   ├── GAIN/    
│   └── ...  
└── README.md ------   项目说明  
