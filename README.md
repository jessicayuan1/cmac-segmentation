# **CNN vs. ViT-Based U-Nets for Diabetic Retinopathy Lesion Segmentation @ WAT.ai**
## Semantic Segmentation of Retinal Lesions caused by Diabetic Retinopathy

### Overview
This repository contains work in progress on the **semantic segmentation of microaneurysms, hemorrhages, soft exudates, hard exudates, and the optic disc** (lesions resulting from Diabetic Retinopathy) from fundus images. In addition to building the full segmentation pipeline, the project also conducts a comparison between **CMAC-UNet** (a convolution-based architecture) and **Swin-UNet** (a transformer-based architecture). This project also conducts comparisons between loss functions: **Binary Cross-Entropy, Focal Tversky, and Dice** to access how different optimization techniques influence small-lesion detection, class imbalance, and overall segmentation quality.

---

### Models
The segmentation architectures used in this project are based directly on the original research papers that introduced them. These papers outline the core design principles and architectural choices behind each model.
* **Swin-UNet** Paper: [Swin-UNet: UNet-like Pure Transformer for Medical Image Segmentation](https://arxiv.org/pdf/2105.05537)
* **CMAC-UNet** Paper: [CMAC-Net: Cascade Multi-Scale Attention Convolution Network for diabetic retinopathy lesion segmentation](https://www.sciencedirect.com/science/article/pii/S1746809425009954?via%3Dihub)

Additionally, full dynamic implementations of both models can be found in this repository.

### Datasets
Two datasets are chosen for the project. Both datasets contain fundus images and segmentation masks. The IDRiD dataset contains masks for microaneurysms, hemorrhages, soft exudates, hard exudates, and the optic disc. The DDR dataset contains the same but is missing optic disk masks. An external segmentation model was used to create the optic disk masks for the DDR dataset.
* **IDRiD**: [Indian Diabetic Retinopathy Dataset from Kaggle](https://www.kaggle.com/dataset/saaryapatel98/indian-diabetic-retinopathy-image-dataset)
* **DDR**: [Diabetic Retinopathy Lesion Segmentation and Lesion Detection Dataset from GitHub](https://github.com/nkicsl/DDR-dataset/tree/master)

---

### References
* [A NOVEL FOCAL TVERSKY LOSS FUNCTION WITH IMPROVED ATTENTION U-NET FOR LESION SEGMENTATION](https://arxiv.org/pdf/1810.07842)  