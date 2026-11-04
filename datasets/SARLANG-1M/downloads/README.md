## SARLANG-1M
[![TGRS Paper](https://img.shields.io/badge/TGRS-paper-00629B.svg)](https://ieeexplore.ieee.org/document/11341914) [![ArXiv Paper](https://img.shields.io/badge/arXiv-paper-b31b1b.svg)](https://arxiv.org/abs/2504.03254) [![Github Project](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Jimmyxichen/SARLANG-1M/tree/main)

<center>
    <img src="Overview.png" alt="SARLANG-1M is a large-scale benchmark tailored for multimodal SAR image understanding." width="900" height="500">
</center>
SARLANG-1M is a large-scale benchmark tailored for multimodal SAR image understanding, with a primary focus on integrating SAR with textual modality. SARLANG-1M comprises more than 1 million high-quality SAR image-text pairs collected from over 59 cities worldwide. It features hierarchical resolutions (ranging from 0.1 to 25 meters), fine-grained semantic descriptions (including both concise and detailed captions), diverse remote sensing categories (1,696 object types and 16 land cover classes), and multi-task question-answering pairs spanning seven applications and 1,012 question types.

SARLANG-1M dataset supports seven SAR applications:

| Application | Application Description | Text Numuber |
|:---:|:---:|:---:|
| Image Description | Describe the SAR image | 45,650 |
| Object Identification | Determine the presence of specific objects | 484,620 |
| Object Classification | Identify the predominant category within the SAR image | 132,525 |
| Instance Counting | Quantify instances within the SAR image | 117,382 |
| Region Referring | Determine the category present in the specific location | 221,450 |
| Object Positioning | Determines the approximate location of a category | 106,171 |
| Others | Predict the object shape, direction, reasoning etc | 18,479 |

The Statistics of Text Annotations in SARLANG-1M dataset: 
<p align="center">
  <img src="fig36.png" alt="Statistics" width="97%">
</p>
The image shows the distribution of seven applications provided in the SARLANG-1M benchmark (a), the numbers of each question type in the 'others' application (b), and the distribution of the 30 most frequent object categories (c).


## Text Annotations in Our SARLANG-1M Dataset 
The entire text annotations in our SARLANG-1M dataset are provided in the Text.zip file.

## Processed SAR Images in Our SARLANG-1M Dataset 

The SAR images provided in our SARLANG-1M dataset come from four sub-datasets: 

- SpaceNet6, DFC2023, OpenEarthMap-SAR: The original SAR images(tif format) and the preprocessed SAR images(png format) are saved in the SARimages_original.zip file and the SARimages_preprocessed.zip file, respectively. Notably, SAR image preprocessing is an optional strategy to improve the performance of VLMs by significantly enhancing image clarity and effectively highlighting key objects within the SAR images. You can choose any version according to your needs.
- SARDet-100K: The SAR images in this dataset has been preprocessed and denoised. Original SAR images are directly collected in our SARLANG-1M dataset without any preprocessing operations. The images can be directly download from the official link of [SARDet_100K](https://github.com/zcablii/SARDet_100K).

## 🤝Acknowledgments
The authors would also like to give special thanks to [SARDet_100K](https://github.com/zcablii/SARDet_100K), [SpaceNet6](https://spacenet.ai/sn6-challenge/), [DFC2023](https://www.grss-ieee.org/community/technical-committees/2023-ieee-grss-data-fusion-contest/) and [OpenEarthMap-SAR](https://zenodo.org/records/14622048) for providing the valuable SAR Images.

## Contact
2364356729@qq.com, YIMIN WEI, The University of Tokyo