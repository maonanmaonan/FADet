## JSTARS(2025) A Benchmark Dataset and Novel Methods for Parallax-Based Flying Aircraft Detection in Sentinel-2 Imagery

![image](https://github.com/user-attachments/assets/66ac1ba5-84ff-4348-b772-9078295cc4c4)

## Abstract
Satellite-based aircraft monitoring is an important complement to ground surveillance systems, providing strong support for the safe, efficient, and reliable operation of global aviation. Most existing aircraft detection datasets are derived from still satellite imagery, making it difficult to detect flying aircraft. Although video satellite imagery can provide motion cues,its spatial coverage is limited, making it challenging to capture flying aircraft targets that are sparsely distributed over wide areas. Each Sentinel-2 satellite image covers a width of hundreds of kilometers, providing favorable conditions for monitoring flying aircraft.Beyond this,the physical design of its multispectral instruments induces parallax effects for moving objects in multispectral imagery, enabling a novel approach for the detection of flying aircraft. We construct a flying aircraft detection dataset (S2Aircraft) based on Sentinel-2 satellite multispectral imagery with a spatial resolution of 10m. The dataset is annotated with oriented bounding boxes and includes both RGB and NIR spectral bands. In addition, we design an efficient flying aircraft detection network (FADet), which maps input
images to a high-dimensional nonlinear feature space while main taining low computational complexity. Moreover, for single-class object detection tasks, the model employs a semidecoupled head to achieve efficient detection. Finally, a loss function is specifically designed according to the geometric characteristics of targets in the S2Aircraft dataset,significantly improving the accuracy and stability of oriented object detection.Extensive experiments demonstrate the effectiveness and advancement of our FADet. Specifically, on our S2Aircraft dataset, FADet achieves competitive performance reaching 2.6 giga floating-point operations per second and 96.3% meanaverage precision (mAP) at 50%intersection over union. On two public datasets, HRSC2016 and CORS-ADD, FADet achieves mAP50 of 90.90% and 94.16%, respectively.
## Dataset
S2Aircraft Dataset is available at [quark](https://pan.quark.cn/s/524cbdea440e) and [google](https://drive.google.com/file/d/1KDgVSP_xIeeIN9587j5WWXNtyYL4_7Vw/view?usp=sharing)
## Installation and Usage
Please refer to the [Ultralytics](https://github.com/ultralytics/ultralytics)
## Results
| Method | Dataset | mAP50 (%) | Params (M) | GFlops | FPS (s) | Download |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| FAdet | S2Aircraft | 96.3 | 98.08 | 98.10 | 98.09 | [models](https://pan.quark.cn/s/74479bafb1e4) |
## Contact
If any questions, kindly contact with Nan Mao via e-mail: 2023124038@chd.edu.cn 
## Citation
If you find this repo useful, please cite our paper.
```bibtex
@ARTICLE{11180886,
  author={Song, Beibei and Mao, Nan and Li, Jingyuan and Du, Wenwang and Wang, Zhe and Shao, Yingzhao and Li, Xiaobo and Bao, Qiudie and Wang, Xiaohan and Sun, Wenfang},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={A Benchmark Dataset and Novel Methods for Parallax-Based Flying Aircraft Detection in Sentinel-2 Imagery}, 
  year={2025},
  volume={18},
  number={},
  pages={25221-25234},
  keywords={Aircraft;Satellites;Aircraft manufacture;Remote sensing;Surveillance;Feature extraction;Radar tracking;Detectors;Spatial resolution;Satellite images;Flying aircraft detection;oriented object detection;parallax effect;Sentinel-2 satellite},
  doi={10.1109/JSTARS.2025.3615068}}
```


