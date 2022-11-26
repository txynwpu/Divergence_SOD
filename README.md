# A General Divergence Modeling Strategy for Salient Object Detection
## ACCV 2022

### [Project Page](https://npucvr.github.io/Divergence_SOD/) | [Paper](https://openaccess.thecvf.com/content/ACCV2022/papers/Tian_A_General_Divergence_Modeling_Strategy_for_Salient_Object_Detection_ACCV_2022_paper.pdf) | [Video](https://youtu.be/r-9I01TsZNU) | [LAB Page](http://npu-cvr.cn/)

## Abstract
Salient object detection is subjective in nature, which implies that multiple estimations should be related to the same input image. Most existing salient object detection models are deterministic following a point to point estimation learning pipeline, making them incapable of estimating the predictive distribution. Although latent variable model based stochastic prediction networks exist to model the prediction variants, the latent space based on the single clean saliency annotation is less reliable in exploring the subjective nature of saliency, leading to less effective saliency “divergence modeling”. Given multiple saliency annotations, we introduce a general divergence modeling strategy via random sampling, and apply our strategy to an ensemble based framework and three latent variable model based solutions to explore the “subjective nature” of saliency. Experimental results prove the superior performance of our general divergence modeling strategy.

## Motivation
![image]()
In the process of saliency map labeling, the labeling is often done by multiple annotators, and then the regions that receive consensus are labeled as saliency after majority voting. However, Most of the existing SOD models intend to achieve the point estimation from input image to the corresponding ground truth saliency map, neglecting the less consistent saliency regions discarded while generating the ground truth maps via majority voting. In this paper, we study “divergence modeling” for SOD, representing the “subjective nature” of saliency.

## Environment
Pytorch 1.10.0
Torchvision 0.11.1
Cuda 11.4

## Dataset
We use the [COME dataset](https://github.com/JingZhang617/cascaded_rgbd_sod) for training as it’s the only large saliency training dataset containing multiple annotations for each image, where an image has 5 gts from different annotators and a majority voting gt. 

Train dataset path structure
    train_data_root
    ├── image
    ├── gt
    └── Multi_Annotators5
        ├── gt1         
        ├── gt2 
        ├── gt3 
        ├── gt4 
        └── gt5 
        
Test dataset path structure
    test_data_root
        ├── COME-E
        ├── COME-H
        ├── DUTS_Test
        ├── ECSSD
        ├── DUT
        └── HKU-IS
        
You need to modify the data path in train.py and test.py to your own data path.

## Training and testing
We present an ensemble based framework and three latent variable model based solutions(VAE, GAN and ABP) to validate our divergence modeling strategy. 
![image]()

To train each method based model, run
```bash
$ python train.py
```
To test each method based model, run
```bash
$ python test.py
```
After testing, for each test image, we can get a majority voting prediction, 5 diverse predictions and an uncertainty map.
![image]()

## Citation
If you are interested in our work, welcome to discuss with us and cite our paper. 

@inproceedings{tian2022general,
  title={A General Divergence Modeling Strategy for Salient Object Detection},
  author={Tian, Xinyu and Zhang, Jing and Dai, Yuchao},
  booktitle={Proceedings of the Asian Conference on Computer Vision},
  pages={2406--2424},
  year={2022}
}




        

