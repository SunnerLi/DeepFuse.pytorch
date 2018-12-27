# DeepFuse.pytorch

### The re-implementation of ICCV 2017 DeepFuse paper idea

[![Packagist](https://img.shields.io/badge/Pytorch-0.4.1-red.svg)]()
[![Packagist](https://img.shields.io/badge/OpenCV-3.4.3-green.svg)]()
[![Packagist](https://img.shields.io/badge/Torchvision_sunner-18.9.15-yellow.svg)](https://github.com/SunnerLi/Torchvision_sunner)

![](https://github.com/SunnerLi/DeepFuse.pytorch/blob/master/img/structure.png)

Abstraction
---
Multi-exposure fusion is a critical issue in computer vision. Additionally, this technique can be adopt in smart phone to demonstrate the image with high lighting quality. However, the original author didn't release the official implementation. In this repository, we try to re-produce the idea of DeepFuse [1], and fuse the under-exposure image and over-exposure image with appropriate manner.     

Result
---
![](https://github.com/SunnerLi/DeepFuse.pytorch/blob/master/img/fuse_result1.png)

The above image shows the training result. The most left sub-figure is the under-exposure image. The second sub-figure is the over-exposure image. The third one is the rendered result, and the most right figure is the ground truth which is compute by MEF-SSIM loss concept. As you can see, the rough information of dark region and light region can be both remained. The following image is another example.    

![](https://github.com/SunnerLi/DeepFuse.pytorch/blob/master/img/fuse_result2.png)

Idea
---
**You should notice that this is not the official implementation.** There are several different between this repository and the paper:
1. Since the dataset that author used cannot be obtained, we use [HDR-Eye dataset](https://mmspg.epfl.ch/hdr-eye?fbclid=IwAR1YLuQvcpu6yM2MsV60LcbURFopzIqqUBKlBUjvbNCQBXxB3iMzgm0Uy8o) [2] which can also deal with multiple exposure fusion problem.
2. Rather use _64*64_ patch size, we set the patch size as _256*256_.    
3. We only train for 20 epochs. (30000 iteration for each epoch)    
4. The calculation of y^hat is different. The detail can be found in [here](https://github.com/SunnerLi/DeepFuse.pytorch/blob/master/loss.py#L100). 

Usage
---
The detail of parameters can be found [here](https://github.com/SunnerLi/DeepFuse.pytorch/blob/master/opts.py). You can just simply use the command to train the DeepFuse:

```
python3 train.py --folder ./SunnerDataset/HDREyeDataset/images/Bracketed_images --batch_size 8 --epoch 15000 
```
Or you can download the pre-trained model [here](https://drive.google.com/file/d/1NYlYeDCyu_KxAjsl9m9X9IXq588rCIq7/view?usp=sharing). Furthermore, inference with two image:

```
python3 inference.py --image1 <UNDER_EXPOSURE_IMG_PATH> --image2 <OVER_EXPOSURE_IMG_PATH> --model train_result/model/latest.pth --res result.png
```

Notice
---
After we check for several machine, we found that the program might get stuck at `cv2.cvtColor` function. We infer the reason is that the OpenCV cannot perfectly embed in the multiprocess mechanism which is provided by Pytorch. As the result, we assign `num_worker` as zero [here](https://github.com/SunnerLi/DeepFuse.pytorch/blob/master/train.py#L38) to avoid the issue. If your machine doesn't encounter this issue, you can add the number to accelerate the loading process.      

Reference
---
[1]  K. R. Prabhakar, V. S. Srikar, and R. V. Babu.  Deepfuse:  A deep unsupervised approach for exposure fusion with extreme exposure image pairs. In 2017 IEEE International Conference on Computer Vision (ICCV). IEEE, pages 4724–4732, 2017.    
[2]  H. Nemoto, P. Korshunov, P. Hanhart, and T. Ebrahimi. Visual attention in ldr and hdr images. In 9th International Workshop on Video Processing and Quality Metrics for Consumer Electronics (VPQM), number EPFL-CONF-203873, 2015.    