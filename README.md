# visual-distortion-in-attack
This repository is the official implementation of Towards Visual Distortion in Black-box Attacks.

![](/framework/model.png)

## Performance

We ran the code with 10 random starts when $\lambda=10, N=1$ and the maximum queries=10,000. The results are reported below: 

| Black-box Network | Success Rate | 1-SSIM | LPIPS | Average Queries |
| :-:| :-: | :-: | :-:| :-: |
|    InceptionV3    |    98.7\%    | 0.065  | 0.084 |       731       |
|     ResNet50      |    100\%     | 0.066  | 0.071 |       401       |
|      VGG16bn      |    100\%     | 0.062  | 0.069 |       251       |

Increasing $\lambda$ can achive lower 1-SSIM and LPIPS at the cost of more number of queries and lower success rate.

## Requirements

**Dependency**

The code is based on Python 3.6 with Tensorflow 1.12.0 and PyTorch 1.0.1. To install requirements,

```setup
pip install -r requirements.txt
```
**Pretrained Model**

The pretrained weights of InceptionV3, ResNet50 and VGG16bn will be downloaded automatically when running the corresponding network.

**Data**

To perform image attack, download images from [ImageNet](http://www.image-net.org/archive/stanford/fall11_whole.tar). For out-of-object attack, please donwload the [object bounding boxes](https://academictorrents.com/download/dfa9ab2528ce76b907047aa8cf8fc792852facb9.torrent). The object bounding boxes are necessary only if you want to perform the out-of-object attack. 

## Evaluation

Before evaluation, please change `IMAGE_DIR` in eval_attack.py to your own data directory. `IMAGE_DIR` indicates the path to the images from ImageNet. If you also want to perform the out-of-object attack, please change `BBOX_DIR` in eval_attack.py to your bounding box directory. For evaluation, run

```eval
python eval_attack.py
```

To test on a single image from ImageNet, run
```test
python demo_attack.py
```
