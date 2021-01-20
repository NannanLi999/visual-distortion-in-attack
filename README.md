# visual-distortion-in-attack
This repository is the official implementation of Towards Visual Distortion in Black-box Attacks.

![](/framework/model.jpg)

## Performance

We ran the code with 10 random starts when $\lambda=10, N=1$ and the maximum queries=10,000. The results are reported below: 

Ours
| Black-box Network | Success Rate | 1-SSIM | LPIPS | CIEDE2000 | Average Queries |
| :-:| :-: | :-: | :-:| :-:| :-: |
|    InceptionV3    |    98.7\%    | 0.075  | 0.094 |   0.692   |       731       |
|     ResNet50      |    100\%     | 0.076  | 0.081 |   0.741   |       401       |
|      VGG16bn      |    100\%     | 0.072  | 0.079 |   0.699   |       251       |

Ouss(${\lambda}_{dynamic}$)
| Black-box Network | Success Rate | 1-SSIM | LPIPS | CIEDE2000 | Average Queries |
| :-:| :-: | :-: | :-:| :-:| :-: |
|    InceptionV3    |    100\%    | 0.016  | 0.023 |   0.215   |       7311       |
|     ResNet50      |    100\%    | 0.009  | 0.009 |   0.204   |       7678       |
|      VGG16bn      |    100\%    | 0.006  | 0.005 |   0.055   |       7602       |

When using a fixed value of $lambda$, increasing it can acheive lower 1-SSIM, LPIPS and CIEDE2000 at the cost of more number of queries and lower success rate.

## Requirements

Clone this repo:

```
git clone https://github.com/Alina-1997/visual-distortion-in-attack
```

**Dependency**

The code is based on Python 3.6 with Tensorflow 1.12.0 and PyTorch 1.0.1. To install requirements,

```setup
pip install -r requirements.txt
```

To evaluate on LPIPS, clone the official repo

```
git clone https://github.com/richzhang/PerceptualSimilarity
```
and put it in the current directory.

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
