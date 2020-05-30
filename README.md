# visual-distortion-in-attack
This repository is the official implementation of Towards Visual Distortion in Black-box Attacks.
<img src="https://github.com/Alina-1997/visual-distortion-in-attack/edit/master/model.png" width="633" >

## Requirements
To install requirements

```setup
pip install -r requirements.txt
```

Dataset

[ImageNet Images](http://www.image-net.org/archive/stanford/fall11_whole.tar)

[ImageNet Bounding Boxes](http://image-net.org/Annotation/Annotation.tar.gz)

## Evaluation
Before evaluation, please change the `IMAGE_DIR` in eval_attack.py to your own data directory.
```eval
python eval_attack.py
```
