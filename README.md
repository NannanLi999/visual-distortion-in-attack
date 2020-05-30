# visual-distortion-in-attack
This repository is the official implementation of Towards Visual Distortion in Black-box Attacks.
<img src="https://github.com/Alina-1997/visual-distortion-in-attack/edit/master/model.png" width="633" >

## Requirements
To install requirements

```setup
pip install -r requirements.txt
```

Dataset

To perform image attack, images from [ImageNet](http://www.image-net.org/archive/stanford/fall11_whole.tar) is required. For out-of-object attack, please donwload the [object bounding boxes](https://academictorrents.com/download/dfa9ab2528ce76b907047aa8cf8fc792852facb9.torrent)


## Evaluation
Before evaluation, please change the `IMAGE_DIR` in eval_attack.py to your own data directory.
```eval
python eval_attack.py
```
