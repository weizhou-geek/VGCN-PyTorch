# VGCN-PyTorch

Thanks for your attention. In this repo, we provide the codes for the paper [[Blind Omnidirectional Image Quality Assessment with Viewport Oriented Graph Convolutional Networks]](https://ieeexplore.ieee.org/document/9163077).

## Prerequisites
+ scipy==1.2.1
+ opencv_python==4.1.0.25
+ numpy==1.16.4
+ torchvision==0.3.0
+ torch==1.1.0
+ Pillow==6.2.0

## Install
To Install all the dependencies in prerequisites

## Prepare Data
+ Obtain [cviqd_local_epoch.pth](https://drive.google.com/file/d/1ROT4InmAEKUisfNbMHwWpWb0nvlDhoSe/view?usp=sharing), [cviqd_global_epoch.pth](https://drive.google.com/file/d/1ggxGi2uvmL3n0BtYLC-HCrWbhna2TkFQ/view?usp=sharing), and [cviqd_model.pth](https://drive.google.com/file/d/19WJHBkogveax0b3IgpWeRco5xXgKQvFl/view?usp=sharing)
+ Download [database](https://drive.google.com/drive/folders/1LqQFIms_46s7uybos83-5EgMAH2r6OCy?usp=sharing)

## Training
```
python main.py --root1 cviqd_local_epoch.pth --root2 cviqd_global_epoch.pth --save test
```

## Testing
```
python main.py --resume cviqd_model.pth --skip_training
```

## Citation
You may cite it in your paper. Thanks a lot.

```
@article{xu2020blind,
  title={Blind Omnidirectional Image Quality Assessment with Viewport Oriented Graph Convolutional Networks},
  author={Xu, Jiahua and Zhou, Wei and Chen, Zhibo},
  journal={arXiv preprint arXiv:2002.09140},
  year={2020}
}
```


