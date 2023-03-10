[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://raw.githubusercontent.com/nvlabs/SPADE/master/LICENSE.md)
![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# 3D-Pretraining
Self-supervised Point Cloud Representation Learning via Separating Mixed Shapes

[pdf](https://www.zdzheng.xyz/files/TMM_3D_Pre_Training.pdf)

## Prerequisites

- Python 3.6
- GPU memory >= 8G 
- NumPy
- PyTorch 1.0+


## ShapeNetPart
Download dataset from [ShapeNetPart](https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip)
```bash
ln -s <dataset_path> data
python train.py --opt ./mixup.yml
```

> This is the part of the code. The complete code will be released after integration into the new code framework, such as more datasets, more backbones.

## ModelNet


## Citation
Please cite this paper if it helps your research:
```bibtex
@article{sun2022self,
  title={Self-supervised Point Cloud Representation Learning via Separating Mixed Shapes},
  author={Sun, Chao and Zheng, Zhedong and Wang, Xiaohan and Xu, Mingliang and Yang, Yi},
  journal={IEEE Transactions on Multimedia},
  year={2022},
  publisher={IEEE}
}
```
