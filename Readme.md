# 3D-Pretraining
Self-supervised Point Cloud Representation Learning via Separating Mixed Shapes

Download dataset from [ShapeNetPart](https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip)
```bash
ln -s <dataset_path> data
python train.py --opt ./mixup.yml
```

> This is the part of the code. The complete code will be released after integration into the new code framework, such as more datasets, more backbones.