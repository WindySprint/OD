# OD
This is the project of paper "Ocean's Duality: Physics-Driven Dual-Branch Framework for Underwater Vision Enhancement".

---

# 1. Abstract 
Since underwater videos involve scarce paired data and mismatched physical priors, underwater visual enhancement faces significant challenges in hybrid static-dynamic scenes. Existing methods typically design single-frame enhancement paradigms to address cross-frame optical variations, resulting in temporally inconsistent video and redundant computations. Although spatial-temporal feature fusion is a common inter-frame correlation strategy to mitigate these issues, mainstream fusion strategies struggle to handle cross-frame variations caused by non-uniform channel attenuation and regional degradation. Therefore, we propose a physics-semantic collaborative dual-branch framework. This framework employs image-branch priors as unified enhancement prompts, guiding the video branch to construct more robust semantic understanding and physical estimation across multiple frames. Specifically, we incorporate a selective channel fusion mechanism into Mamba to efficiently aggregate complementary spatial-temporal cues. Moreover, we devise a novel physical parameter estimation paradigm for adaptively interpreting dynamic degradation, thereby guiding color correction and degradation removal in underwater video. Extensive experiments on six datasets demonstrate that the proposed method achieves superior enhancement performance compared to the state-of-the-art methods.

---

# 2. Environment

```bash
Python 3.10.13
PyTorch 2.1.1
Torchvision 0.16.1
OpenCV-Python 4.9.0.80
NumPy 1.26.3
Mamba-ssm 1.2.0.post1
```

---

# 3. Checkpoints

Pretrained models are provided in the `checkpoints` folder.

---

# 4. Data Format

```text
Dataset
├── Images
│   ├── Image_1.jpg
│   ├── Image_2.jpg
│   └── ...
│
└── Video_Frames
    ├── Video_1
    │   ├── Frame_1.jpg
    │   ├── Frame_2.jpg
    │   └── ...
    │
    ├── Video_2
    │   ├── Frame_1.jpg
    │   ├── Frame_2.jpg
    │   └── ...
    │
    └── Video_M
        ├── Frame_1.jpg
        ├── Frame_2.jpg
        └── ...
```

You can use the following script to extract frames from videos:

```bash
utils/VideosToFrames.py
```

---

# 5. Testing

1. Modify the configuration file:

```
configs/Video/test_config.yml
```

or

```
configs/Image/test_config.yml
```

2. Run the testing script:

```bash
python test_video.py
```

or

```bash
python test_image.py
```

---

# 6. Training

1. Modify the training configuration file:

```
configs/Video/train_config.yml
```

or

```
configs/Image/train_config.yml
```

2. Run the training script:

```bash
python train_video.py
```

or

```bash
python train_image.py
```

---

# 7. Contact

If you have any questions, please contact:

**Zhixiong Huang**  
Dalian University of Technology  
📧 hzxcyanwind@mail.dlut.edu.cn
