# OD
This is the project of paper "Ocean's Duality: Physics-Driven Dual-Branch Framework for Underwater Vision Enhancement".

# 1. Abstract 
Since underwater videos involve scarce paired data and mismatched physical priors, underwater visual enhancement faces significant challenges in hybrid static-dynamic scenes. Existing methods typically design single-frame enhancement paradigms to address cross-frame optical variations, resulting in temporally inconsistent video and redundant computations. Although spatial-temporal feature fusion is a common inter-frame correlation strategy to mitigate these issues, mainstream fusion strategies struggle to handle cross-frame variations caused by non-uniform channel attenuation and regional degradation. Therefore, we propose a physics-semantic collaborative dual-branch framework. This framework employs image-branch priors as unified enhancement prompts, guiding the video branch to construct more robust semantic understanding and physical estimation across multiple frames. Specifically, we incorporate a selective channel fusion mechanism into Mamba to efficiently aggregate complementary spatial-temporal cues. Moreover, we devise a novel physical parameter estimation paradigm for adaptively interpreting dynamic degradation, thereby guiding color correction and degradation removal in underwater video. Extensive experiments on six datasets demonstrate that the proposed method achieves superior enhancement performance compared to the state-of-the-art methods.

## Environment
```
1. Python 3.10.13
2. PyTorch 2.1.1
3. Torchvision 0.16.1
4. OpenCV-Python 4.9.0.80
5. NumPy 1.26.3
6. Mamba-ssm 1.2.0.post1
```

## Checkpoints
in the 'checkpoints' foloder.

## Data Format

```text
Data format
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
You can use utils/VideosToFrames.py to generate Video Frames.

## Test
1. Changed configs/Video/test_config.yml or configs/Image/test_config.yml
2. Run test_video.py or test_image.py

## Train
1. Changed configs/Video/train_config.yml or configs/Image/train_config.yml
2. Run train_video.py or train_image.py

## Contact
If you have any questions, please contact: Zhixiong Huang: hzxcyanwind@mail.dlut.edu.cn
