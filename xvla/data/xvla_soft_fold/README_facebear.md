---
license: mit
task_categories:
- robotics
language:
- en
size_categories:
- 1K<n<10K
---
# Cloth-Folding Dataset for X-VLA Paper

This dataset contains 1,500 episodes of cloth folding, collected using Agilex's robotic arm. It was used in the **X-VLA** paper for cloth-folding tasks, showcasing a near-perfect success rate in folding accuracy.

# Dataset Overview

- Total Episodes: ～1,500
- Task: Automated cloth folding
- Robot: Agilex Aloha
- Performance: Near 100% success rate in completing the folding task

# Hardware setup

We observed that the camera setup of the official Agilex Aloha platform is positioned relatively low, which prevents it from capturing the full cloth-folding process, where many frames fail to include the robot arms. To address this issue, we modified the camera setup accordingly.
You can find the `.stl`, `.step`, `.sldprt` files of our new camera mount, which can be used for 3D printing. The installation instruction can be found in the `camera_mount_install.md`.

# Usage
You can find `.hdf5` and `.mp4` files in each directory. The `.mp4` files are just used for visulization and are not used for training. The `.hdf5` files contains all necessary keys and data, including:

## HDF5 file hierarchy
```
├── action # nx14 absolute bimanual joints, not used in our paper
├── base_action # nx2 chassis actions, not used in our paper
├── language_instruction # 🌟"fold the cloth"
├── observations
│   ├── eef # nx14 absolute eef pos using euler angles to represent the rotation, not used in our paper
│   │   eef_quaternion # nx16 absolute eef pos using quaternion to represent the rotation, not used in our paper
│   │   eef_6d # 🌟nx20 absolute eef pos using rotate6d to represent the rotation
│   │   eef_left_time # 🌟nx1 the time stamp for left arm eef pos, can be used for resample or interpolation
│   │   eef_right_time # 🌟nx1 the time stamp for right arm eef pos, can be used for resample or interpolation
│   ├── qpos # nx14 absolute bimanual joints, not used in our paper
│   ├── qpos_left_time # nx1 the time stamp for left arm joint pos, can be used for resample or interpolation, not used in our paper
│   ├── qpos_right_time # nx1 the time stamp for right arm joint pos, can be used for resample or interpolation, not used in our paper
│   ├── qvel # nx14 bimanual joint velocity, not used in our paper
│   ├── effort # nx14 bimanual joint effort, not used in our paper
│   ├── images
│   │   ├── cam_high  # 🌟the encoded head cam view, should be decoded using cv2
│   │   ├── cam_left_wrist  # 🌟the encoded left wrist view, should be decoded using cv2
│   │   ├── cam_right_wrist  # 🌟the encoded right wrist view, should be decoded using cv2
├── time_stamp # the time stamp for each sample, not used in our paper
```

How to read the hdf5 file:

```
import h5py
import cv2
import io
from mmengine import fileio


path = "REPLACE TO YOUR HDF5 FILE PATH HERE"

# load the hdf5 file
value = fileio.get(path)
f = io.BytesIO(value)
h = h5py.File(f,'r')

# you can monitor the hdf5 hierarchy by print out its keys
print(h.keys())

# this is one example to read out the data, for example, the 'cam_high' data
head_view_bytes = h['observations/images/cam_high'][()]  # 🌟 NOTE: we compress all images to bytes using cv2.imencode
head_view = cv2.imdecode(head_view_bytes, cv2.IMREAD_COLOR)  # 🌟 NOTE: we should decode it back to RGB images for further usage

#Then you can go free to use our data :)
# ...
# ...
```

## Visualize the data
You can find some dictionary have `.mp4` file for visulization. If you want to visualize all the `.hdf5` file, you can run the following code:

```
from mmengine import fileio
import io
import h5py
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from IPython.display import display, Image as IPImage
from IPython.display import Video
import os
import imageio
import numpy as np

# 🌟 Just replace the path here, then run this script. This script will generate all the .mp4 files for the .hdf5 file
top_path = 'REPLACE TO YOUR XVLA-SOFT-FOLD PATH'
hdf5_files = fileio.list_dir_or_file(top_path, suffix='.hdf5', recursive=True, list_dir=False)

for hdf5_name in hdf5_files:
    path = os.path.join(top_path, hdf5_name)
    # Prepare OpenCV VideoWriter to save as MP4
    video_path = path.replace('.hdf5', '.mp4')
    fps = 30  # Adjust the FPS if needed
    image_list = []
    print(video_path)
    if os.path.exists(video_path):
        print(f"pass {video_path}, it already exists")
        continue
    
    
    value = fileio.get(path)
    f = io.BytesIO(value)
    h = h5py.File(f,'r')

    images = h['/observations/images/cam_high'][()]
    images_left = h['/observations/images/cam_left_wrist'][()]
    images_right = h['/observations/images/cam_right_wrist'][()]
    ep_len = images.shape[0]
    
    for i in tqdm(range(ep_len)):
        img = images[i]
        img_left = images_left[i]
        img_right = images_right[i]
        
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)  # Decode image from bytes
        img_left = cv2.imdecode(img_left, cv2.IMREAD_COLOR)  # Decode image from bytes
        img_right = cv2.imdecode(img_right, cv2.IMREAD_COLOR)  # Decode image from bytes
        
        img = np.concatenate([img, img_left, img_right], axis = 1)
        image_list.append(img)

    # Release the VideoWriter and show output
    imageio.mimsave(video_path, image_list, fps=fps)
```


# Citation
If you use this dataset in your research or for any related work, please cite the X-VLA Paper:

```
@article{zheng2025x,
  title={X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model},
  author={Zheng, Jinliang and Li, Jianxiong and Wang, Zhihao and Liu, Dongxiu and Kang, Xirui and Feng, Yuchun and Zheng, Yinan and Zou, Jiayin and Chen, Yilun and Zeng, Jia and others},
  journal={arXiv preprint arXiv:2510.10274},
  year={2025}
}
```