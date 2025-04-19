# PECH
# implementation for **PECH: Prior-Enhanced CLIP Hashing for Out-of-Distribution Retrieval

### Prerequisites:
- python == 3.12.2
- pytorch == 2.2.1
- torchvision == 0.17.1
- numpy, scipy, sklearn, PIL, argparse, tqdm, loguru

### Dataset:

- Please manually download the datasets [Office](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view), [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view), [VisDA-C](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification) from the official websites, and modify the path of images in each '.txt' under the folder './object/data/'. The source model of office-home and visda can be downloaded in this [Url](https://drive.google.com/drive/folders/1eiJtky4seNApOSYJiGrDywfJbCBp_3sb)


### Training:
		python run.py
   
