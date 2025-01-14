# Let’s Make Pottery: GANs for Voxel Completion in Ancient Pottery Dataset

This repository implemented GAN for pottery completion.

![](https://fancy-icebreaker-99b.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2F69ca2e73-08a0-44d6-970a-605a1725db83%2Ff55c52d4-36be-44a5-a9ae-894ed0f7be9b%2FUntitled.png?table=block&id=c1b5527c-8c2e-48f2-a833-ce17bca2ad72&spaceId=69ca2e73-08a0-44d6-970a-605a1725db83&width=1420&userId=&cache=v2)

## 1. Installation

The code has been tested on Ubuntu 20.04 with 1 Nvidia Tesla V100S-PCIE-32GB GPU.

1. You can download the dataset from the link:
https://disk.pku.edu.cn:443/link/6B902375BB5D488F5BB8B0FF51512F12

2. Run it locally using python 3.11 and installing the dependencies:
   ```shell
   pip install - r requirements.txt
   conda env create -f environment.yml
   ```

3. Activate the new environment:
   ```shell
   conda activate vasija
   ```
In this project, we use pyvox to open and write the vox files.
pyvox: https://github.com/gromgull/py-vox-io

## 2. Visualize Dataset

1. After downloading the data, you can visualize it by running the following command.

   ```shell
   python utils/visualize.py
   ```

   Then you need to type in the path to the data file you want to visualize.

## 3. Training

1. To train the model, run the following command.

   ```shell
   python training_GAN32.py
   python training_GAN64.py
   ```

   The first command will train the model on 32x32x32 voxel data, and the second command will train the model on 64x64x64 voxel data. The 32 resolution will take about ..., and the 64 resolution will take about ...

## 4. Testing

1. To test the model, run the following command.
   ```shell
   python test_visualize.py --dim_size 32 --model_path G32_2_2025-01-13-01-22-37.pth --selecte_class 3
   python test_visualize.py --dim_size 64 --model_path G64_16_2025-01-13-17-33-07.pth --selecte_class 8
   ```
   The command will random select a sample file from test dataset and visualize the generated result and the ground truth. You can choose the dim_size, model_path and the selected_class(selected_class=0 means choose 1 example from all 11 classes)
