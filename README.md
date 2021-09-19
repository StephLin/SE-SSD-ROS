# SE-SSD-ROS: Self-Ensembling Single-Stage Object Detection From Point Cloud with ROS Wrapper

A ROS wrapper for [SE-SSD](https://github.com/Vegeta2020/SE-SSD), an accurate and fast single-stage 3D object detection framework on [the KITTI 3D object detection dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

Original README: [here](./SE-SSD.md)

## Installation

### Prerequisites

1. Python 3.6+ (We recommend using virtualenv)
2. PyTorch 1.6
3. ROS Melodic (You don't need to re-compile ROS with Python 3)
4. [jsk_recognition](https://github.com/jsk-ros-pkg/jsk_recognition) `sudo apt install ros-melodic-jsk-recognition-msgs ros-melodic-jsk-rviz-plugins`
5. Boost `sudo apt install libboost-all-dev`
6. cmake **3.13.2+**

### 0. CMake 3.13.2+

Since the default cmake version w.r.t. apt is 3.10.2, you have to prepare a newer version of cmake for your own machine.

In this way, we suggest to build cmake from source, and to set the binary directory as part of `$PATH`.
Taking `~/Documents/third-party` as the storage path, run the following commands:

```bash
cd ~/Documents/third-party
wget https://github.com/Kitware/CMake/releases/download/v3.21.2/cmake-3.21.2.tar.gz
tar xzvf cmake-3.21.2.tar.gz
cd cmake-3.21.2
./bootstrap
make
CMAKE_BIN_PATH=$PWD/bin
```

Note that you don't need to run `make install` since it is probably to pollute your host environment.
Instead, you have to manually add the binary directory, i.e., `$CMAKE_BIN_PATH`, to your `$PATH` variable, once the cmake is needed.

### 1. Install by Script

Run the following commands:

```bash
python3 -m virtualenv venv && source venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install torch==1.6.0 torchvision==0.7.0  # CUDA 10.2
python3 install.py --cmake_executable=$CMAKE_BIN_PATH/cmake
```

Once the installation script complete without error, you are ready to start the detection.

## Usage

We provide a single script to launch a LiDAR detection ROS node via

```bash
python3 ros_main.py
```

You can check available arguments via

```bash
python3 ros_main.py --help
```
