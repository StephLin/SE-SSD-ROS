# SE-SSD-ROS: Self-Ensembling Single-Stage Object Detection From Point Cloud with ROS Wrapper

A ROS wrapper for [SE-SSD](https://github.com/Vegeta2020/SE-SSD), an accurate and fast single-stage 3D object detection framework on [the KITTI 3D object detection dataset](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

Original README: [here](./SE-SSD.md)

## Installation (Native)

### Prerequisites

1. Python 3.6+ (We recommend using virtualenv)
2. PyTorch 1.6
3. ROS Melodic (You don't need to re-compile ROS with Python 3)
4. [jsk_recognition](https://github.com/jsk-ros-pkg/jsk_recognition) `sudo apt install ros-melodic-jsk-recognition-msgs ros-melodic-jsk-rviz-plugins`
5. Boost `sudo apt install libboost-all-dev`
6. cmake **3.13.2+**
7. (Runtime) [LIO-SEGMOT](https://github.com/StephLin/LIO-SEGMOT)

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

## Installation (Docker)

### Prerequisites

1. ROS Melodic (You don't need to re-compile ROS with Python 3)
2. [Docker](https://www.docker.com/) with [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)
3. (Runtime) [LIO-SEGMOT](https://github.com/StephLin/LIO-SEGMOT)

### 1. Build with docker-compose

Run the following command to build the SE-SSD image:

```bash
docker compose build
```

### 2. Configure [docker-compose.yml](./docker-compose.yml)

You should replace the following path with your local machine's one. For example, if your catkin_ws's location is `/home/alice/catkin_ws`, then you should modify the line as follows:

```diff
-     - /path/to/catkin_ws/devel/lib/python2.7/dist-packages/lio_segmot:/opt/ros/melodic/lib/python2.7/dist-packages/lio_segmot
+     - /home/alice/catkin_ws/devel/lib/python2.7/dist-packages/lio_segmot:/opt/ros/melodic/lib/python2.7/dist-packages/lio_segmot
```

## Usage

We provide a single script to launch a ROS node or a ROS service:

- Run SE-SSD as a ROS node:

  ```bash
  python3 ros_main.py
  ```

- Run SE-SSD as a ROS service (e.g., for **LIO-SEGMOT**):

  ```bash
  python3 ros_main.py --mode lio_segmot
  ```

You can check available arguments via

```bash
python3 ros_main.py --help
```

If you deploy SE-SSD-ROS with docker, you can run the following the command to launch it:

```bash
docker compose up
```
