FROM nvcr.io/nvidia/pytorch:20.07-py3

# ROS Melodic
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt update && \
    ln -fs /usr/share/zoneinfo/Asia/Taipei /etc/localtime && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata && \
    apt install -y ros-melodic-ros-base && \
    apt install -y python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential && \
    rosdep init && \
    rosdep update

RUN apt install -y ros-melodic-jsk-recognition-msgs

COPY Caffe2Targets.cmake /opt/conda/lib/python3.6/site-packages/torch/share/cmake/Caffe2/Caffe2Targets.cmake
COPY . /se-ssd
WORKDIR /se-ssd

# SE-SSD dependencies
RUN python3 -m pip install -r requirements.txt && \
    python3 -m pip install typing-extensions==4.1.0 && \
    python3 install.py

COPY bashrc /root/.bashrc

CMD ["bash", "-c", "NUMBAPRO_NVVM=/usr/local/cuda/nvvm/lib64/libnvvm.so NUMBAPRO_LIBDEVICE=/usr/local/cuda/nvvm/libdevice/ python3 ros_main.py --subscribed_topic /lio_segmot/keyframe/cloud_info --verbose --mode lio_segmot"]
