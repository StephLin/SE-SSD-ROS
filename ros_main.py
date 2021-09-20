#!/usr/bin/env python3
import argparse
import contextlib as ctx
import os

from rich import box as rich_box
from rich import traceback
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
traceback.install()

with console.status('[bold green]Importing core packages ...'):
    import numpy as np
    import torch
    from scipy.spatial.transform.rotation import Rotation as R

    from det3d import torchie
    with ctx.redirect_stdout(None):
        from det3d.models import build_detector
    import ros_numpy
    import rospy
    from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
    from sensor_msgs.msg import PointCloud2

    from det3d.datasets.pipelines import AssignTarget, Preprocess, Reformat, Voxelization
    from det3d.torchie.apis import init_dist
    from det3d.torchie.parallel import MegDataParallel, collate_kitti
    from det3d.torchie.trainer import load_checkpoint
    from det3d.torchie.trainer.trainer import example_to_device

    console.log('Imported core packages successfully.')


class Callback:
    DETECTION_PUBLISH_TOPIC = '/se_ssd/detected_objects'
    FRONT_DETECTION_PUBLISH_TOPIC = '/se_ssd/front_detected_objects'
    BACK_DETECTION_PUBLISH_TOPIC = '/se_ssd/back_detected_objects'
    CLOUD_PUBLISH_TOPIC = '/se_ssd/cloud'

    detection_pub = rospy.Publisher(DETECTION_PUBLISH_TOPIC,
                                    BoundingBoxArray,
                                    queue_size=10)
    front_detection_pub = rospy.Publisher(FRONT_DETECTION_PUBLISH_TOPIC,
                                          BoundingBoxArray,
                                          queue_size=10)
    back_detection_pub = rospy.Publisher(BACK_DETECTION_PUBLISH_TOPIC,
                                         BoundingBoxArray,
                                         queue_size=10)
    cloud_pub = rospy.Publisher(CLOUD_PUBLISH_TOPIC,
                                PointCloud2,
                                queue_size=10)

    def __init__(self,
                 model,
                 cfg,
                 device='cuda',
                 distributed=False,
                 range_detection=False):
        self.model = model
        if distributed:
            self.model = self.model.module
        self.model.eval()
        self.cfg = cfg
        self.device = device
        self.range_detection = range_detection

        # data processing
        self.preprocess = Preprocess(cfg=cfg.val_preprocessor)
        self.voxelize = Voxelization(cfg=cfg.voxel_generator)
        self.assign_target = AssignTarget(cfg=cfg.assigner)
        self.reformat = Reformat()

        # step
        self.step = 1

    def __call__(self, cloud_msg: PointCloud2):
        data = self._convert_cloud_to_tensor(cloud_msg)

        self.step = (self.step + 1) % 2
        if self.step != 0:
            return

        with torch.no_grad():
            outputs = self.model(data, return_loss=False, rescale=True)
            boxes = outputs[0]['box3d_lidar'].detach().cpu().numpy()
            scores = outputs[0]['scores'].detach().cpu().numpy()
            labels = outputs[0]['label_preds'].detach().cpu().numpy()

            if self.range_detection:
                back_boxes = outputs[1]['box3d_lidar'].detach().cpu().numpy()
                back_scores = outputs[1]['scores'].detach().cpu().numpy()
                back_labels = outputs[1]['label_preds'].detach().cpu().numpy()

        detection_msg = BoundingBoxArray()
        detection_msg.header = cloud_msg.header

        # front detections
        front_detection_msg = BoundingBoxArray()
        front_detection_msg.header = cloud_msg.header

        for box, score, label in zip(boxes, scores, labels):
            detection = self._convert_model_output_to_jsk_bounding_box(
                box, score, label, front_detection_msg.header)

            detection_msg.boxes.append(detection)
            front_detection_msg.boxes.append(detection)

        # back detections
        back_detection_msg = BoundingBoxArray()
        back_detection_msg.header = cloud_msg.header

        if self.range_detection:
            for box, score, label in zip(back_boxes, back_scores, back_labels):
                box[:2] *= -1
                box[6] += np.pi
                detection = self._convert_model_output_to_jsk_bounding_box(
                    box, score, label, back_detection_msg.header)

                detection_msg.boxes.append(detection)
                back_detection_msg.boxes.append(detection)

        console.print('Boxes: %d' % len(detection_msg.boxes))

        self.cloud_pub.publish(cloud_msg)
        self.detection_pub.publish(detection_msg)
        self.front_detection_pub.publish(front_detection_msg)
        self.back_detection_pub.publish(back_detection_msg)

    def _convert_cloud_to_tensor(self, cloud):
        numified_cloud = ros_numpy.numpify(cloud)
        cloud = np.zeros((numified_cloud.shape[0], 4), dtype=np.float32)
        cloud[:, 0] = numified_cloud['x']
        cloud[:, 1] = numified_cloud['y']
        cloud[:, 2] = numified_cloud['z']
        cloud[:, 3] = numified_cloud['i']

        front_points = np.where(cloud[:, 0] > -1)
        back_points = np.where(cloud[:, 0] <= 1)

        front_data = {
            'mode': 'test',
            'lidar': {
                'points': cloud[front_points],
                'annotations': {}
            },
            'metadata': {}
        }
        info = dict()

        front_data, info = self.preprocess(front_data, info)
        front_data, info = self.voxelize(front_data, info)
        front_data, info = self.assign_target(front_data, info)
        front_data, info = self.reformat(front_data, info)

        if self.range_detection:
            reversed_cloud = cloud[:]
            reversed_cloud[:, 0] *= -1
            reversed_cloud[:, 1] *= -1

            back_data = {
                'mode': 'test',
                'lidar': {
                    'points': reversed_cloud[back_points],
                    'annotations': {}
                },
                'metadata': {}
            }
            info = dict()

            back_data, info = self.preprocess(back_data, info)
            back_data, info = self.voxelize(back_data, info)
            back_data, info = self.assign_target(back_data, info)
            back_data, info = self.reformat(back_data, info)

            collated_data = collate_kitti([front_data, back_data])
        else:
            collated_data = collate_kitti([front_data])

        collated_data = example_to_device(collated_data,
                                          torch.device(self.device))

        return collated_data

    def _convert_model_output_to_jsk_bounding_box(self, box, score, label,
                                                  header):
        detection = BoundingBox()
        detection.header = header
        detection.pose.position.x = box[0]
        detection.pose.position.y = box[1]
        detection.pose.position.z = box[2]
        detection.dimensions.x = box[4]
        detection.dimensions.y = box[3]
        detection.dimensions.z = box[5]

        theta = -(box[6] + np.pi / 2)
        rot = R.from_rotvec(theta * np.array([0, 0, 1]))    # type: R
        quat = rot.as_quat()
        detection.pose.orientation.x = quat[0]
        detection.pose.orientation.y = quat[1]
        detection.pose.orientation.z = quat[2]
        detection.pose.orientation.w = quat[3]

        detection.value = score
        detection.label = int(label)

        return detection


def parse_args():
    parser = argparse.ArgumentParser(description="MegDet test detector")
    parser.add_argument("--config",
                        default='config.py',
                        help="test config file path")
    parser.add_argument("--checkpoint",
                        default='se-ssd-model.pth',
                        help="checkpoint file")
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--subscribed_topic',
                        default='/kitti/velo/pointcloud',
                        help='ros topic for point cloud')
    parser.add_argument('--range_detection', type=bool, default=False)
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)
    return args


def print_info(args):
    table = Table(show_header=False, show_edge=True, box=rich_box.SIMPLE)
    table.add_column('key', justify='right', min_width=36)
    table.add_column('value', justify='left', style='bold green', min_width=36)

    table.add_row('ROS Node:', rospy.get_name())
    table.add_row('Config:', args.config)
    table.add_row('Model:', 'SE-SSD')
    table.add_row('Checkpoint:', args.checkpoint)
    table.add_row('Enabled range detection:', str(args.range_detection))
    table.add_row('Subscribed topic (PointCloud2):', args.subscribed_topic)
    table.add_row('Published topic (BoundingBoxArray):',
                  Callback.DETECTION_PUBLISH_TOPIC)
    table.add_row('Published topic (BoundingBoxArray):',
                  Callback.FRONT_DETECTION_PUBLISH_TOPIC)
    table.add_row('Published topic (BoundingBoxArray):',
                  Callback.BACK_DETECTION_PUBLISH_TOPIC)
    table.add_row('Published topic (PointCloud2):',
                  Callback.CLOUD_PUBLISH_TOPIC)

    panel = Panel(Align(table, align='center'),
                  title='ROS Wrapper for 3D LiDAR Detection',
                  width=console.width)
    console.print(panel, justify='center')


def main():
    args = parse_args()

    rospy.init_node('se_ssd', anonymous=True)
    print_info(args)

    with console.status('[bold green]Loading configuration ...'):
        cfg = torchie.Config.fromfile(args.config)
        if cfg.get("cudnn_benchmark", False):    # False
            torch.backends.cudnn.benchmark = True

        # cfg.model.pretrained = None
        # cfg.data.test.test_mode = True
        cfg.data.val.test_mode = True

        # init distributed env first, since logger depends on the dist info.
        if args.launcher == "none":
            distributed = False
        else:
            distributed = True
            init_dist(args.launcher, **cfg.dist_params)

        # build the model and load checkpoint
        model = build_detector(cfg.model,
                               train_cfg=None,
                               test_cfg=cfg.test_cfg)
        checkpoint_path = os.path.join(cfg.work_dir, args.checkpoint)
        checkpoint = load_checkpoint(model,
                                     checkpoint_path,
                                     map_location="cpu")

        # old versions did not save class info in checkpoints, this workaround is for backward
        # compatibility
        if "CLASSES" in checkpoint["meta"]:
            model.CLASSES = checkpoint["meta"]["CLASSES"]
        else:
            raise NotImplementedError("cannot retrieve object classes")

        model = MegDataParallel(model, device_ids=[0])
        console.log('Loaded configuration successfully.')

    with console.status('[bold green]Initializing ROS wrapper ...'):
        callback = Callback(model, cfg, 'cuda', distributed,
                            args.range_detection)

        rospy.Subscriber(args.subscribed_topic, PointCloud2, callback)
        console.log('Initialized ROS wrapper successfully.')

    console.log('[bold yellow]Start working ...')
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
