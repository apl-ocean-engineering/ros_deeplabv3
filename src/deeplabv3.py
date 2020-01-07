#!/usr/bin/env python3
import rospy
import numpy as np
import torch
import time
from scipy.io import loadmat

import cv2
from cv_bridge import CvBridge, CvBridgeError

from deepLabv3.utils import load_model, cv_image_to_tensor, create_color_cv_image
from deepLabv3.detector import Detector
from deepLabv3.argLoader import ArgLoader
import deepLabv3.classes as classes

from std_msgs.msg import Header
from sensor_msgs.msg import Image

from manipulation_context_slam_msgs.msg import DetectionMetaData, SemanticDetectionMat


class DeepLabWrapper:
    def __init__(self):
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        self.bridge = CvBridge()
        self.cv_image = np.zeros((10, 10, 3), np.uint8)

        self.args = self._args
        self.cmap = self._cmap
        self.detector = self._load_detector

        self.md = DetectionMetaData()
        self.md.publishing = False

        self.image_sub = rospy.Subscriber(
                "/camera/rgb/image_raw",
                Image, self.img_callback)

        self.metadata_publisher = rospy.Publisher(
            '/detection/metadata', DetectionMetaData, queue_size=10)
        self.semantic_publisher = rospy.Publisher(
            '/detection/semantic', SemanticDetectionMat, queue_size=10)

    def img_callback(self, img):
        #print("here")
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(img, "bayer_grbg8")
        except CvBridgeError as e:
            print(e)

    @property
    def _args(self):
        argloader = ArgLoader()
        argloader.parser.add_argument(
            "--saved_model", type=str, required=True, default=" ")
        argloader.parser.add_argument(
            "--cmap_location", type=str, default=" ")

        args = argloader.args

        return args

    @property
    def _load_detector(self):
        if self.args.saved_model == " ":
            print("No model specified")
            exit()
        model, model_fname = load_model(self.args, classes.VOC_classes)
        torch.cuda.set_device(self.args.gpu)
        model = model.cuda()
        model.eval()
        checkpoint = torch.load(self.args.saved_model)
        state_dict = {k[7:]: v for k, v in checkpoint['state_dict'].items()
                      if 'tracked' not in k}

        model.load_state_dict(state_dict)

        detector = Detector(model)

        return detector

    @property
    def _cmap(self):
        if self.args.cmap_location != " ":
            cmap = loadmat(self.args.cmap_location)['colormap']
            cmap = (cmap * 255).astype(np.uint8).flatten().tolist()

            return cmap

        return None

    def run(self):
        loop_rate = rospy.Rate(30)
        for i, val in enumerate(classes.VOC_classes):
            self.md.classes.append(val)
        self.md.class_nums = i + 1

        while not rospy.is_shutdown():
            #print('here')
            prev_time = time.time()

            header = Header()
            header.frame_id = '/camera/left'

            self.md.header = header
            self.md.header.stamp = rospy.get_rostime()
            if self.cv_image.shape[0] > 60 and self.cv_image.shape[1] > 60:
                color = cv2.cvtColor(self.cv_image, cv2.COLOR_GRAY2RGB)

                img = cv_image_to_tensor(color)

                pred = self.detector.inference(img)
                semantic_detection_mat = SemanticDetectionMat()
                semantic_detection_mat.header = header
                semantic_detection_mat.header.stamp = rospy.get_rostime()
                try:
                    pred_img = self.bridge.cv2_to_imgmsg(pred, "mono8")
                    semantic_detection_mat.detection_mat = pred_img
                except CvBridgeError as e:
                    print(e)

                if self.cmap is not None and self.args.display:
                    cv2.imshow("fname", self.cv_image)
                    cv2.imshow(
                        "output", create_color_cv_image(pred, self.cmap))
                    cv2.waitKey(1)

                self.semantic_publisher.publish(semantic_detection_mat)
                print("time elapsed", time.time() - prev_time)

            self.metadata_publisher.publish(self.md)

            loop_rate.sleep()


if __name__ == "__main__":
    assert torch.cuda.is_available()
    rospy.init_node("ros_deeplab_wrapper")
    dlv3 = DeepLabWrapper()
    dlv3.run()
