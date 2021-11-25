#!/usr/bin/env

import numpy as np
import torch
from torchvision.io import read_image
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision.utils import save_image
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import message_filters
import cv2
import rospy
from robot_environment.msg import predictionMessage
from os.path import dirname, abspath

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.image_features_ = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv2d(16, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv2d(256, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.linear_features_ = nn.Sequential(
            nn.Linear(128*6*6, 128*6*6*2),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128*6*6*2, 128*6*8),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128*6*8, 4),
        )
    def forward(self, x):
        x = self.image_features_(x)
        x=x.view(-1, 128*6*6)
        x=self.linear_features_(x)
        return x


def callback(rgb_msg, camera_info):
    bgr_image = CvBridge().imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
    camera_info_K = np.array(camera_info.K).reshape([3,3])
    camera_info_D = np.array(camera_info.D)
    image = cv2.undistort(bgr_image, camera_info_K, camera_info_D)

    #Loading and implementing pytorch model saved in model directory
    dir_=dirname(dirname(abspath(__file__)))
    device = torch.device("cpu")
    loaded_model=NeuralNetwork()
    loaded_model.load_state_dict(torch.load(dir_+ '/model/hand_gesture.pt',map_location=device))
    loaded_model.to(device)
    loaded_model.eval()

    #transforming image
    image=np.transpose(image,(2,0,1))
    image=torch.from_numpy(image)
    image=transforms.functional.adjust_brightness(image,4)
    crop_image=transforms.CenterCrop((720,720))
    image=crop_image(image)

    #Transforms to match trained data
    transform=transforms.Compose([transforms.Resize(256),
                                transforms.RandomCrop((224,224)),
                                transforms.Normalize(
                                            (177.3580, 169.3307, 160.4894),
                                            (37.1419, 42.7550, 50.8975))])

    #image patches
    img_patches=torch.stack([transform(image[:, 110:330, 110:330].float()),
                transform(image[:, 110:330, 390:610].float()),
                transform(image[:, 390:610, 110:330].float()),
                transform(image[:, 390:610, 390:610].float())],dim=0)

    #prediction for image patches
    predictions=np.array(loaded_model(img_patches).argmax(dim=1))

    #Publishing the prediction
    pred_pub=rospy.Publisher('prediction_publisher',predictionMessage,queue_size=10)
    msg=predictionMessage()
    msg.top_left_image=predictions[0]
    msg.top_right_image=predictions[1]
    msg.bottom_left_image=predictions[2]
    msg.bottom_right_image=predictions[3]
    pred_pub.publish(msg)

    #save the image with predicted number in it
    image=image.cpu().numpy().transpose((1, 2, 0))
    image = cv2.putText(image, str(int(predictions[0])), (200,200), cv2.FONT_HERSHEY_SIMPLEX, 
                    3, (255, 0, 0), 2, cv2.LINE_AA)
    image = cv2.putText(image, str(int(predictions[1])), (450,200), cv2.FONT_HERSHEY_SIMPLEX, 
                    3, (255, 0, 0), 2, cv2.LINE_AA)
    image = cv2.putText(image, str(int(predictions[2])), (200,450), cv2.FONT_HERSHEY_SIMPLEX, 
                    3, (255, 0, 0), 2, cv2.LINE_AA)
    image = cv2.putText(image, str(int(predictions[3])), (450,450), cv2.FONT_HERSHEY_SIMPLEX, 
                    3, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imwrite(dir_+'/predictions/output.png',image)


def camera_sub():
    global image
    rospy.init_node('cam_info_receiver_python_node', anonymous=True)
    sub_camera=message_filters.Subscriber('/distorted_camera/link/camera/image', Image)
    sub_info = message_filters.Subscriber('/distorted_camera/link/camera/camera_info', CameraInfo)
    rate=rospy.Rate(1)
    ts = message_filters.ApproximateTimeSynchronizer([sub_camera, sub_info], 10, 0.2)
    ts.registerCallback(callback)
    rate.sleep()
    rospy.spin()

if __name__=='__main__':
    try:
        camera_sub()
    except rospy.ROSInterruptException:
        pass
