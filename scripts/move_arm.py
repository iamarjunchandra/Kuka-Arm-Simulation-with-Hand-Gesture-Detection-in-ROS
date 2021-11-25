from robot_environment.msg import predictionMessage
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import rospy
import sys


class Server:
    def __init__(self,x):
        self.find = x if x in [0,1,2,3]
        self.prediction = None

    def prediciton_callback(self, data):
        # "Store" the message received.
        self.prediction=[data.top_left_image,data.top_right_image,data.bottom_left_image,data.bottom_right_image]

    def move(self):

        if self.prediction is not None:

            self.index=self.prediction.index(self.find)
            self.pub=rospy.Publisher('/mmt_workspace/arm_controller/command',JointTrajectory,queue_size=10)
            self.msg=JointTrajectory()
            self.msg.header.stamp=rospy.Time.now()
            self.msg.header.frame_id=''
            self.msg.joint_names=['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6']
            self.point=JointTrajectoryPoint()
            if self.index==0:
                self.point.positions=[0.5,0,0,0,0,0]
            elif self.index==1:
                self.point.positions=[2,0,0,0,0,0]
            elif self.index==2:
                self.point.positions=[-0.5,0,0,0,0,0]
            elif self.index==2:
                self.point.positions=[-2,0,0,0,0,0]
            self.point.accelerations=[]
            self.point.effort=[]
            self.point.time_from_start=rospy.Duration(1)
            self.msg.points.append(point)
            self.pub.publish(self.msg)


if __name__ == '__main__':
    try:
        rospy.init_node('move_arm_node',anonymous=True)
        server = Server(int(sys.argv[1]))
        rospy.Subscriber('prediction_publisher', predictionMessage , server.prediciton_callback)
        server.move()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass