#!/usr/bin/env
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import sys

def controller_pub(p1,p2,p3,p4,p5,p6):

    pub=rospy.Publisher('/mmt_workspace/arm_controller/command',JointTrajectory,queue_size=10)
    rospy.init_node('controller_python_node',anonymous=True)
    rospy.loginfo('controller node active')
    rate=rospy.Rate(10)

    while not rospy.is_shutdown():

        msg=JointTrajectory()
        msg.header.stamp=rospy.Time.now()
        msg.header.frame_id=''
        msg.joint_names=['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6']

        point=JointTrajectoryPoint()
        point.positions=[p1,p2,p3,p4,p5,p6]
        point.accelerations=[]
        point.effort=[]
        point.time_from_start=rospy.Duration(1)

        msg.points.append(point)
        pub.publish(msg)


        rate.sleep()

if __name__=='__main__':
    try:
        controller_pub(int(sys.argv[1])),int(sys.argv[2]),int(sys.argv[3],int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6]))
    except rospy.ROSInterruptException:
        pass
