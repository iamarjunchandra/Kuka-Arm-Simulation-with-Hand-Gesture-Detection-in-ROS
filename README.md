

My objective was to:
  - Design a machine learning algorithm that classifies (based on a supplied dataset in ```/hand_gestures_dataset/```) the images of the gestures from the camera feed.

  ## A CNN model is built in pytorch which has an accuracy of 98% in the test dataset. When the package is copied to your directory for the first time, from the scripts directory, run bash command `python3 ai_trainer.py`. This will launch the ai_trainer script which will train the CNN model in the hand gesture dataset provided and copy the model parameters to model directory as 'hand_gestures.pt'. From the next run onwards, ROS will take this file for prediction purpose.

  - Spawn a 6-DOF robot on the center of the red cross.
    - You have the freedom to obtain a pre-written urdf of a commercially available robot. Please indicate the source if you do so.

    - Write/Copy in the description of the robot into ```/urdf/workspace.urdf.xacro```
  ## A pre-written URDF of the Kuka Arm 6DOF robot is forked and modified for the task. The world can be launched using the command `roslaunch robot_environment launch_world.launch`. 

  - Implement a control pipeline to control the robot using external commands. You can use any strategy (MoveIt!, custom controllers, etc.)

  ## A control pipleline is built to control the movement of robot using the GAZEBO controller plugin. The 6 joints of the robot can be controlled using the command `rosrun robot_environment myRobo_controller,py p1 p2 p3 p4 p5 p6` where p1 to p6 are the values to which each joint needs to be moved.

  - Write an algorithm that takes in the camera feed from the simulated environment (check Notes to know how to access camera feed), and labels the hand gestures on the base using the previously designed ML model, and saves this output as a .png file.

  ## A ros node with subsciber is built in the camera_vision.py file which subscribes to the camera feed, produce image from the input , feeds to the neural network and finally save the predicted output as `~/predictions/output.png`. The node is attached to the launch_world.launch and hence is called automatically once the world is launched. so no need to launch it seperately. 

  - Design a program that takes in user input (in the form of an int from 0 to 3) and then moves the simulated robot such that it points in the direction of the hand gesture on the base.

  ## A ros node with publisher is built in the move_arm.py file which takes an int from 0-3 and moves the simulated robot such that it points in the direction of the hand gesture on the base. To run the node, run command `rosrun robot_environment move_arm.py x` where x is the int representing 0-3.


  Regards,
  Arjun Chandrababu
  arjun.aiengineer@gmail.com