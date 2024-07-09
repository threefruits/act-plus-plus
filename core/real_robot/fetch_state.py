import os
import socket
import time

import cv2
import numpy as np
import rospy

from sensor_msgs.msg import JointState
from sensor_msgs.msg import CompressedImage
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal, \
    GripperCommandAction, GripperCommandGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import actionlib
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory, RobotState
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Image, PointCloud2
import ros_numpy

my_path = os.path.dirname(os.path.abspath(__file__))
prepare_conf = {
    'open_fridge': {
        'position': [-4.03109417, 3.71815991, 0.0],
        # 'position': [-3.95109417, 3.65815991, 0.0],
        # 'position': [-3.95109417, 3.65815991, 0.0],
        'orientation': [0.0, 0.0, 0.924788, 0.38313],
        'arm_joints': [0.29470240364379885, 1.1211147828796386, -1.2186898401245116, 1.876310651525879,
                       1.202369851473999, -1.8404571460021972, 1.7227418721292114],
        'torso': 0.37,
        'head': [0, 0.47738],
        'back_before_arm': False,
        'back_and_forth_duration': [0.2, 0],
        'max_len': 550,
        'speed': 2.5
    },
    'close_fridge':{
        'position': [-3.9968581384505035, 3.9624174172152378, 0.0],
        'orientation': [0.0, 0.0, 0.8686781359942997, 0.49537692320642973],
        'arm_joints': [0.29470240364379885, 1.1211147828796386, -1.2186898401245116, 1.876310651525879,
                       1.202369851473999, -1.8404571460021972, 1.7227418721292114],
        'torso': 0.37,
        'head': [0, 0.5705784579223633],
        'back_before_arm': False,
        'back_and_forth_duration': [0.2, 0],
        'max_len': 550,
        'speed': 2.5
    },
    'grasp_food_from_fridge': {
        'position': [-4.0856, 4.23324, 0.0],
        'orientation': [0.0, 0.0, 0.974788, 0.23313],
        'arm_joints': [0.29470240364379885, 1.1211147828796386, -1.2186898401245116, 1.876310651525879,
                       1.202369851473999, -1.8404571460021972, 1.7227418721292114],
        'torso': 0.37,
        'head': [0, 0.68738],
        'back_before_arm': True,
        'back_and_forth_duration': [2, 3],
        'max_len': 130,
        'speed': 2.0
    },
    'placeon_food_counter': {
        'position': [-4.0056, 3.515325, 0.0],
        'orientation': [0.0, 0.0, -0.99409, 0.1085167],
        'arm_joints': [0.06690632591552734, 0.993794016430664, -1.5496464183792114, 2.2291262795776365,
                       1.1632533496658326, -1.9716121600402832, 2.187921196279297],
        'torso': 0.37,
        'head': [0, 0.47738],
        'back_before_arm': False,
        'max_len': 150,
        'speed': 2.0
    },
    'close_fridge': {
        'position': [-3.84106148, 4.01676566, 0.0],
        'orientation': [0.0, 0.0, 0.82902942, 0.559204980],
        'arm_joints': [0.06690632591552734, 0.993794016430664, -1.5496464183792114, 2.2291262795776365,
                       1.1632533496658326, -1.9716121600402832, 2.187921196279297],
        'torso': 0.37,
        'head': [0, 0.68738],
        'back_before_arm': False,
        'max_len': 300,
        'speed': 2.0
    },
    'open_microwave': {
        'position': [-3.900177, 2.674713112, 0.0],
        # 'position': [-3.800177, 2.674713112, 0.0],
        # 'position': [-3.8800177, 2.674713112, 0.0],
        # 'position': [-3.910177, 2.674713112, 0.0],
        'orientation': [0.0, 0.0, 0.9991653614575754, 0.040848],
        'arm_joints': [0.06690632591552734, 0.993794016430664, -1.5496464183792114, 2.2291262795776365,
                       1.1632533496658326, -1.9716121600402832, 2.187921196279297],
        'torso': 0.37,
        'head': [0, 0.47738],
        'back_before_arm': False,
        'max_len': 600,
        'speed': 1.5
    },
    'grasp_food_from_table': {
        'position': [-4.0056, 3.515325, 0.0],
        'orientation': [0.0, 0.0, -0.998496, 0.06072118],
        'arm_joints': [0.06690632591552734, 0.993794016430664, -1.5496464183792114, 2.2291262795776365,
                       1.1632533496658326, -1.9716121600402832, 2.187921196279297],
        'torso': 0.37,
        'head': [0, 0.47738],
        'back_before_arm': False,
        'max_len': 200,
        'speed': 2.0
    },
    'placein_microwave': {
        # 'position': [-3.9957247443, 2.985438424, 0.0],
        # 'position': [-3.9957247443, 3.085438424, 0.0],
        'position': [-3.9657247443, 3.015438424, 0.0],
        'orientation': [0.0, 0.0, -0.968685166, 0.248292263],
        'arm_joints': [0.06690632591552734, 0.993794016430664, -1.5496464183792114, 2.2291262795776365,
                       1.1632533496658326, -1.9716121600402832, 2.187921196279297],
        'torso': 0.37,
        'head': [0, 0.47738],
        'back_before_arm': False,
        'max_len': 210,
        'speed': 1.5
    },
    'close_microwave': {
        # 'position': [-3.9532, 2.788273, 0.0],
        # 'position': [-3.9532, 2.738273, 0.0],
        # 'position': [-3.9532, 2.708273, 0.0],
        'position': [-3.9832, 2.788273, 0.0],

        'orientation': [0.0, 0.0, -0.9743223, 0.2],
        'arm_joints': [0.06690632591552734, 0.993794016430664, -1.5496464183792114, 2.2291262795776365,
                       1.1632533496658326, -1.9716121600402832, 2.187921196279297],
        'torso': 0.37,
        'head': [0, 0.47738],
        'back_before_arm': True,
        'max_len': 350,
        'speed': 1.5
    },
    'switchon_microwave': {
        # 'position': [-3.91041365462, 2.72948194, 0.0],
        # 'position': [-3.88041365462, 2.75948194, 0.0],
        # 'position': [-3.88041365462, 2.72948194, 0.0],
        'position': [-3.92041365462, 2.73248194, 0.0],
        # 'position': [-3.94041365462, 2.73248194, 0.0],
        'orientation': [0.0, 0.0, -0.9989, 0.04558719],
        # 'orientation': [0.0, 0.0, -0.99999, 0.0001],
        'arm_joints': [0.06690632591552734, 0.993794016430664, -1.5496464183792114, 2.2291262795776365,
                       1.1632533496658326, -1.9716121600402832, 2.187921196279297],
        'torso': 0.37,
        'head': [0, 0.47738],
        'back_before_arm': False,
        'max_len': 220,
        'speed': 1.2
    },
    'grasp_food_from_microwave': {
        # 'position': [-3.88306647, 2.76382929, 0.0],
        # 'position': [-3.93306647, 2.85382929, 0.0],
        # 'position': [-3.97306647, 2.88382929, 0.0],
        # 'position': [-3.93306647, 2.74382929, 0.0],
        'position': [-3.93306647, 2.76382929, 0.0],
        # 'orientation': [0.0, 0.0, -0.9986960, 0.051050903],
        'orientation': [0, 0, -0.9958808, 0.0906716],
        'arm_joints': [0.06690632591552734, 0.993794016430664, -1.5496464183792114, 2.2291262795776365,
                       1.1632533496658326, -1.9716121600402832, 2.187921196279297],
        'torso': 0.37,
        # 'torso': 0.35,
        'head': [0, 0.47738],
        'back_before_arm': False,
        'max_len': 220,
        'speed': 1.5
    },
    'placeon_food_dinning_table': {
        'position': [0.06236604, -2.562564, 0.0],
        'orientation': [0.0, 0.0, -0.9901, 0.139951],
        'arm_joints': [0.06690632591552734, 0.993794016430664, -1.5496464183792114, 2.2291262795776365,
                       1.1632533496658326, -1.9716121600402832, 2.187921196279297],
        'torso': 0.37,
        'head': [0, 0.72],
        'back_before_arm': False,
        'max_len': 150,
        'speed': 2.0
    },
    'pickplace_pen': {
        'position': [-2.6788816, 0.8, 0.0],
        'orientation': [0.0, 0.0, 0.74599863, 0.6659474],
        'arm_joints': [-0.8458124469726562, 0.06995397050170898, -0.8666412046417237, -1.5229913542419433,
                       1.3369766897003175, 2.0255578114257813, 0.5431105793092346],
        'torso': 0.37,
        'head': [-0.1, 0.6],
        'back_before_arm': True,
        'back_and_forth_duration': [2, 5],
        'max_len': 650,
        'speed': 2.0
    },
}


class FetchStateEnv(object):

    def __init__(self, node, light=False):

        self.set_ROS_IP_address()
        # rospy.init_node(node, anonymous=True)
        try:
            rospy.init_node(node, anonymous=True)
        except rospy.exceptions.ROSException as e:
            print("Node has already been initialized, do nothing")

        # Initialize image and joint state containers
        self.rgb_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.depth_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.point_cloud = None
        self.joint_state = None
        self.arm_joint_goal = None
        self.whole_body_joint_state = None
        self.gripper_state = None
        self.amcl_pose = np.array([0, 0, 0, 1])

        # cv2.namedWindow("img")
        # cv2.moveWindow("img", 0, 0)
        # cv2.namedWindow("depth")
        # cv2.moveWindow("depth", 650, 0)

        # Initialize image and joint state subscribers

        # rospy.Subscriber('/joint_states', JointState, self.joint_states_callback)
        rospy.Subscriber('/joint_states_throttle', JointState, self.joint_states_callback)
        rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.amcl_pose_callback)
        if not light:
            # rospy.Subscriber('/head_camera/rgb/image_raw/compressed', CompressedImage, self.rgbc_callback)
            rospy.Subscriber('/head_camera/rgb/image_raw/compressed_throttle', CompressedImage, self.rgbc_callback)
            # rospy.Subscriber('/head_camera/depth/image_raw/compressed', CompressedImage, self.depthc_callback)
            rospy.Subscriber('/head_camera/depth/image_raw/compressed_throttle', CompressedImage, self.depthc_callback)
            # rospy.Subscriber('/head_camera/depth_downsample/points', PointCloud2, self.point_cloud_callback)

        while self.joint_state is None:
            rospy.sleep(0.1)
        print("Fetch Robot created!")

        # Define the arm joints names
        self.arm_joints_names = ["shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint", "elbow_flex_joint",
                                 "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]
        self.whole_body_joint_names = ["l_wheel_joint", "r_wheel_joint", "torso_lift_joint", "bellows_joint",
                                       "head_pan_joint", "head_tilt_joint", "shoulder_pan_joint",
                                       "shoulder_lift_joint", "upperarm_roll_joint", "elbow_flex_joint",
                                       "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint"]

        # Create a ROS publisher on the trajectory action server for the arm
        self.arm_traj_client = actionlib.SimpleActionClient('arm_controller/follow_joint_trajectory',
                                                            FollowJointTrajectoryAction)
        self.arm_traj_client.wait_for_server()
        print("Fetch Robot actionlib 0 connected!")

        self.gripper_traj_client = actionlib.SimpleActionClient('gripper_controller/gripper_action',
                                                                GripperCommandAction)
        self.gripper_traj_client.wait_for_server()
        print("Fetch Robot actionlib connected!")

        if not light:
            self.pub_trajectory = rospy.Publisher('/move_group/display_planned_path_fetch',
                                                  DisplayTrajectory,
                                                  latch=True,
                                                  queue_size=5)

    @staticmethod
    def set_ROS_IP_address():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't even have to be reachable
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except:
            IP = '127.0.0.1'
        finally:
            s.close()

        os.environ["ROS_IP"] = IP

    def rgbc_callback(self, data):
        # cv_image = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding='bgr8')
        # cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')

        str_msg = data.data
        buf = np.ndarray(shape=(1, len(str_msg)),
                         dtype=np.uint8, buffer=data.data)
        cv_image = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)
        self.rgb_image = cv_image.copy()

    def depthc_callback(self, data):
        # cv_image = self.bridge.compressed_imgmsg_to_cv2(data)
        str_msg = data.data
        buf = np.ndarray(shape=(1, len(str_msg)),
                         dtype=np.uint8, buffer=data.data)
        cv_image = cv2.imdecode(buf, cv2.IMREAD_ANYCOLOR)

        depth_normalized = cv2.normalize(cv_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_normalized = depth_normalized[..., None].repeat(3, -1)
        self.depth_image = depth_normalized

    def joint_states_callback(self, data):
        # parse data into list of joint values
        if len(data.position) == 2:
            self.gripper_state = list(data.position)
        else:
            self.joint_state = list(data.position)[-7:]
            self.whole_body_joint_state = list(data.position)
        # print(self.gripper_state)

    def amcl_pose_callback(self, data):
        # parse data into list of joint values
        pose = data.pose.pose
        self.amcl_pose = np.array([pose.position.x, pose.position.y, pose.orientation.z, pose.orientation.w])
        # print(self.gripper_state)

    def point_cloud_callback(self, ros_point_cloud):
        pc = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(ros_point_cloud, remove_nans=False)
        self.point_cloud = pc.copy()
        # pass

    def get_robot_state(self):
        robot_state = RobotState()
        robot_state.joint_state.header.frame_id = "base_link"

        for i in range(len(self.whole_body_joint_names)):
            robot_state.joint_state.name.append(self.whole_body_joint_names[i])
            robot_state.joint_state.position.append(self.whole_body_joint_state[i])
        return robot_state

    def move_joint(self, joints, f, speed=2.0):

        # if self.arm_joint_goal is not None:
        #     state_change = np.array(self.arm_joint_goal[:7]) - np.array(self.joint_state[:7])
        #     print("diff to goal:", state_change)

        # Initialize trajectory goal for the arm
        arm_goal = FollowJointTrajectoryGoal()
        arm_goal.trajectory = JointTrajectory()
        arm_goal.trajectory.joint_names = self.arm_joints_names

        # Loop through the provided trajectories for the arm
        time_from_start = 1.0 / f / speed

        # Create a trajectory point
        point = JointTrajectoryPoint()
        point.positions = tuple(joints[:7])
        point.time_from_start = rospy.Duration(time_from_start)
        arm_goal.trajectory.points.append(point)

        # Send the goal to the action server for the arm
        self.arm_traj_client.send_goal(arm_goal)

        goal = GripperCommandGoal()
        # goal.command.position = joints[-1] * 2
        goal.command.position = 0.5 if joints[-1] >= 0.03 else 0.001
        goal.command.max_effort = 150
        # goal.command.max_effort = 50
        # print("Gripper goal: ", goal.command.position, joints[-2:])

        # self.arm_traj_client.wait_for_result(rospy.Duration(1.0 / f))
        self.gripper_traj_client.send_goal(goal)
        self.arm_joint_goal = joints[:7]


if __name__ == "__main__":
    fetch_state = FetchStateEnv("fetch_state_env")
    rospy.spin()
