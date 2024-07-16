import os
import time
from copy import copy

import cv2
import numpy as np
import rospy
import rospkg
import torch
from control_msgs.msg import FollowJointTrajectoryGoal
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float32MultiArray, String
from relaxed_ik_ros1.msg import EEPoseGoals
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Twist, Vector3

from scipy.spatial.transform import Rotation as R

from act_plus_plus.real_robot.fetch_state import FetchStateEnv, prepare_conf
from act_plus_plus.real_robot.ranged_ik.robot import Robot
# from core.real_robot.vr_controller import VRController
from act_plus_plus.real_robot.vr_controller_remote import VRController

my_path = os.path.dirname(os.path.abspath(__file__))


class FetchEnv(FetchStateEnv):

    def __init__(self, task):
        super().__init__(task)

        self.task = task
        self.test_time = True
        self.in_recording = False
        self.ready_for_action = False
        self.step_count = 0
        self.max_step = prepare_conf[task]['max_len']
        self.pub = rospy.Publisher('/move_joint_by_small_pose_change', Float32MultiArray, queue_size=1)
        self.joint_act_pub = rospy.Publisher('/move_joint_by_small_joint_change', Float32MultiArray, queue_size=1)
        self.process_get_ready_pub = rospy.Publisher('/process_get_ready', String, queue_size=1)
        self.process_post_action_pub = rospy.Publisher('/process_post_action', String, queue_size=1)
        rospy.Subscriber("/ready_for_action", String, self.process_ready_for_action)
        time.sleep(2)

        self.process_get_ready_pub.publish(f"{self.task}|{str(self.test_time)}")
        self.observed_btn_states = [[0, 0, 0, 0, 0, 0]]

        # ranged IK setup
        self.ee_pose_pub = rospy.Publisher('relaxed_ik/ee_pose_goals', EEPoseGoals, queue_size=5)
        self.reset_pub = rospy.Publisher('relaxed_ik/reset', JointState, queue_size=1)
        rospy.Subscriber('/relaxed_ik/joint_angle_solutions', JointState, self.joint_angle_solutions_callback)

        path_to_src = rospkg.RosPack().get_path('relaxed_ik_ros1') + '/relaxed_ik_core'
        deault_setting_file_path = path_to_src + '/configs/settings.yaml'

        # setting_file_path = rospy.get_param('setting_file_path')
        # if setting_file_path == '':
        setting_file_path = deault_setting_file_path

        self.ranged_ik_robot = Robot(setting_file_path)
        self.starting_ee_poses = self.ranged_ik_robot.fk(self.joint_state)
        self.tolerance = Twist(Vector3(0.001, 0.001, 0.001), Vector3(0.0, 0.0, 0.0))

        print("Waiting for robot to be ready for action ...")
        while not self.ready_for_action:
            rospy.sleep(0.1)

    def process_ready_for_action(self, msg):
        if msg.data == "ready_for_action":
            self.ready_for_action = True

    def seed(self, seed):
        np.random.seed(seed)

    def step(self, action):

        f = 10
        action[9:13] = np.round(action[9:13])
        print(f"step {self.step_count} :", action)
        self.observed_btn_states.append(action[9:15])

        message = Float32MultiArray()
        message.data = list(np.array(action))
        self.joint_act_pub.publish(message)
        time.sleep(1.0 / f / prepare_conf[self.task]['speed'])

        obs = self.get_obs_joint()
        reward = 0
        done = False
        info = {}
        self.step_count += 1

        if not self.test_time:
            cv2.imshow("img", self.rgb_image)
            cv2.waitKey(5)
        else:
            cv2.imwrite("robot_img.jpg", self.rgb_image)

        if action[-1] == 1 or self.step_count > self.max_step:
            done = True
            self.process_post_action_pub.publish(self.task)
            time.sleep(2)
            exit(0)

        return obs, reward, done, info

    def step_dagger(self, actions, duration=3.0):
        print("step_dagger", actions.shape)
        actions = actions.cpu().numpy()

        f = 10

        robot_state = self.get_robot_state()
        # Initialize trajectory goal for the arm
        arm_goal = FollowJointTrajectoryGoal()
        arm_goal.trajectory = JointTrajectory()
        arm_goal.trajectory.header.frame_id = "base_link"
        arm_goal.trajectory.joint_names = self.arm_joints_names

        # Loop through the provided trajectories for the arm
        time_from_start = 1.0 / f
        steps = int(f * duration)
        for i in range(steps):
            # if i > 0:
            #     state_change = np.array(actions[i]) - np.array(actions[i - 1])
            #     if np.max(np.abs(state_change)) > 0.3:
            #         print("state change too large, skip it", state_change)
            #         actions[i] = actions[i - 1]

            state = actions[i]
            # Create a trajectory point
            point = JointTrajectoryPoint()
            point.positions = tuple(state[:7])
            point.time_from_start = rospy.Duration(time_from_start + i / f)
            arm_goal.trajectory.points.append(point)

        robot_traj = RobotTrajectory()
        robot_traj.joint_trajectory = arm_goal.trajectory

        display_traj = DisplayTrajectory()
        display_traj.model_id = "fetch"
        display_traj.trajectory_start = robot_state
        display_traj.trajectory.append(robot_traj)

        # Publish the trajectory for visualization in RViz
        self.pub_trajectory.publish(display_traj)
        if self.test_time:
            feed_back = "y"
        else:
            # feed_back = input("Execute on real robot?")
            feed_back = "n"

        if feed_back == "y":
            for i in range(steps):
                self.step(actions[i])
        elif feed_back == "s":
            print("skipping execution, no demo")
        elif feed_back == "q":
            self.process_post_action_pub.publish(self.task)
            time.sleep(2)
            exit(0)
        else:
            print("skipping execution, ask expert for label")
            self.start_record(f)

        return actions

    def reset(self):
        return

    def get_obs_joint(self):
        imgs = [self.rgb_image, self.depth_image]

        all_joint_states = copy(self.joint_state)
        all_joint_states.extend(self.gripper_state)
        all_joint_states.extend(self.observed_btn_states[-1])
        all_joint_states.extend(self.amcl_pose)
        all_joint_states.extend([0])  # stop signal

        return imgs, all_joint_states

    @staticmethod
    def process_vr_button_states(button_states):
        touchpad, trigger, grip, menu, direction = False, False, False, False, np.zeros((2,))
        # Converting ulButtonPressed to binary format
        button_states_bin = format(button_states.ulButtonPressed, "032b")
        print("ulButtonPressed in binary: ", button_states_bin)

        # Check if touchpad button has been pressed
        touchpad_button_index = 32
        if button_states.ulButtonPressed & (1 << touchpad_button_index):
            touchpad = True

            touchpad_axis = button_states.rAxis[0]
            # print(touchpad_axis.x,touchpad_axis.y)
            if abs(touchpad_axis.x) < 0.5 and touchpad_axis.y < 0:
                direction += np.array([0, -1])
            elif abs(touchpad_axis.x) < 0.5 and touchpad_axis.y > 0:
                direction += np.array([0, 1])
            elif abs(touchpad_axis.y) < 0.5 and touchpad_axis.x < 0:
                direction += np.array([-1, 0])
            elif abs(touchpad_axis.y) < 0.5 and touchpad_axis.x > 0:
                direction += np.array([1, 0])

        # Check if trigger button has been pressed
        trigger_button_index = 33
        if button_states.ulButtonPressed & (1 << trigger_button_index):
            trigger = True

        # Check if grip button has been pressed
        grip_button_index = 2
        if button_states.ulButtonPressed & (1 << grip_button_index):
            grip = True

        # Check if menu button has been pressed
        menu_button_index = 1
        if button_states.ulButtonPressed & (1 << menu_button_index):
            menu = True

        return touchpad, trigger, grip, menu, direction

    def start_record(self, f=10):

        # Initialize episode_id
        self.episode_id = self.get_episode_id()


        # Start recording
        msg = np.array(rospy.wait_for_message('/pose_msg', Float32MultiArray).data)
        ee_position, ee_rotation = msg[:3], msg[3:]
        ee_rotation = R.from_quat(ee_rotation).as_matrix()

        # Convert the rotation matrix to Euler angles (ZYX convention for swapping)
        ee_rotation_zyx = R.from_matrix(ee_rotation).as_euler('XYZ')

        # Reorder the angles and invert the signs
        ee_rotation_zyx = np.array([ee_rotation_zyx[1], ee_rotation_zyx[2], ee_rotation_zyx[0]])
        ee_rotation_zyx[0] *= -1
        ee_rotation_zyx[2] *= -1

        # Convert the Euler angles back to a rotation matrix
        ee_rotation = R.from_euler('XYZ', ee_rotation_zyx).as_matrix()

        vr_controller = VRController()
        controller_index = vr_controller.get_controller_index()[0]

        vr_position, vr_rotation = vr_controller.sync(controller_index, ee_position, ee_rotation)

        # R_B = R_AB * R_A

        print("Recording...")
        # Initialize image and joint state containers
        self.rgb_images = []
        self.depth_images = []
        self.point_clouds = []
        self.joint_states = []
        self.gripper_states = []
        self.controller_btn_states = []
        self.acml_poses = []

        # Start recording
        while not rospy.is_shutdown():
            self.rgb_images.append(self.rgb_image)
            self.depth_images.append(self.depth_image)
            self.point_clouds.append(self.point_cloud)
            all_joint_states = copy(self.joint_state)
            all_joint_states.extend(self.gripper_state)
            self.joint_states.append(all_joint_states)
            self.acml_poses.append(self.amcl_pose)
            # print(self.joint_state)

            # move robot
            controller_index = vr_controller.get_controller_index()[0]

            np_pose = vr_controller.get_controller_position(controller_index)
            new_position = np_pose[:, 3]
            translate = new_position - vr_position
            translate = translate[[2, 0, 1]]
            translate[0] *= -1
            translate[1] *= -1
            new_ee_position = ee_position + translate

            new_vr_rotation = np_pose[:, :-1]
            new_vr_rotation_xyz = R.from_matrix(new_vr_rotation).as_euler('xyz')
            new_vr_rotation_xyz[0] *= -1
            new_vr_rotation_xyz[2] *= -1
            new_vr_rotation = R.from_euler('xyz', new_vr_rotation_xyz[[2, 0, 1]]).as_matrix()

            new_rotation = R.from_matrix(new_vr_rotation).as_quat()
            # print("translate", translate)

            # get trigger state
            # _, button_states = vr_controller.vr_system.getControllerState(controller_index)
            # touchpad, trigger, grip, menu, direction = self.process_vr_button_states(button_states)
            controller_index = vr_controller.get_controller_index()[0]
            touchpad, trigger, grip, menu, direction = vr_controller.get_controller_button_state(controller_index)


            if menu and grip:
                # exit
                break

            pose = np.zeros((13,))
            pose[:3] = new_ee_position
            pose[3:7] = new_rotation
            pose[7] = int(touchpad)
            pose[8] = int(trigger)
            pose[9] = int(grip)
            pose[10] = int(menu)
            pose[11:] = direction
            message = Float32MultiArray()
            message.data = list(pose)
            self.pub.publish(message)
            self.controller_btn_states.append(pose[7:])

            cv2.imshow("img", self.rgb_image)
            cv2.waitKey(5)
            time.sleep(1.0 / f)
            cv2.imshow("img", self.rgb_image)
            cv2.waitKey(5)

        self.process_post_action_pub.publish(self.task)
        # self.move_arm_by_pose(ee_position, ee_rotation, f=f)
        # Save the data
        self.save_data(f)
        exit(0)

    def joint_angle_solutions_callback(self, data):

        print("curr joint", self.joint_state)
        print("goal joint", data.position)
        print("starting_ee_poses", self.ranged_ik_robot.fk(self.joint_state))
        print("goal_ee_poses", self.ranged_ik_robot.fk(data.position))

        state_change = np.array(data.position) - np.array(self.joint_state)
        # print("joint state change", state_change)
        if np.max(np.abs(state_change)) > 0.3:
            print("state change too large, skip it", state_change)
            return

        self.move_joint(data.position, f=15)

    def reset_ik(self):
        self.js_msg = JointState()
        self.js_msg.name = self.ranged_ik_robot.articulated_joint_names
        self.js_msg.position = []
        self.js_msg.header.stamp = rospy.Time.now()
        self.js_msg.position = list(self.joint_state)
        self.reset_pub.publish(self.js_msg)

    def move_arm_by_pose(self, position, rotation, f=20, speed=2.0):
        self.starting_ee_poses[0].position.x = position[0]
        self.starting_ee_poses[0].position.y = position[1]
        self.starting_ee_poses[0].position.z = position[2]
        self.starting_ee_poses[0].orientation.x = rotation[0]
        self.starting_ee_poses[0].orientation.y = rotation[1]
        self.starting_ee_poses[0].orientation.z = rotation[2]
        self.starting_ee_poses[0].orientation.w = rotation[3]

        self.reset_ik()
        ee_pose_goals = EEPoseGoals()
        ee_pose_goals.header.stamp = rospy.Time.now()
        ee_pose_goals.ee_poses.append(self.starting_ee_poses[0])
        ee_pose_goals.tolerances.append(self.tolerance)
        self.ee_pose_pub.publish(ee_pose_goals)

    def get_episode_id(self):
        dir_path = f"{my_path}/../env/expert_demo/{self.task}"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        existing_dirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
        existing_ids = [int(d.split('_')[-1]) for d in existing_dirs]
        episode_id = max(existing_ids) + 1 if existing_ids else 0
        return episode_id

    def save_data(self, f):
        print("Saving data...")
        print(self.joint_states[0])
        dir_path = f"{my_path}/../env/expert_demo/{self.task}/episode_{self.episode_id}"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        for i, img in enumerate(self.rgb_images):
            cv2.imwrite(os.path.join(dir_path, "rgb_{}.jpg".format(i)), img)

        for i, img in enumerate(self.depth_images):
            cv2.imwrite(os.path.join(dir_path, "depth_{}.jpg".format(i)), img)

        # Again, note that you may need to adapt the following to your specific needs
        self.point_clouds = np.array(self.point_clouds)
        self.point_clouds = np.nan_to_num(self.point_clouds)
        np.save(os.path.join(dir_path, "point_clouds.npy"), self.point_clouds)
        np.save(os.path.join(dir_path, "joint_states.npy"), np.array(self.joint_states))
        np.save(os.path.join(dir_path, "frequency.npy"), np.array(f))
        np.save(os.path.join(dir_path, "controller_btn_states.npy"), np.array(self.controller_btn_states))
        np.save(os.path.join(dir_path, "acml_poses.npy"), np.array(self.acml_poses))

        stop_signal = np.zeros(len(self.acml_poses))
        stop_signal[-1] = 1
        np.save(os.path.join(dir_path, "stop_signal.npy"), np.array(stop_signal))

        print(self.controller_btn_states[0])
        print(np.array(self.joint_states).shape)

    def replay_traj(self, traj, f):
        for state in traj:  # Exclude the last two joints for the gripper
            self.move_joint(state, f)


if __name__ == '__main__':
    env = FetchEnv("grasp_bottle")
    env.reset()

    time_start = time.time()
    for i in range(100):
        pos, rot = env.solve_fk(env.get_robot_state())
        # print(pos, rot)
    print("time: ", time.time() - time_start)
