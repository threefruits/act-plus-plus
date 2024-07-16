import copy
import os
import sys
import threading
import time

import actionlib
import moveit_commander
import numpy as np
import ros_numpy
import rospy
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest, GetPositionFK, GetPositionFKRequest
from shape_msgs.msg import SolidPrimitive
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float32MultiArray, String
from moveit_commander import PlanningSceneInterface
from moveit_msgs.msg import PlanningScene, CollisionObject
import geometry_msgs.msg
from sensor_msgs.msg import Image, PointCloud2
from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory, RobotState
from geometry_msgs.msg import Twist

from fetch_state import FetchStateEnv, prepare_conf

my_path = os.path.dirname(os.path.abspath(__file__))


class FetchController(FetchStateEnv):

    def __init__(self):
        super(FetchController, self).__init__("fetch_controller", light=True)
        self.step_count = 0

        rospy.wait_for_service('compute_fk')
        self.compute_fk = rospy.ServiceProxy('compute_fk', GetPositionFK)
        rospy.wait_for_service('compute_ik')
        self.compute_ik = rospy.ServiceProxy('compute_ik', GetPositionIK)
        print("Fetch Robot fk, ik connected!")

        self.head_traj_client = actionlib.SimpleActionClient('head_controller/follow_joint_trajectory',
                                                             FollowJointTrajectoryAction)
        self.head_traj_client.wait_for_server()

        self.torso_traj_client = actionlib.SimpleActionClient('torso_controller/follow_joint_trajectory',
                                                              FollowJointTrajectoryAction)
        self.torso_traj_client.wait_for_server()

        rospy.Subscriber("/move_joint_by_small_pose_change", Float32MultiArray, self.move_joint_by_small_pose_change)
        rospy.Subscriber("/move_joint_by_small_joint_change", Float32MultiArray, self.move_joint_by_small_joint_change)
        rospy.Subscriber("/process_get_ready", String, self.process_get_ready)
        rospy.Subscriber("/process_post_action", String, self.process_post_action)
        self.pose_msg_publisher = rospy.Publisher('/pose_msg', Float32MultiArray, queue_size=2)
        self.ready_for_action_pub = rospy.Publisher('/ready_for_action', String, queue_size=1)
        self.pose_msg_thread = threading.Thread(target=self.publish_pose_messages, args=())
        self.pose_msg_thread.start()

        moveit_commander.roscpp_initialize(sys.argv)

        ## Instantiate a RobotCommander object
        robot = moveit_commander.RobotCommander()

        ## Instantiate a PlanningSceneInterface object
        scene = moveit_commander.PlanningSceneInterface()

        ## Instantiate a MoveGroupCommander object
        self.arm_group = moveit_commander.MoveGroupCommander("arm", wait_for_servers=25.0)
        self.arm_group.set_max_velocity_scaling_factor(0.7)
        self.arm_group.set_planner_id("BITstar")

        self.home_joints = [1.3205522228271485, 1.399532370159912, -0.19974325208511354, 1.719844644293213,
                            0.0004958728740930562, 1.4, 0]
        self.stand_by_joints = [-1.553744442364502, -0.48266214888305664, 2.7431989500061036, -1.1897336790710449,
                                1.4102242416183473, -1.5474664614929199, 0.821144968328247]
        self.fridge_stand_by_joints = [0.29470240364379885, 1.1211147828796386, -1.2186898401245116, 1.876310651525879,
                                       1.202369851473999, -1.8404571460021972, 1.7227418721292114]

        self.moped_stand_by_joints = [0.5458916355163574, 0.2785754724243164, -1.6547240427001952, 2.0803297211975096,
                                      -1.2496984773834228, -1.7096855090393066, 1.6886107028100585]

        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
        self.move_base_publisher = rospy.Publisher('/base_controller/command', Twist, queue_size=2)
        self.link_poses = []

    def solve_ik(self, position, rot):
        # Initialize the service proxy
        robot_state = self.get_robot_state()

        try:
            req = GetPositionIKRequest()
            req.ik_request.group_name = "arm"
            req.ik_request.robot_state = robot_state
            req.ik_request.ik_link_name = "gripper_link"
            req.ik_request.pose_stamped.header.frame_id = "base_link"
            req.ik_request.pose_stamped.pose.position.x = position[0]
            req.ik_request.pose_stamped.pose.position.y = position[1]
            req.ik_request.pose_stamped.pose.position.z = position[2]
            req.ik_request.pose_stamped.pose.orientation.x = rot[0]
            req.ik_request.pose_stamped.pose.orientation.y = rot[1]
            req.ik_request.pose_stamped.pose.orientation.z = rot[2]
            req.ik_request.pose_stamped.pose.orientation.w = rot[3]
            req.ik_request.avoid_collisions = True
            rospy.wait_for_service('compute_ik')
            resp = self.compute_ik(req)
            # print(resp.solution.joint_state.name)
            return resp.solution.joint_state.name, resp.solution.joint_state.position
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def solve_fk(self, robot_state):
        try:
            req = GetPositionFKRequest()
            # req.fk_link_names = ["gripper_link"]
            req.fk_link_names = ["gripper_link", "elbow_flex_link", "forearm_roll_link",
                                 "shoulder_lift_link", "shoulder_pan_link", "upperarm_roll_link", "wrist_flex_link"]
            req.robot_state = robot_state
            req.header.frame_id = "base_link"
            rospy.wait_for_service('compute_fk')
            resp = self.compute_fk(req)
            pos, rot = resp.pose_stamped[0].pose.position, resp.pose_stamped[0].pose.orientation
            pos = np.array([pos.x, pos.y, pos.z])
            rot = np.array([rot.x, rot.y, rot.z, rot.w])

            link_poses = []
            for item in resp.pose_stamped:
                position_ = np.array([item.pose.position.x, item.pose.position.y, item.pose.position.z])
                link_poses.append(position_)

            return pos, rot, link_poses
        except:
            print("compute_fk Service call failed, restarting")
            rospy.wait_for_service('compute_fk')
            self.compute_fk = rospy.ServiceProxy('compute_fk', GetPositionFK)

    def plan_to_move_joints(self, joints):
        self.arm_group.set_joint_value_target(joints)
        self.arm_group.go(wait=True)
        self.arm_group.stop()

    def send_goal(self, goal_position, goal_orientation):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        # Set the goal position
        goal.target_pose.pose.position.x = goal_position[0]
        goal.target_pose.pose.position.y = goal_position[1]
        goal.target_pose.pose.position.z = goal_position[2]

        # Set the goal orientation
        goal.target_pose.pose.orientation.x = goal_orientation[0]
        goal.target_pose.pose.orientation.y = goal_orientation[1]
        goal.target_pose.pose.orientation.z = goal_orientation[2]
        goal.target_pose.pose.orientation.w = goal_orientation[3]

        # Send the goal to the move_base action server
        self.client.send_goal(goal)

    def cancel_goal(self):
        self.client.cancel_goal()

    def wait_for_result(self):
        self.client.wait_for_result()

    def get_result(self):
        return self.client.get_result()

    def check_safe_joint_values(self, positions):
        state_change = np.array(positions[-9:-2]) - np.array(self.joint_state)
        # print("joint state change", state_change)
        if np.max(np.abs(state_change)) > 1.0:
            print("state change too large, skip it", state_change)
            return False

        robot_state = RobotState()
        robot_state.joint_state.header.frame_id = "base_link"

        for i in range(len(self.whole_body_joint_names)):
            robot_state.joint_state.name.append(self.whole_body_joint_names[i])
            if self.whole_body_joint_names[i] in self.arm_joints_names:
                joint_index = self.arm_joints_names.index(self.whole_body_joint_names[i])
                robot_state.joint_state.position.append(positions[joint_index])
            else:
                robot_state.joint_state.position.append(self.whole_body_joint_state[i])

        _, _, link_poses_next = self.solve_fk(robot_state)
        link_poses_next = np.array(link_poses_next)  # n x 3
        link_poses = np.array(self.link_poses)  # n x 3
        diff = ((link_poses_next - link_poses) ** 2).sum(axis=1) ** 0.5
        # print("diff", diff)
        # if np.max(diff) > 0.2:
        if np.max(diff) > 0.2:
            print("link pose change too large, skip it", diff)
            print("original link poses", self.link_poses)
            print("next link poses", link_poses_next)
            return False

        self.link_poses = link_poses_next
        return True

    def move_arm(self, pos, rot, trigger, f):

        if not trigger:
            gripper_command = np.array([0.05, 0.05])
        else:
            gripper_command = np.array([-0.01, 0.01])

        joint_names, positions = robot.solve_ik(pos, rot)
        positions = positions[:-2] + tuple(gripper_command)
        if len(positions) < 11:
            print("no solution found")
            return

        state_change = np.array(positions[-9:-2]) - np.array(self.joint_state)
        # print("joint state change", state_change)
        if np.max(np.abs(state_change)) > 1.0:
            print("state change too large, skip it", state_change)
            return

        robot_state = RobotState()
        robot_state.joint_state.header.frame_id = "base_link"

        for i in range(len(self.whole_body_joint_names)):
            robot_state.joint_state.name.append(self.whole_body_joint_names[i])
            if self.whole_body_joint_names[i] in joint_names:
                joint_index = joint_names.index(self.whole_body_joint_names[i])
                robot_state.joint_state.position.append(positions[joint_index])
            else:
                robot_state.joint_state.position.append(self.whole_body_joint_state[i])

        _, _, link_poses_next = self.solve_fk(robot_state)
        link_poses_next = np.array(link_poses_next)  # n x 3
        link_poses = np.array(self.link_poses)  # n x 3
        diff = ((link_poses_next - link_poses) ** 2).sum(axis=1) ** 0.5
        # print("diff", diff)
        if np.max(diff) > 0.2:
            print("link pose change too large, skip it", diff)
            return

        self.link_poses = link_poses_next
        # if self.check_safe_joint_values(positions):
        robot.move_joint(positions[-9:], f=f, speed=prepare_conf[self.task]["speed"])

    def move_torso_head(self, torso_position, head_position, duration=1.0):
        head_joints = ["head_pan_joint", "head_tilt_joint"]
        torso_joints = ["torso_lift_joint"]

        head_goal = FollowJointTrajectoryGoal()
        head_goal.trajectory = JointTrajectory()
        head_goal.trajectory.joint_names = head_joints

        torso_goal = FollowJointTrajectoryGoal()
        torso_goal.trajectory = JointTrajectory()
        torso_goal.trajectory.joint_names = torso_joints

        head_point = JointTrajectoryPoint()
        head_point.positions = tuple(head_position)
        head_point.time_from_start = rospy.Duration(duration)
        head_goal.trajectory.points.append(head_point)

        torso_point = JointTrajectoryPoint()
        torso_point.positions = (torso_position,)
        torso_point.time_from_start = rospy.Duration(duration)
        torso_goal.trajectory.points.append(torso_point)

        self.head_traj_client.send_goal(head_goal)
        self.torso_traj_client.send_goal(torso_goal)

        # self.head_traj_client.wait_for_result(rospy.Duration(duration))
        # self.torso_traj_client.wait_for_result(rospy.Duration(duration))

    def process_get_ready(self, task):
        self.get_ready(task.data)

    def process_post_action(self, task):
        # if task.data not in ["open_fridge"]:
        self.move_back()
        if task.data in ["pickplace_pen"]:
            self.move_back(duration=5)
        robot.plan_to_move_joints(robot.home_joints)

        if task.data in ["open_fridge"]:
            self.move_back(v=0.1, duration=3.6)

        # rotate 360
        # twist = Twist()
        # twist.angular.z = -0.6
        # f = 10
        # for i in range(int(f * 10)):
        #     time.sleep(1.0 / f)
        #     self.move_base_publisher.publish(twist)

    def move_joint_by_small_pose_change(self, pose):
        f = 15
        pose = pose.data

        pos = pose[:3]
        rot = pose[3:7]
        touchpad, trigger, grip, menu = pose[7:11]
        direction = np.array(pose[11:13])

        # self.move_arm(pos, rot, trigger, f)
        curr_torso_pos = self.whole_body_joint_state[2]
        curr_head_pos = self.whole_body_joint_state[4:6]

        if menu and touchpad:
            print("lift torso")

            next_torso_pos = curr_torso_pos + 0.08 * direction[1]
            self.move_torso_head(next_torso_pos, curr_head_pos, duration=1.0 / f)

        if touchpad and not grip and not menu:
            print("move head")

            next_head_pos = curr_head_pos + -0.2 * direction
            self.move_torso_head(curr_torso_pos, next_head_pos, duration=1.0 / f)

        if grip and not menu and touchpad:
            print("move base")
            twist = Twist()
            twist.linear.x = direction[1] * 0.1
            twist.angular.z = direction[0] * -0.6
            self.move_base_publisher.publish(twist)

    def move_joint_by_small_joint_change(self, action):
        f = 15
        self.step_count += 1
        action = action.data
        positions = action[:9]

        if self.check_safe_joint_values(positions):
            robot.move_joint(positions, f=f, speed=prepare_conf[self.task]["speed"])

        touchpad, trigger, grip, menu = np.round(action[9:13])
        direction = np.array(action[13:15])
        print(self.step_count, "touchpad, trigger, grip, menu", touchpad, trigger, grip, menu, 'direction', direction)

        curr_torso_pos = self.whole_body_joint_state[2]
        curr_head_pos = self.whole_body_joint_state[4:6]

        if menu and touchpad:
            print("lift torso")

            next_torso_pos = curr_torso_pos + 0.08 * direction[1]
            self.move_torso_head(next_torso_pos, curr_head_pos, duration=1.0 / f)

        if touchpad and not grip and not menu:
            print("move head")

            next_head_pos = curr_head_pos + -0.2 * direction
            self.move_torso_head(curr_torso_pos, next_head_pos, duration=1.0 / f)

        if grip and not menu and touchpad:
            print("move base")
            twist = Twist()
            twist.linear.x = direction[1] * 0.1
            twist.angular.z = direction[0] * -0.6
            self.move_base_publisher.publish(twist)

    def move_back(self, duration=2.0, v=-0.1):
        twist = Twist()
        twist.linear.x = v
        f = 10
        for i in range(int(f * duration)):
            time.sleep(1.0 / f)
            self.move_base_publisher.publish(twist)

    def publish_pose_messages(self):
        while not rospy.is_shutdown():
            try:
                pos, rot, link_poses = self.solve_fk(self.get_robot_state())
                self.link_poses = link_poses
                msg = Float32MultiArray()
                msg.data = np.concatenate([pos, rot]).tolist()
                self.pose_msg_publisher.publish(msg)
                time.sleep(1 / 2.0)

                # print("publish_pose_messages", msg.data)
            except Exception as e:
                continue

    def get_ready(self, task_text):
        print("get ready for task: ", task_text)
        self.step_count = 0
        task = task_text.split("|")[0]
        test_time = task_text.split("|")[1]
        self.task = task

        # self.move_back(duration=3.0, v=-0.1)
        # rotate 360
        twist = Twist()
        twist.angular.z = -0.6
        f = 10
        for i in range(int(f * 6)):
            time.sleep(1.0 / f)
            self.move_base_publisher.publish(twist)

        target_position = prepare_conf[task]['position']  # x, y, z coordinates
        target_orientation = prepare_conf[task]['orientation']  # quaternion

        # if test_time == "False":
        #     print("add noise to target position and orientation")
        #     # add some noise to the target position
        #     target_position = np.array(target_position) + np.random.random(size=(3,)) * 0.1 - 0.05
        #     target_position[-1] = 0.0
        #
        #     # add some noise to the target orientation
        #     target_orientation = np.array(target_orientation) + np.random.random(size=(4,)) * 0.1 - 0.05
        #     target_orientation[:2] = 0.0
        #     target_orientation = target_orientation / np.linalg.norm(target_orientation)

        # if task in ["close_microwave"]:
        #     target_position_ = copy.copy(target_position)
        #     target_position_[1] += 0.5
        #     self.send_goal(target_position_, target_orientation)
        #     self.wait_for_result()

        # if task in ["open_fridge"]:
        #     target_position_ = copy.copy(target_position)
        #     target_position_[1] -= 0.5
        #     # target_position_[0] += 0.0
        #     self.send_goal(target_position_, target_orientation)
        #     self.wait_for_result()
        
        # if task in ["open_fridge", "grasp_food_from_microwave", "open_microwave"]:
            # target_position_ = copy.copy(target_position)
            # target_position_[1] -= 0.5
            # # target_position_[0] += 0.3
            # self.send_goal(target_position_, target_orientation)
            # self.wait_for_result()
        if task in ["switchon_microwave", "placeon_food_counter",
                    "grasp_food_from_table", "placein_microwave", "close_microwave"]:
            target_position_ = copy.copy(target_position)
            target_position_[0] += 0.3
            # target_position_[0] += 0.3
            self.send_goal(target_position_, target_orientation)
            self.wait_for_result()

        self.send_goal(target_position, target_orientation)
        self.wait_for_result()

        if prepare_conf[task]['back_before_arm']:
            self.move_back()

        if 'torso' in prepare_conf[task]:
            self.move_torso_head(prepare_conf[task]['torso'], prepare_conf[task]['head'], duration=1.0)
        self.plan_to_move_joints(prepare_conf[task]['arm_joints'])

        if prepare_conf[task]['back_before_arm']:
            if 'back_and_forth_duration' in prepare_conf[task]:
                self.move_back(duration=prepare_conf[task]['back_and_forth_duration'][1], v=0.1)
            else:
                self.move_back(v=0.1)

        self.ready_for_action_pub.publish("ready_for_action")


if __name__ == '__main__':
    robot = FetchController()

    # robot.get_ready("open_fridge")
    # robot.get_ready("grasp_food")
    # robot.get_ready("placeon_food")
    # robot.get_ready("close_fridge")
    # robot.get_ready("open_microwave")
    # robot.get_ready("grasp_food_from_table")
    # robot.get_ready("placein_microwave")
    # robot.get_ready("close_microwave")
    # robot.get_ready("switchon_microwave")
    # robot.get_ready("open_microwave")
    # robot.get_ready("grasp_food_from_microwave")
    # robot.get_ready("placeon_food_dinning_table")

    # goal_position = [-4.669160, 3.6565535, 0.0]  # x, y, z coordinates
    # goal_orientation = [0.0, 0.0, 0.924788, 0.38313]
    # robot.send_goal(goal_position, goal_orientation)
    # robot.wait_for_result()
    #
    # # robot.move_back()
    # robot.move_torso_head(0.37, [0, 0.47738], duration=1.0)
    # robot.plan_to_move_joints(robot.fridge_stand_by_joints)

    # robot.move_torso_head(0.37, [0, 0.47738], duration=1.0)
    # robot.plan_to_move_joints(robot.moped_stand_by_joints)
    # robot.move_torso_head(0.2, [0, 0.87738], duration=1.0)

    # subscribe to the topic /pose_msg and get one message
    # msg = np.array(rospy.wait_for_message('/pose_msg', Float32MultiArray).data)
    # pos, rot = msg[:3], msg[3:]
    # print("pos: ", pos, "rot: ", rot)
    #
    # pub = rospy.Publisher('/move_joint_by_small_pose_change', Float32MultiArray, queue_size=10)
    # time_start = time.time()
    # for i in range(10):
    #     pos[2] -= 0.02
    #     pose = np.concatenate([pos, rot, np.zeros((2,))])
    #     message = Float32MultiArray()
    #     message.data = list(pose)
    #     pub.publish(message)
    #     time.sleep(1 / 10.0)
    #     print("i: ", i, "pos: ", pos, "rot: ", rot)
    # print("time: ", time.time() - time_start)
