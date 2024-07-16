#! /usr/bin/env python3

import numpy as np
import os
import rospkg
import rospy
import transformations as T

from geometry_msgs.msg import Pose, Vector3
from std_msgs.msg import Float32MultiArray, Bool, String
from sensor_msgs.msg import JointState
from timeit import default_timer as timer
from urdf_parser_py.urdf import URDF
import PyKDL
# from kdl_parser import kdl_tree_from_urdf_model
import yaml


from urdf_parser_py.urdf import URDF

def euler_to_quat(r, p, y):
    sr, sp, sy = np.sin(r/2.0), np.sin(p/2.0), np.sin(y/2.0)
    cr, cp, cy = np.cos(r/2.0), np.cos(p/2.0), np.cos(y/2.0)
    return [sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy,
            cr*cp*cy + sr*sp*sy]

def urdf_pose_to_kdl_frame(pose):
    pos = [0., 0., 0.]
    rot = [0., 0., 0.]
    if pose is not None:
        if pose.position is not None:
            pos = pose.position
        if pose.rotation is not None:
            rot = pose.rotation
    return PyKDL.Frame(PyKDL.Rotation.Quaternion(*euler_to_quat(*rot)),
                     PyKDL.Vector(*pos))

def urdf_joint_to_kdl_joint(jnt):
    origin_frame = urdf_pose_to_kdl_frame(jnt.origin)
    if jnt.joint_type == 'fixed':
        return PyKDL.Joint(jnt.name, getattr(PyKDL.Joint, 'None'))
    axis = PyKDL.Vector(*[float(s) for s in jnt.axis])
    if jnt.joint_type == 'revolute':
        return PyKDL.Joint(jnt.name, origin_frame.p,
                         origin_frame.M * axis, PyKDL.Joint.RotAxis)
    if jnt.joint_type == 'continuous':
        return PyKDL.Joint(jnt.name, origin_frame.p,
                         origin_frame.M * axis, PyKDL.Joint.RotAxis)
    if jnt.joint_type == 'prismatic':
        return PyKDL.Joint(jnt.name, origin_frame.p,
                         origin_frame.M * axis, PyKDL.Joint.TransAxis)
    print ("Unknown joint type: %s." % jnt.joint_type)
    return PyKDL.Joint(jnt.name, getattr(PyKDL.Joint, 'None'))

def urdf_inertial_to_kdl_rbi(i):
    origin = urdf_pose_to_kdl_frame(i.origin)
    rbi = PyKDL.RigidBodyInertia(i.mass, origin.p,
                               PyKDL.RotationalInertia(i.inertia.ixx,
                                                     i.inertia.iyy,
                                                     i.inertia.izz,
                                                     i.inertia.ixy,
                                                     i.inertia.ixz,
                                                     i.inertia.iyz))
    return origin.M * rbi

def kdl_tree_from_urdf_model(urdf):
    root = urdf.get_root()
    tree = PyKDL.Tree(root)
    def add_children_to_tree(parent):
        if parent in urdf.child_map:
            for joint, child_name in urdf.child_map[parent]:
                for lidx, link in enumerate(urdf.links):
                    if child_name == link.name:
                        child = urdf.links[lidx]
                        if child.inertial is not None:
                            kdl_inert = urdf_inertial_to_kdl_rbi(child.inertial)
                        else:
                            kdl_inert = PyKDL.RigidBodyInertia()
                        for jidx, jnt in enumerate(urdf.joints):
                            if jnt.name == joint:
                                kdl_jnt = urdf_joint_to_kdl_joint(urdf.joints[jidx])
                                kdl_origin = urdf_pose_to_kdl_frame(urdf.joints[jidx].origin)
                                kdl_sgm = PyKDL.Segment(child_name, kdl_jnt,
                                                      kdl_origin, kdl_inert)
                                tree.addSegment(kdl_sgm, parent)
                                add_children_to_tree(child_name)
    add_children_to_tree(root)
    return tree

class Robot():
    def __init__(self, setting_path = None):
        path_to_src = rospkg.RosPack().get_path('relaxed_ik_ros1') + '/relaxed_ik_core'
        setting_file_path = path_to_src + '/configs/settings.yaml'
        if setting_path != '':
           setting_file_path = setting_path
        os.chdir(path_to_src)

        # Load the infomation
        setting_file = open(setting_file_path, 'r')
        settings = yaml.load(setting_file, Loader=yaml.FullLoader)

        urdf_name = settings["urdf"]
        
        self.robot = URDF.from_xml_file(path_to_src + "/configs/urdfs/" + urdf_name)
        self.kdl_tree = kdl_tree_from_urdf_model(self.robot)
        
        # all non-fixed joint         
        self.joint_lower_limits = []
        self.joint_upper_limits = []
        self.joint_vel_limits = []
        self.all_joint_names = []
        for j in self.robot.joints:
            if j.type != 'fixed':
                self.joint_lower_limits.append(j.limit.lower)
                self.joint_upper_limits.append(j.limit.upper)
                self.joint_vel_limits.append(j.limit.velocity)
                self.all_joint_names.append(j.name)

        # joints solved by relaxed ik
        self.articulated_joint_names = []
        assert len(settings['base_links']) == len(settings['ee_links']) 
        self.num_chain = len(settings['base_links'])

        for i in range(self.num_chain):
            arm_chain = self.kdl_tree.getChain( settings['base_links'][i],
                                                settings['ee_links'][i])
            for j in range(arm_chain.getNrOfSegments()):
                joint = arm_chain.getSegment(j).getJoint()
                # 8 is fixed joint   
                if joint.getType() != 8:
                    self.articulated_joint_names.append(joint.getName())

        if 'starting_config' in settings:
            assert len(self.articulated_joint_names) == len(settings['starting_config']), \
                        "Number of joints parsed from urdf should be the same with the starting config in the setting file."

        self.arm_chains = []
        self.fk_p_kdls = []
        self.fk_v_kdls = []
        self.ik_p_kdls = []
        self.ik_v_kdls = []
        self.num_jnts = []
        for i in range(self.num_chain):
            arm_chain = self.kdl_tree.getChain( settings['base_links'][i],
                                                settings['ee_links'][i])
            self.arm_chains.append(arm_chain)
            self.fk_p_kdls.append(PyKDL.ChainFkSolverPos_recursive(arm_chain))
            self.fk_v_kdls.append(PyKDL.ChainFkSolverVel_recursive(arm_chain))
            self.ik_v_kdls.append(PyKDL.ChainIkSolverVel_pinv(arm_chain))
            self.ik_p_kdls.append(PyKDL.ChainIkSolverPos_NR(arm_chain, 
                                                            self.fk_p_kdls[i],
                                                            self.ik_v_kdls[i]))
            self.num_jnts.append(arm_chain.getNrOfJoints())

    def fk_single_chain(self, fk_p_kdl, joint_angles, num_jnts):
        assert len(joint_angles) == num_jnts, "length of input: {}, number of joints: {}".format(len(joint_angles), num_jnts)
    
        kdl_array = PyKDL.JntArray(num_jnts)
        for idx in range(num_jnts):
            kdl_array[idx] = joint_angles[idx]

        end_frame = PyKDL.Frame()
        fk_p_kdl.JntToCart(kdl_array, end_frame)

        pos = end_frame.p
        rot = PyKDL.Rotation(end_frame.M)
        rot = rot.GetQuaternion()
        pose = Pose()
        pose.position.x = pos[0]
        pose.position.y = pos[1]
        pose.position.z = pos[2]
        pose.orientation.x = rot[0]
        pose.orientation.y = rot[1]
        pose.orientation.z = rot[2]
        pose.orientation.w = rot[3]
        return pose

    def fk(self, joint_angles):
        l = 0
        r = 0
        poses = []
        for i in range(self.num_chain):
            r += self.num_jnts[i]
            pose = self.fk_single_chain(self.fk_p_kdls[i], joint_angles[l:r], self.num_jnts[i])
            l = r
            poses.append(pose)
        
        return poses

    def get_joint_state_msg(self, joint_angles):
        js = JointState()
        js.header.stamp = rospy.Time.now()
        js.name = self._joint_names
        js.position = joint_angles
        return js