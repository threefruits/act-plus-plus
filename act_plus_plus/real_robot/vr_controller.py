import openvr
import sys
import time
import numpy as np
import open3d as o3d
import simpleaudio as sa
from scipy.spatial.transform import Rotation as R


class VRController:
    def __init__(self):
        openvr.init(openvr.VRApplication_Scene)
        self.vr_system = openvr.VRSystem()

    def get_controller_trigger_state(self, controller_index):
        _, button_states = self.vr_system.getControllerState(controller_index)
        # trigger_state = (button_states.ulButtonPressed & openvr.k_eControllerAxis_Trigger) != 0
        trigger_state = button_states.ulButtonPressed != 0
        return trigger_state

    def get_controller_position(self):
        poses = self.vr_system.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0, openvr.k_unMaxTrackedDeviceCount)

        for i in range(openvr.k_unMaxTrackedDeviceCount):
            if self.vr_system.getTrackedDeviceClass(i) == openvr.TrackedDeviceClass_Controller:
                pose = poses[i]
                if pose.bPoseIsValid:
                    np_pose = self.convert_to_numpy(pose.mDeviceToAbsoluteTracking)
                    return i, np_pose

        return -1, None

    @staticmethod
    def convert_to_numpy(mat):
        np_mat = np.zeros((3, 4), dtype=np.float32)
        for i in range(3):
            for j in range(4):
                np_mat[i, j] = mat[i][j]
        return np_mat

    def create_frame(self, position, rotation_matrix):
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        rotation_matrix = rotation_matrix.astype(np.float64)
        frame.rotate(rotation_matrix, center=(0, 0, 0))
        frame.translate(position)
        return frame

    @staticmethod
    def generate_sine_wave(frequency, duration, sample_rate=44100):
        t = np.linspace(0, duration, int(duration * sample_rate), False)
        sine_wave = np.sin(frequency * 2 * np.pi * t)
        audio_data = (sine_wave * 32767 / np.max(np.abs(sine_wave))).astype(np.int16)
        return audio_data

    @staticmethod
    def play_audio(audio_data, sample_rate=44100):
        play_obj = sa.play_buffer(audio_data, 1, 2, sample_rate)
        play_obj.wait_done()

    def beep_short(self):
        beep_short = self.generate_sine_wave(1000, 0.1)
        self.play_audio(beep_short)

    def beep_long(self):
        beep_long = self.generate_sine_wave(1000, 0.5)
        self.play_audio(beep_long)

    def sync(self, target_position=None, target_rotation=None):
        controller_index, np_pose = self.get_controller_position()
        target_position = np_pose[:, 3]

        if target_rotation is None:
            target_rotation = np_pose[:, :-1]
            print("*******************  warning: using current rotation as target rotation")
        else:
            # convert quoterion to rotation matrix
            if target_rotation.shape != (3, 3):
                target_rotation = R.from_quat(target_rotation).as_matrix()

        # Visualization setup
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1280, height=720)
        # vis.get_render_option().load_from_json("default_render_option.json")
        vis.get_render_option().point_size = 5.0
        vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame())
        vis.add_geometry(self.create_frame(target_position, target_rotation))

        try:
            controller_frame = self.create_frame(np.zeros(3), np.identity(3))
            vis.add_geometry(controller_frame)
            sync_progress = 0
            while True:
                controller_index, np_pose = self.get_controller_position()
                if controller_index != -1:
                    vis.remove_geometry(controller_frame)

                    position = np_pose[:, 3]
                    rotation_matrix = np_pose[:, :-1]
                    controller_frame = self.create_frame(position, rotation_matrix)
                    vis.add_geometry(controller_frame)

                    trigger_state = self.get_controller_trigger_state(controller_index)

                    # calc l1 distance bewteen target and current rotation
                    l1_distance = np.sum(np.abs(target_rotation - rotation_matrix))
                    print("position:", position, "Trigger state:", trigger_state, "l1 distance:", l1_distance)
                    if l1_distance < 0.36:
                        sync_progress += 1
                    else:
                        sync_progress = 0

                    if sync_progress == 1 or sync_progress == 50:
                        self.beep_short()

                    if sync_progress > 100:
                        print("Sync completed!")
                        self.beep_long()
                        break

                    # _, button_states = vr_controller.vr_system.getControllerState(controller_index)
                    # print("button_states.ulButtonPressed", button_states.ulButtonPressed)


                vis.poll_events()
                vis.update_renderer()

                time.sleep(1.0 / 60)

        finally:
            # openvr.shutdown()
            vis.destroy_window()
        return position, rotation_matrix


if __name__ == "__main__":
    vr_controller = VRController()

    _, np_pose = vr_controller.get_controller_position()
    target_position = np_pose[:, 3]
    target_rotation = np_pose[:, :-1]
    vr_controller.sync(target_position, target_rotation)