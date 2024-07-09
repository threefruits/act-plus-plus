import openvr
import time

def get_tracked_devices():
    tracked_devices = []
    for device_index in range(openvr.k_unMaxTrackedDeviceCount):
        device_class = openvr.VRSystem().getTrackedDeviceClass(device_index)
        if device_class != openvr.TrackedDeviceClass_Invalid:
            tracked_devices.append((device_index, device_class))
    return tracked_devices

def get_controller_pose(controller_id):
    poses = openvr.VRCompositor().waitGetPoses()
    pose = poses[controller_id]
    if pose.bPoseIsValid:
        return pose.mDeviceToAbsoluteTracking
    else:
        return None

def print_pose_matrix(matrix):
    for row in matrix:
        print(' '.join(f'{val: .4f}' for val in row))

def main():
    try:
        # Initialize OpenVR
        openvr.init(openvr.VRApplication_Other)
    except openvr.OpenVRError as e:
        print(f"OpenVR initialization error: {e}")
        return

    try:
        # List all tracked devices
        tracked_devices = get_tracked_devices()
        print("Tracked devices and their classes:")
        for device_index, device_class in tracked_devices:
            print(f"Device Index: {device_index}, Device Class: {device_class}")

        # Get the ID of the first controller
        controller_id = None
        for device_index, device_class in tracked_devices:
            if device_class == openvr.TrackedDeviceClass_Controller:
                controller_id = device_index
                break

        if controller_id is None:
            print("No controller found.")
            return

        print(f"Using controller with ID: {controller_id}")

        # Get the pose of the controller
        while True:
            pose_matrix = get_controller_pose(controller_id)
            if pose_matrix:
                print("Controller Pose Matrix:")
                print_pose_matrix(pose_matrix)
            else:
                print("Controller pose is not valid.")
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        # Shutdown OpenVR
        openvr.shutdown()

if __name__ == '__main__':
    main()
