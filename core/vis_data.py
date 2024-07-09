import cv2
import numpy as np
import open3d as o3d
import os
import re

def set_camera_view(vis, front, lookat, up, zoom):
    """
    Set the camera view.
    Arguments:
    - vis: The visualizer object.
    - front: The front vector of the camera.
    - lookat: The point the camera is looking at.
    - up: The up vector of the camera.
    - zoom: Zoom level of the camera.
    """
    ctr = vis.get_view_control()
    ctr.set_front(front)
    ctr.set_lookat(lookat)
    ctr.set_up(up)
    ctr.set_zoom(zoom)

def generate_point_cloud(rgb_image, depth_image, intrinsics):
    print(depth_image.max())
    """Generate a point cloud object from an RGB and a corresponding depth image using provided intrinsics."""
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb_image),
        o3d.geometry.Image(depth_image),
        depth_scale=1000.0,
        depth_trunc=3000.0,
        convert_rgb_to_intensity=False
    )
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsics
    )
    
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return pcd


def extract_index(filename):
    """Extract the numeric index from filenames like 'rgb_0.png' or 'depth_0.png'."""
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else None

def point_clouds_to_images(input_folder, output_folder, intrinsics=None):
    """Read RGB and Depth images from folder, generate point clouds using provided intrinsics and save them as images."""
    if intrinsics is None:
        intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, 554.254691191187, 554.254691191187, 320.5, 240.5)

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Setup visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # set_camera_view(vis, [0, 0, -2], [0, 0, 0], [0, -1, 0], 0.9)

    # List and sort files to align RGB and Depth pairs
    files = os.listdir(input_folder)
    rgb_files = sorted([f for f in files if 'rgb' in f], key=extract_index)
    depth_files = sorted([f for f in files if 'depth' in f], key=extract_index)

    # Process each pair of RGB and Depth images
    for idx, (rgb_file, depth_file) in enumerate(zip(rgb_files, depth_files)):
        rgb_img = cv2.imread(os.path.join(input_folder, rgb_file))
        depth_img = cv2.imread(os.path.join(input_folder, depth_file), cv2.IMREAD_UNCHANGED)
        
        pcd = generate_point_cloud(rgb_img, depth_img, intrinsics)
        
        # Visualize the point cloud
        vis.clear_geometries()
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        # Capture the current view and save to image
        image_path = os.path.join(output_folder, f'frame_{idx:04d}.png')
        vis.capture_screen_image(image_path, do_render=True)

    # Cleanup
    vis.destroy_window()

# Usage
custom_intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, 554.254691191187, 554.254691191187, 320.5, 240.5)
point_clouds_to_images('./env/expert_demo/open_fridge/episode_1', './point_cloud_images', intrinsics=custom_intrinsics)
