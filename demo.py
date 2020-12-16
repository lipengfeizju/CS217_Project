import numpy as np 
import subprocess
import os 
from scipy.spatial.transform import Rotation as R
from utils.visualize import visualize_point_cloud, random_transform, save_transformed_data, load_pcl_open3d

def main():
    base_dir = "/home/anny/cuda-workspace/icp_project/icp/data/samples/"
    pcl_original = "airplane_0001.txt"
    pcl_transformed = "airplane_0001_rotated.txt"
    pcl_recoverd = "recovered.txt"

    subprocess.run(["./build/registration", base_dir , pcl_original, pcl_transformed])


    point_set, color_array = load_pcl_open3d(base_dir + pcl_original, point_color=[0.1, 0.5, 0.2])

    point_set2, color_array2 = load_pcl_open3d(base_dir + pcl_transformed, point_color=[0.5, 0.1, 0.2])

    point_set3, color_array3 = load_pcl_open3d(base_dir + pcl_recoverd, point_color=[0.2, 0.1, 0.5])

    point_set = np.concatenate((point_set, point_set2, point_set3), axis=0)
    color_array = np.concatenate((color_array, color_array2, color_array3), axis=0)
    visualize_point_cloud(point_set, color_array)

def random_transform_data():
    base_dir = "/home/anny/cuda-workspace/icp_project/icp/data/samples/"
    pcl_original = "airplane_0001.txt"

    point_set, color_array = load_pcl_open3d(base_dir + pcl_original, point_color=[0.1, 0.5, 0.2])

    transform = random_transform()
    save_transformed_data(point_set, transform, base_dir + "airplane_0001_rotated.txt")

if __name__ == "__main__":
    main()
    # random_transform_data()
