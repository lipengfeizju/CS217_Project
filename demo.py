import numpy as np 
import subprocess
import os 
from scipy.spatial.transform import Rotation as R
from utils.visualize import visualize_point_cloud

def load_pcl_open3d(pcl_path, point_color =[0.1, 0.5, 0.2]):
    '''
    Load PCL data, return point cloud array and color array
    '''
    assert os.path.exists(pcl_path), "File not exists! " + pcl_path + "\n"
    point_set = np.loadtxt(pcl_path, delimiter=',').astype(np.float32)
    point_set = point_set[:,:3]
    num_point = point_set.shape[0]
    color_array  = np.repeat(point_color, num_point).reshape([3, num_point]).astype(np.float64).T

    return point_set, color_array

def random_transform(sigma_r = 0.5, sigma_t = 0.5):
    from scipy.spatial.transform import Rotation as R
    r1 = np.random.uniform(-1,1,3)
    r1 = r1/np.sqrt(np.sum(r1**2))*sigma_r
    rot_mat = R.from_rotvec(r1).as_matrix()

    t1 = np.random.uniform(-1,1,3)
    t1 = t1/np.sqrt(np.sum(t1**2))*sigma_t

    transform = np.zeros([3,4])
    transform[:3, :3] = rot_mat
    transform[:3, 3] = t1

    return transform

def save_transformed_data(point_original, transform, file_path):
    '''
    transform the original point cloud
    '''

    assert point_original.shape[1] == 3
    num_point = point_original.shape[0]
    point_set = np.ones((num_point, 4))
    point_set[:,:3] = point_original
    point_set = transform @ point_set.T
    np.savetxt(file_path, point_set.T, delimiter=",", fmt="%.6f")

def main():
    base_dir = "/home/anny/cuda-workspace/icp_project/icp/data/samples/"
    pcl_original = "airplane_0002.txt"
    pcl_transformed = "airplane_0002_rotated.txt"
    pcl_recoverd = "recovered.txt"

    subprocess.run(["./build/registration", base_dir , pcl_original, pcl_transformed])


    point_set, color_array = load_pcl_open3d(base_dir + pcl_original, point_color=[0.1, 0.5, 0.2])

    point_set2, color_array2 = load_pcl_open3d(base_dir + pcl_transformed, point_color=[0.5, 0.1, 0.2])

    point_set3, color_array3 = load_pcl_open3d(base_dir + pcl_recoverd, point_color=[0.2, 0.1, 0.5])

    point_set = np.concatenate((point_set, point_set2, point_set3), axis=0)
    color_array = np.concatenate((color_array, color_array2, color_array3), axis=0)
    visualize_point_cloud(point_set, color_array)

def save_test():
    base_dir = "/home/anny/cuda-workspace/icp_project/icp/data/samples/"
    pcl_original = "airplane_0002.txt"

    point_set, color_array = load_pcl_open3d(base_dir + pcl_original, point_color=[0.1, 0.5, 0.2])

    transform = random_transform()
    save_transformed_data(point_set, transform, base_dir + "airplane_0002_rotated.txt")

if __name__ == "__main__":
    main()
    # save_test()
