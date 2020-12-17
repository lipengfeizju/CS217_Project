'''
    Code Credit :  Pengfei Li
    Email: pli081@ucr.edu
'''
import open3d as o3d
import numpy as np 
import os 

def visualize_point_cloud(point_array, color_array = None):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=960)
    render_op = vis.get_render_option()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_array.astype(np.float64))
    if color_array is not None:
        assert color_array.shape == point_array.shape
        # pcd.paint_uniform_color(np.array(point_color).astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(color_array)

    vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()

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
    '''
    Generate random transformation
    '''
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
    transform the original point cloud and save it to file
    '''

    assert point_original.shape[1] == 3
    num_point = point_original.shape[0]
    point_set = np.ones((num_point, 4))
    point_set[:,:3] = point_original
    point_set = transform @ point_set.T
    np.savetxt(file_path, point_set.T, delimiter=",", fmt="%.6f")

##################  small test example   #########################   
def test():
    point_set = np.loadtxt("data/samples/airplane_0000.txt",delimiter=',').astype(np.float32)
    point_set = point_set[:,:3]

    num_point = point_set.shape[0]
    point_color =[0.1, 0.5, 0.2]
    color_array  = np.repeat(point_color, num_point).reshape([3, num_point]).astype(np.float64).T


    point_set2 = np.loadtxt("data/samples/airplane_0000_rotated.txt",delimiter=',').astype(np.float32)
    point_set2 = point_set2[:,:3]

    num_point = point_set2.shape[0]
    point_color =[0.5, 0.1, 0.2]
    color_array2  = np.repeat(point_color, num_point).reshape([3, num_point]).astype(np.float64).T


    point_set3 = np.loadtxt("data/samples/airplane_0000_recovered.txt",delimiter=',').astype(np.float32)
    point_set3 = point_set3[:,:3]

    num_point = point_set3.shape[0]
    point_color =[0.2, 0.1, 0.5]
    color_array3  = np.repeat(point_color, num_point).reshape([3, num_point]).astype(np.float64).T


    point_set = np.concatenate((point_set, point_set2, point_set3), axis=0)
    color_array = np.concatenate((color_array, color_array2, color_array3), axis=0)
    visualize_point_cloud(point_set, color_array)


if __name__ == "__main__":
    test()
