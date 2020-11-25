import open3d as o3d
import numpy as np 

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

def test():
    point_set = np.loadtxt("data/samples/airplane_0000_test.txt",delimiter=',').astype(np.float32)
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
