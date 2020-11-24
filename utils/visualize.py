import open3d as o3d
import numpy as np 

def visualize_point_cloud(point_array):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=960)
    render_op = vis.get_render_option()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_array.astype(np.float64))
    vis.add_geometry(pcd)

    vis.run()
    vis.destroy_window()

def test():
    point_set = np.loadtxt("data/samples/airplane_0001.txt",delimiter=',').astype(np.float32)
    point_set = point_set[:,:3]
    visualize_point_cloud(point_set)

if __name__ == "__main__":
    test()
