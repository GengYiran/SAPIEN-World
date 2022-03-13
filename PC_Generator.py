from re import U
import sapien.core as sapien
from sapien.utils import Viewer
import numpy as np
import trimesh
import open3d


def mkdir(path):
    # 引入模块
    import os
 
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
 
    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists=os.path.exists(path)
 
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path) 
 
        print(path +' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path+' 目录已存在')
        return False
 
# 定义要创建的目录
mkpath="d:\\qttc\\web\\"
# 调用函数
mkdir(mkpath)


class PC_Generator:
	def __init__(self, URDF_path, save_path, n_points):
		self.URDF_path = URDF_path
		self.save_path = save_path
		self.n_points = n_points

	def get_global_mesh(self,obj, num):
		final_vs = [];
		final_fs = [];
		vid = 0;
		for l in [obj.get_links()[num]]:
			vs = []
			for s in l.get_collision_shapes():
				v = np.array(s.geometry.vertices, dtype=np.float32)
				f = np.array(s.geometry.indices, dtype=np.uint32).reshape(-1, 3)
				vscale = s.geometry.scale
				v[:, 0] *= vscale[0];
				v[:, 1] *= vscale[1];
				v[:, 2] *= vscale[2];
				ones = np.ones((v.shape[0], 1), dtype=np.float32)
				v_ones = np.concatenate([v, ones], axis=1)
				pose = s.get_local_pose()
				transmat = pose.to_transformation_matrix()
				v = (v_ones @ transmat.T)[:, :3]
				vs.append(v)
				final_fs.append(f + vid)
				vid += v.shape[0]
			if len(vs) > 0:
				vs = np.concatenate(vs, axis=0)
				ones = np.ones((vs.shape[0], 1), dtype=np.float32)
				vs_ones = np.concatenate([vs, ones], axis=1)
				transmat = l.get_pose().to_transformation_matrix()
				vs = (vs_ones @ transmat.T)[:, :3]
				final_vs.append(vs)
		if(final_fs!=[] and final_fs!=[]):
			final_vs = np.concatenate(final_vs, axis=0)
			final_fs = np.concatenate(final_fs, axis=0)
		return final_vs, final_fs

	def sample_pc(self,v, f, n_points):
		mesh = trimesh.Trimesh(vertices=v, faces=f)
		points, __ = trimesh.sample.sample_surface(mesh=mesh, count=n_points)
		return points

	def demo(self,fix_root_link, balance_passive_force):
		engine = sapien.Engine()
		renderer = sapien.VulkanRenderer()
		engine.set_renderer(renderer)

		scene_config = sapien.SceneConfig()
		scene = engine.create_scene(scene_config)
		scene.set_timestep(1 / 240.0)
		scene.add_ground(0)

		rscene = scene.get_renderer_scene()
		rscene.set_ambient_light([0.5, 0.5, 0.5])
		rscene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

		viewer = Viewer(renderer)
		viewer.set_scene(scene)
		# viewer.set_camera_xyz(x=-2, y=0, z=1)
		viewer.set_camera_rpy(r=0, p=-0.3, y=0)


		loader: sapien.URDFLoader = scene.create_urdf_loader()
		loader.fix_root_link = fix_root_link
		robot: sapien.Articulation = loader.load(self.URDF_path)
		robot.set_root_pose(sapien.Pose([0, 0, 1], [1, 0, 0, 0]))
		

		for i in range (len(robot.get_links())):
			print("get_global_mesh")
			vs, fs = self.get_global_mesh(robot, i)
			if(vs==[] or fs==[]):
				continue
			print("mesh generated!")
			obj_pc = self.sample_pc(v=vs, f=fs, n_points=self.n_points)
			print(obj_pc)
			obj_pc = np.array(obj_pc)
			# path for each part of object
			save_path = self.save_path+str(robot.get_links()[i])
			# make folder path
			mkdir(self.save_path)

			np.save(save_path, obj_pc)

			point_cloud = open3d.geometry.PointCloud()
			point_cloud.points = open3d.utility.Vector3dVector(obj_pc)
			# open3d.visualization.draw_geometries([point_cloud])
			vis = open3d.visualization.Visualizer()
			
			vis.create_window()
			vis.add_geometry(point_cloud)
			# vis.update_geometry(point_cloud)
			vis.poll_events()
			vis.update_renderer()
			# image path
			image_path = self.save_path+str(robot.get_links()[i])+'.jpg'
			vis.capture_screen_image(image_path)
			vis.destroy_window()

			
## points number to sample
n_points = 4096*100
## URDF path
URDF_path = "/home/gyr/file/Sapien-World/assets/mjcf/102080/mobility.urdf"
## the object name
obj_name='pot3'
## the folder path
save_path = "/home/gyr/file/Sapien-World/PCfiles/{}_PC_all_parts/".format(obj_name)

generator = PC_Generator(URDF_path=URDF_path, save_path=save_path, n_points=n_points)
generator.demo(False, False)