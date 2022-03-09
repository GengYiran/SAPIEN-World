import sapien.core as sapien
from sapien.utils import Viewer
import numpy as np
import trimesh
import open3d


def get_global_mesh(obj):
	final_vs = [];
	final_fs = [];
	vid = 0;
	for l in obj.get_links():
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
	final_vs = np.concatenate(final_vs, axis=0)
	final_fs = np.concatenate(final_fs, axis=0)
	return final_vs, final_fs

def sample_pc(v, f, n_points=10000000):
	mesh = trimesh.Trimesh(vertices=v, faces=f)
	points, __ = trimesh.sample.sample_surface(mesh=mesh, count=n_points)
	return points

def demo(fix_root_link, balance_passive_force):
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
	robot: sapien.Articulation = loader.load("/home/gyr/file/Sapien-World/assets/mjcf/100015/mobility.urdf")
	robot.set_root_pose(sapien.Pose([0, 0, 1], [1, 0, 0, 0]))


	# while not viewer.closed:  # Press key q to quit
	# 	scene.step()  # Simulate the world
	# 	scene.update_render()  # Update the world to the renderer
	# 	viewer.render()

	print("get_global_mesh")
	vs, fs = get_global_mesh(robot)
	print("mesh generated!")
	print(vs.shape)
	print(fs.shape)
	obj_pc = sample_pc(v=vs, f=fs, n_points=4096*100)
	print(obj_pc)
	obj_pc = np.array(obj_pc)

	np.save("PCfiles/pot_PC.npy", obj_pc)

	point_cloud = open3d.geometry.PointCloud()
	point_cloud.points = open3d.utility.Vector3dVector(obj_pc)
	open3d.visualization.draw_geometries([point_cloud])

demo(False, False)


