


print("get_global_mesh")
vs, fs = env.get_global_mesh(env.object)
print("mesh generated!")
print(vs.shape)
print(fs.shape)
obj_pc = env.sample_pc(v=vs, f=fs, n_points=4096*2)



def get_global_mesh(self, obj):
       final_vs = [];
       final_fs = [];
       vid = 0;
       for l in obj.get_links():
           vs = []
           for s in l.get_collision_shapes():
               v = np.array(s.convex_mesh_geometry.vertices, dtype=np.float32)
               f = np.array(s.convex_mesh_geometry.indices, dtype=np.uint32).reshape(-1, 3)
               vscale = s.convex_mesh_geometry.scale
               v[:, 0] *= vscale[0];
               v[:, 1] *= vscale[1];
               v[:, 2] *= vscale[2];
               ones = np.ones((v.shape[0], 1), dtype=np.float32)
               v_ones = np.concatenate([v, ones], axis=1)
               transmat = s.pose.to_transformation_matrix()
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

def sample_pc(self, v, f, n_points=4096):
       mesh = trimesh.Trimesh(vertices=v, faces=f)
       points, __ = trimesh.sample.sample_surface(mesh=mesh, count=n_points)
       return points

