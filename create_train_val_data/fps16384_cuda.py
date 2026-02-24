import trimesh
import numpy as np
from scipy.spatial import cKDTree
import os
from models.model_utils import fps_subsample
import torch
from sklearn.neighbors import KDTree


def fps(pc, n_point):
    pc_pt = torch.from_numpy(pc).unsqueeze(0).float().cuda()
    pc_sampled = fps_subsample(pc_pt, n_points=n_point)
    pc_sampled_arr = pc_sampled.squeeze(0).detach().cpu().numpy()
    return pc_sampled_arr.astype(np.float32)


def model_flat_indices(verts):
    tree = KDTree(verts[:, 0:3])
    neighbours = tree.query(verts[:, 0:3], 20, return_distance=False)
    norms = verts[neighbours][:, :, 3:]
    norms /= (np.sqrt(np.sum(norms ** 2, axis=-1, keepdims=True)) + 1e-6)
    mean_norm = np.mean(norms, axis=1, keepdims=True)
    flats = np.einsum('ijk,ikn->ijn', norms, mean_norm.transpose([0, 2, 1])).squeeze()
    flats = np.mean(np.arccos(np.clip(flats, -1, 1)), axis=-1)
    return flats


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc, m, centroid


def UpSamplePoints(ptcloud, n_points):
    curr = ptcloud.shape[0]
    need = n_points - curr

    if need < 0:
        return ptcloud[np.random.choice(curr, n_points)]

    while curr <= need:
        ptcloud = np.tile(ptcloud, (2, 1))
        need -= curr
        curr *= 2

    choice = np.random.permutation(need)
    ptcloud = np.concatenate((ptcloud, ptcloud[choice]))

    return ptcloud


# landmark_data *.json
# Read the corresponding mesh from the existing landmark data.
json_files = os.listdir('F:/MICCAI_TeethLand24_data/landmark_all/landmark')
for json_file in json_files:
    tooth_name = json_file.split('_')[0]
    tooth_type = json_file.split('_')[1]
    # mesh_path
    obj_file = f"F:/MICCAI_TeethLand24_data/data/{tooth_type}/{tooth_name}/{tooth_name}_{tooth_type}.obj"
    output_dir = f"F:/MICCAI_TeethLand24_data/newdata/cuda_16384_normalization/{tooth_name}_{tooth_type}"
    output_dir2 = f"F:/MICCAI_TeethLand24_data/newdata/cuda_16384_no_normalization/{tooth_name}_{tooth_type}"
  
    mesh = trimesh.load_mesh(obj_file)

  
    points = mesh.vertices
    normals = mesh.vertex_normals


    num_samples = 16384
    if len(points) > 16384:
        sampled_points = fps(points, num_samples)
    else:
        sampled_points = UpSamplePoints(points, num_samples)
    os.makedirs(output_dir2, exist_ok=True)
    np.savetxt(output_dir2 + r"/coord.txt", sampled_points)
   
    tree = cKDTree(points)
    _, indices = tree.query(sampled_points, k=1)
    sampled_normals = normals[indices]

 
    os.makedirs(output_dir, exist_ok=True)
    curvature_file = output_dir + r"/curvature.txt"
    flats = model_flat_indices(np.concatenate([sampled_points, sampled_normals], axis=-1))
    np.savetxt(curvature_file, flats)

    sampled_points, _, _ = pc_normalize(sampled_points)
    sampled_normals /= (np.sqrt(np.sum(sampled_normals ** 2, axis=-1, keepdims=True)) + 1e-6)

    coord_file = output_dir + r"/coord.txt"
    np.savetxt(coord_file, sampled_points)


    normal_file = output_dir + r"/normal.txt"
    np.savetxt(normal_file, sampled_normals)

