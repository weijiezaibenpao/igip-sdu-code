import trimesh
import numpy as np
from scipy.spatial import cKDTree
import os
from models.model_utils import fps_subsample
import torch
from sklearn.neighbors import KDTree


def model_flat_indices(verts):
    tree = KDTree(verts[:, 0:3])
    neighbours = tree.query(verts[:, 0:3], 20, return_distance=False)
    norms = verts[neighbours][:, :, 3:]
    norms /= (np.sqrt(np.sum(norms ** 2, axis=-1, keepdims=True)) + 1e-6)
    mean_norm = np.mean(norms, axis=1, keepdims=True)
    flats = np.einsum('ijk,ikn->ijn', norms, mean_norm.transpose([0, 2, 1])).squeeze()
    flats = np.mean(np.arccos(np.clip(flats, -1, 1)), axis=-1)
    return flats
def fps(pc, n_point):
    pc_pt = torch.from_numpy(pc).unsqueeze(0).float().cuda()
    pc_sampled = fps_subsample(pc_pt, n_points=n_point)
    pc_sampled_arr = pc_sampled.squeeze(0).detach().cpu().numpy()
    return pc_sampled_arr.astype(np.float32)

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc,m,centroid
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


# 文件路径
data_files = os.listdir('/input')
for data_file in data_files:
    tooth = data_file.split('.')[0]
    obj_file = os.path.join('/input', data_file)
    output_dir = './Pointcept/data/teeth_land/test/' + tooth
    # 读取OBJ文件
    mesh = trimesh.load_mesh(obj_file)

    # 获取顶点和法线
    points = mesh.vertices
    normals = mesh.vertex_normals

    # 最远点采样，采样16384个点
    num_samples = 16384
    if len(points) > 16384:
        sampled_points = fps(points, num_samples)
    else:
        sampled_points = UpSamplePoints(points, num_samples)
    os.makedirs('./temp_data/points', exist_ok=True)
    np.savetxt('./temp_data/points/' +tooth + '.txt', sampled_points)
    # 使用cKDTree查找最近邻来匹配法线
    tree = cKDTree(points)
    _, indices = tree.query(sampled_points, k=1)
    sampled_normals = normals[indices]
    # 保存采样点的曲率 (curvature.txt)
    os.makedirs(output_dir, exist_ok=True)
    curvature_file = output_dir + "/curvature.txt"
    flats = model_flat_indices(np.concatenate([sampled_points, sampled_normals], axis=-1))
    np.savetxt(curvature_file, flats)

    sampled_points, _, _ = pc_normalize(sampled_points)
    # sampled_normals /= (np.sqrt(np.sum(sampled_normals ** 2, axis=-1))[:, None])
    sampled_normals /= (np.sqrt(np.sum(sampled_normals ** 2, axis=-1, keepdims=True)) + 1e-6)

    # 保存采样后的坐标 (coord.txt)
    coord_file = output_dir + "/coord.txt"
    np.savetxt(coord_file, sampled_points)

    # 保存采样点的法线 (normal.txt)
    normal_file = output_dir + "/normal.txt"
    np.savetxt(normal_file, sampled_normals)


    conf_file = output_dir + "/segment.txt"
    conf = np.zeros((6,num_samples))
    np.savetxt(conf_file, conf)

