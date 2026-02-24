import json
import math
import os.path
import numpy as np
import trimesh
import os
import csv
from collections import defaultdict
from pydpc import Cluster
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

def select_MinPts(data, k):
    k_dist = []
    for i in range(data.shape[0]):
        dist = (((data[i] - data) ** 2).sum(axis=1) ** 0.5)
        dist.sort()
        k_dist.append(dist[k])
    return np.array(k_dist)


def find_eps(data, k):
    k_dist = select_MinPts(data, k)
    k_dist.sort()
    k_dist = k_dist[::-1]
    max_dist_gap = 0
    eps = np.mean(k_dist)
    for i in range(len(k_dist) - 1):
        gap = k_dist[i] - k_dist[i + 1]
        if gap > max_dist_gap:
            max_dist_gap = gap
            eps = k_dist[i + 1]
    return eps

def pc_normalize(pc):
    centroid = np.mean(pc[:, 0:3], axis=0)
    pc[:, 0:3] = pc[:, 0:3] - centroid
    m = np.max(np.sqrt(np.sum(pc[:, 0:3] ** 2, axis=1)))
    pc[:, 0:3] = pc[:, 0:3] / m
    return pc, m, centroid

def point_cluster_dbscan(data, confidence_threshold):
    filtered_points = data[data[:, 3] > confidence_threshold]
    if len(filtered_points) == 0:
        return []
    data, m, c = pc_normalize(filtered_points)
    # k = 2 * (data.shape[-1] - 4) - 1
    k = 3
    eps = find_eps(data[:, 0:3], k)
    # print(eps)
    dbscan_model = DBSCAN(eps=eps, min_samples=k + 1)
    labels = dbscan_model.fit_predict(data[:, 0:3])

    # colors = np.random.rand(max(labels) + 1, 3) * 255
    # RGB = np.zeros(data.shape)
    # for idx, value in enumerate(labels):
    #     if value >= 0:
    #         RGB[idx] = colors[value]
    # res = np.concatenate([data * m + c, RGB], axis=1)

    unique_labels = np.unique(labels)
    clustered_data = []
    for label in unique_labels:
        if label == -1:
            continue
        cluster_data = data[labels == label]
        cluster_data[:, 0:3] = cluster_data[:, 0:3] * m + c
        landmark = cluster_data[np.argmax(cluster_data[:, -1])]

        clustered_data.append(landmark)

    return clustered_data

def point_cluster_kmeans(data, confidence_threshold, n_clusters):
    # 过滤出置信度大于给定阈值的点
    filtered_points = data[data[:, 3] > confidence_threshold]
    if len(filtered_points) == 0:
        return []
    
    # 数据归一化
    data, m, c = pc_normalize(filtered_points)
    
    # 使用KMeans进行聚类
    kmeans_model = KMeans(n_clusters=n_clusters)
    labels = kmeans_model.fit_predict(data[:, 0:3])

    # 获取每个簇中的中心点
    unique_labels = np.unique(labels)
    clustered_data = []
    
    for label in unique_labels:
        cluster_data = data[labels == label]
        cluster_data[:, 0:3] = cluster_data[:, 0:3] * m + c
        landmark = cluster_data[np.argmax(cluster_data[:, -1])]
        clustered_data.append(landmark)

    return clustered_data


confidence_path = './Pointcept/exp/teeth_land/semseg-pt-v3m1-0-base/result_test_norm_new'
landmark_save_path = '/output'
os.makedirs(landmark_save_path, exist_ok=True)
tooth_files = os.listdir(confidence_path)
colors = [[255, 0, 0], [0, 255, 0], [255, 0, 255], [0, 255, 255], [255, 255, 0], [0, 0, 255]]

with open(os.path.join(landmark_save_path, 'predictions.csv'), mode='w', newline='') as file:
    writer = csv.writer(file)
    # 写入表头
    writer.writerow(["key", "coord_x", "coord_y", "coord_z","class", "score"])
    # 写入数据
    for tooth in tooth_files:
        tooth_name = tooth.split('.')[0]
        teeth_Mesial = []
        teeth_Distal = []
        teeth_InnerPoint = []
        teeth_OuterPoint = []
        teeth_FacialPoint = []
        teeth_Cusp = []

        confidence_data = np.loadtxt(os.path.join(confidence_path, tooth))
        tooth_data = confidence_data[:, :3]
        confidence_data = confidence_data[:, 3:9].transpose()

        Distal_confidence = np.concatenate((tooth_data, confidence_data[1].reshape(-1, 1)), axis=1)
        Distal_data = point_cluster_dbscan(Distal_confidence,  0.8)
        # Distal_data = Distal_confidence[np.argmax(Distal_confidence[:, -1])]
        # Distal_data = np.concatenate((Distal_data, colors[1]))

        Mesial_confidence = np.concatenate((tooth_data, confidence_data[0].reshape(-1, 1)), axis=1)
        Mesial_data = point_cluster_dbscan(Mesial_confidence,  0.8)
        # Mesial_data = point_cluster_kmeans(Mesial_confidence,  0.8,len(Distal_data))
        
        # Mesial_data = Mesial_confidence[np.argmax(Mesial_confidence[:, -1])]
        # Mesial_data = np.concatenate((Mesial_data, colors[0]))



        FacialPoint_confidence = np.concatenate((tooth_data, confidence_data[4].reshape(-1, 1)), axis=1)
        FacialPoint_data = point_cluster_dbscan(FacialPoint_confidence, 0.8)
        # FacialPoint_data = FacialPoint_confidence[np.argmax(FacialPoint_confidence[:, -1])]
        # FacialPoint_data = np.concatenate((FacialPoint_data, colors[2]))

        OuterPoint_confidence = np.concatenate((tooth_data, confidence_data[3].reshape(-1, 1)), axis=1)
        OuterPoint_data = point_cluster_dbscan(OuterPoint_confidence, 0.8)
        # OuterPoint_data = OuterPoint_confidence[np.argmax(OuterPoint_confidence[:, -1])]
        # OuterPoint_data = np.concatenate((OuterPoint_data, colors[3]))

        InnerPoint_confidence = np.concatenate((tooth_data, confidence_data[2].reshape(-1, 1)), axis=1)
        InnerPoint_data = point_cluster_dbscan(InnerPoint_confidence, 0.8)
        # InnerPoint_data = InnerPoint_confidence[np.argmax(InnerPoint_confidence[:, -1])]
        # InnerPoint_data = np.concatenate((InnerPoint_data, colors[4]))

        Cusp_confidence = np.concatenate((tooth_data, confidence_data[5].reshape(-1, 1)), axis=1)

        # Cusp_data = point_cluster(Cusp_confidence, 'Cusp', 0.7, 0.5)
        Cusp_data = point_cluster_dbscan(Cusp_confidence, 0.8)
        # Cusp_data = point_cluster_dbscan2(Cusp_confidence, 0.8, 0.1, 5)

        if len(Mesial_data) == 0:
            continue
        teeth_Mesial.append(np.asarray(Mesial_data))
        if len(Distal_data) == 0:
            continue
        teeth_Distal.append(np.asarray(Distal_data))
        if len(InnerPoint_data) == 0:
            continue
        teeth_InnerPoint.append(np.asarray(InnerPoint_data))
        if len(OuterPoint_data) == 0:
            continue
        teeth_OuterPoint.append(np.asarray(OuterPoint_data))
        if len(FacialPoint_data) == 0:
            continue
        teeth_FacialPoint.append(np.asarray(FacialPoint_data))
        if len(Cusp_data) == 0:
            continue
        # elif len(Cusp_data) > 1:
        #     color = np.ones((len(Cusp_data), 3)) * colors[5]
        #     Cusp_data = np.concatenate((Cusp_data, color), axis=1)
        # else:
        #     Cusp_data = np.concatenate((Cusp_data[0], colors[5]))
        teeth_Cusp.append(np.asarray(Cusp_data))
        for i in range(len(teeth_Mesial)):
            for  Mesial_data in teeth_Mesial[i]:
                writer.writerow([tooth_name, Mesial_data[0], Mesial_data[1],Mesial_data[2],'Mesial', 1])
        for i in range(len(teeth_Distal)):
            for  Distal_data in teeth_Distal[i]:
                writer.writerow([tooth_name , Distal_data[0], Distal_data[1], Distal_data[2], 'Distal', 1])
        for i in range(len(teeth_InnerPoint)):
            for  InnerPoint_data in teeth_InnerPoint[i]:
                writer.writerow([tooth_name, InnerPoint_data[0], InnerPoint_data[1],InnerPoint_data[2],'InnerPoint', 1])
        for i in range(len(teeth_OuterPoint)):
            for  OuterPoint_data in teeth_OuterPoint[i]:
                writer.writerow([tooth_name, OuterPoint_data[0],OuterPoint_data[1],OuterPoint_data[2], 'OuterPoint', 1])
        for i in range(len(teeth_FacialPoint)):
            for  FacialPoint_data in teeth_FacialPoint[i]:
                writer.writerow([tooth_name, FacialPoint_data[0], FacialPoint_data[1],FacialPoint_data[2],'FacialPoint', 1])
        for i in range(len(teeth_Cusp)):
            for  Cusp_data in teeth_Cusp[i]:
                writer.writerow([tooth_name, Cusp_data[0], Cusp_data[1],Cusp_data[2],'Cusp', 1])
