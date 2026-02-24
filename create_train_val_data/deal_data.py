import os
import numpy as np
import json


def dist_confidednce(distance, k):
    return np.exp(-k * distance)


def pc_to_dist_confidence(landmark_id_path, teeth_pc_data_path, confidence_sava_path):
    tooth_pc_data = np.loadtxt(teeth_pc_data_path)
    with open(landmark_id_path, 'r') as f:
        landmark_id_data = json.load(f)
    landmark_data_Mesial = []
    landmark_data_Distal = []
    landmark_data_Cusp = []
    landmark_data_InnerPoint = []
    landmark_data_OuterPoint = []
    landmark_data_FacialPoint = []
    for i in range(len(landmark_id_data['objects'])):
        if landmark_id_data['objects'][i]['class'] == 'Mesial':
            landmark_data_Mesial.append(landmark_id_data['objects'][i]['coord'])
        if landmark_id_data['objects'][i]['class'] == 'Distal':
            landmark_data_Distal.append(landmark_id_data['objects'][i]['coord'])
        if landmark_id_data['objects'][i]['class'] == 'Cusp':
            landmark_data_Cusp.append(landmark_id_data['objects'][i]['coord'])
        if landmark_id_data['objects'][i]['class'] == 'InnerPoint':
            landmark_data_InnerPoint.append(landmark_id_data['objects'][i]['coord'])
        if landmark_id_data['objects'][i]['class'] == 'OuterPoint':
            landmark_data_OuterPoint.append(landmark_id_data['objects'][i]['coord'])
        if landmark_id_data['objects'][i]['class'] == 'FacialPoint':
            landmark_data_FacialPoint.append(landmark_id_data['objects'][i]['coord'])
    # ED
    if len(landmark_data_Mesial) != 0:
        dist_Mesial = []
        for i in range(len(landmark_data_Mesial)):
            dis = np.linalg.norm(landmark_data_Mesial[i] - tooth_pc_data[:, 0:3], axis=1)
            dist_Mesial.append(dis)
        min_Mesial = [min(column) for column in zip(*np.asarray(dist_Mesial))]
    else:
        min_Mesial = np.full(len(tooth_pc_data), 12)

    if len(landmark_data_Distal) != 0:
        dist_Distal = []
        for i in range(len(landmark_data_Distal)):
            dis = np.linalg.norm(landmark_data_Distal[i] - tooth_pc_data[:, 0:3], axis=1)
            dist_Distal.append(dis)
        min_Distal = [min(column) for column in zip(*np.asarray(dist_Distal))]
    else:
        min_Distal = np.full(len(tooth_pc_data), 12)

    if len(landmark_data_InnerPoint) != 0:
        dist_InnerPoint = []
        for i in range(len(landmark_data_InnerPoint)):
            dis = np.linalg.norm(landmark_data_InnerPoint[i] - tooth_pc_data[:, 0:3], axis=1)
            dist_InnerPoint.append(dis)
        min_InnerPoint = [min(column) for column in zip(*np.asarray(dist_InnerPoint))]
    else:
        min_InnerPoint = np.full(len(tooth_pc_data), 12)

    if len(landmark_data_OuterPoint) != 0:
        dist_OuterPoint = []
        for i in range(len(landmark_data_OuterPoint)):
            dis = np.linalg.norm(landmark_data_OuterPoint[i] - tooth_pc_data[:, 0:3], axis=1)
            dist_OuterPoint.append(dis)
        min_OuterPoint = [min(column) for column in zip(*np.asarray(dist_OuterPoint))]
    else:
        min_OuterPoint = np.full(len(tooth_pc_data), 12)

    if len(landmark_data_FacialPoint) != 0:
        dist_FacialPoint = []
        for i in range(len(landmark_data_FacialPoint)):
            dis = np.linalg.norm(landmark_data_FacialPoint[i] - tooth_pc_data[:, 0:3], axis=1)
            dist_FacialPoint.append(dis)
        min_FacialPoint = [min(column) for column in zip(*np.asarray(dist_FacialPoint))]
    else:
        min_FacialPoint = np.full(len(tooth_pc_data), 12)

    if len(landmark_data_Cusp) != 0:
        dist_Cusp = []
        for i in range(len(landmark_data_Cusp)):
            dis = np.linalg.norm(landmark_data_Cusp[i] - tooth_pc_data[:, 0:3], axis=1)
            dist_Cusp.append(dis)
        min_Cusp = [min(column) for column in zip(*np.asarray(dist_Cusp))]
    else:
        min_Cusp = np.full(len(tooth_pc_data), 12)

    dist = np.asarray([min_Mesial, min_Distal, min_InnerPoint, min_OuterPoint, min_FacialPoint, min_Cusp])
    confidence = dist.copy()
    confidence = dist_confidednce(confidence, 0.25)  # v4
    np.savetxt(os.path.join(confidence_sava_path, 'segment.txt'), confidence)


if __name__ == '__main__':
    teeth_names = os.listdir('F:/MICCAI_TeethLand24_data/newdata/cuda_16384_no_normalization')
    for teeth_name in teeth_names:
        teeth_type = teeth_name.split('_')[-1]
        teeth_id = teeth_name.split('_')[0]
        # landmark_path
        landmark_id_path = os.path.join('F:/MICCAI_TeethLand24_data/landmark_all/landmark',
                                        teeth_id + '_' + teeth_type + '__kpt.json')
        teeth_pc_data_path = os.path.join('F:/MICCAI_TeethLand24_data/newdata/cuda_16384_no_normalization', teeth_name, 'coord.txt')
        confidence_sava_path = os.path.join('F:/MICCAI_TeethLand24_data/newdata/cuda_16384_normalization', teeth_name)
        pc_to_dist_confidence(landmark_id_path, teeth_pc_data_path, confidence_sava_path)
