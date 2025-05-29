import numpy as np

# Posture Feature: stride length

def compute_stride_length(sequence):
    """
    Input: sequence (30, 48) — 30 frames, 16 joints, each joint has 3D coords
    Output: stride length (float)
    """
    num_frames = sequence.shape[0]
    num_joints = 16
    coords = sequence.reshape(num_frames, num_joints, 3)

    idx_l_foot = 12  # index of left foot
    idx_r_foot = 15  # index of right foot

    dists = np.linalg.norm(coords[:, idx_l_foot, :] - coords[:, idx_r_foot, :], axis=1)
    stride_length = np.max(dists)
    return stride_length


# Per-frame features

def extract_features_per_frame(sequence):
    """
    sequence: np.ndarray with shape (30, 48)
    Each row is a frame, containing 16 joints with 3D coordinates in X,Z,Y order.
    Output: np.ndarray with shape (30, 13)
    """
    num_frames = sequence.shape[0]
    num_joints = 16
    joint_coords = sequence.reshape(num_frames, num_joints, 3)

    # Joint indices
    idx = {
        'root': 0,
        'spine': 1,
        'neck': 2,
        'head': 3,
        'l_shoulder': 4,
        'l_elbow': 5,
        'l_hand': 6,
        'r_shoulder': 7,
        'r_elbow': 8,
        'r_hand': 9,
        'l_hip': 10,
        'l_knee': 11,
        'l_foot': 12,
        'r_hip': 13,
        'r_knee': 14,
        'r_foot': 15
    }

    def angle_between(a, b, c):
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

    def triangle_area(a, b, c):
        ab = b - a
        ac = c - a
        return 0.5 * np.linalg.norm(np.cross(ab, ac))

    features = []

    for t in range(num_frames):
        joints = joint_coords[t]

        min_xyz = joints.min(axis=0)
        max_xyz = joints.max(axis=0)
        volume = np.prod(max_xyz - min_xyz)

        angle_neck_shoulders = angle_between(joints[idx['l_shoulder']], joints[idx['neck']], joints[idx['r_shoulder']])
        angle_r_shoulder = angle_between(joints[idx['neck']], joints[idx['r_shoulder']], joints[idx['l_shoulder']])
        angle_l_shoulder = angle_between(joints[idx['neck']], joints[idx['l_shoulder']], joints[idx['r_shoulder']])

        vertical = np.array([0, 1, 0])
        angle_neck_vertical_back = angle_between(joints[idx['neck']] + vertical, joints[idx['neck']], joints[idx['spine']])
        angle_neck_head_back = angle_between(joints[idx['head']], joints[idx['neck']], joints[idx['spine']])

        dist_r_hand_root = np.linalg.norm(joints[idx['r_hand']] - joints[idx['root']])
        dist_l_hand_root = np.linalg.norm(joints[idx['l_hand']] - joints[idx['root']])
        dist_r_foot_root = np.linalg.norm(joints[idx['r_foot']] - joints[idx['root']])
        dist_l_foot_root = np.linalg.norm(joints[idx['l_foot']] - joints[idx['root']])

        area_hands_neck = triangle_area(joints[idx['l_hand']], joints[idx['r_hand']], joints[idx['neck']])
        area_feet_root = triangle_area(joints[idx['l_foot']], joints[idx['r_foot']], joints[idx['root']])

        stride_length = np.linalg.norm(joints[idx['l_foot']] - joints[idx['r_foot']])

        features.append([
            volume,
            angle_neck_shoulders,
            angle_r_shoulder,
            angle_l_shoulder,
            angle_neck_vertical_back,
            angle_neck_head_back,
            dist_r_hand_root,
            dist_l_hand_root,
            dist_r_foot_root,
            dist_l_foot_root,
            stride_length,
            area_hands_neck,
            area_feet_root
        ])

    return np.array(features)  # (30, 13)


def extract_posture_feature(sequence):
    """
    sequence: (30, 48)
    return: vector of length 14
    """
    stride = compute_stride_length(sequence)
    per_frame = extract_features_per_frame(sequence)
    avg_features = np.mean(per_frame, axis=0)  # shape (13,)
    return np.concatenate([avg_features, [stride]])  # shape (14,)


def compute_movement_features(sequence):
    """
    Input:
        - sequence: np.ndarray of shape (30, 48) — 30 frames, 16 joints, mỗi joint có (X, Z, Y)
    Output:
        - movement features: np.ndarray of shape (16,)
    """

    num_frames = sequence.shape[0]
    num_joints = 16
    coords = sequence.reshape(num_frames, num_joints, 3)

    # Các chỉ số khớp
    idx_joints = {
        'r_hand': 9,
        'l_hand': 6,
        'head': 3,
        'r_foot': 15,
        'l_foot': 12
    }

    # (5, 30, 3)
    joint_trajs = np.stack([coords[:, idx, :] for idx in idx_joints.values()])

    # --- Tốc độ ---
    velocity = np.diff(joint_trajs, axis=1)                   # (5, 29, 3)
    speed = np.linalg.norm(velocity, axis=2)                  # (5, 29)
    avg_speed = np.mean(speed, axis=1)                        # (5,)

    # --- Gia tốc ---
    acceleration = np.diff(speed, axis=1)                     # (5, 28)
    avg_acceleration = np.mean(np.abs(acceleration), axis=1) # (5,)

    # --- Jerk ---
    jerk = np.diff(acceleration, axis=1)                      # (5, 27)
    avg_jerk = np.mean(np.abs(jerk), axis=1)                 # (5,)

    # --- Gait cycle time ---
    gait_cycle_time = num_frames / 30.0  # Giả sử 30 FPS

    # Gộp đặc trưng
    features = np.concatenate([
        avg_speed,
        avg_acceleration,
        avg_jerk,
        [gait_cycle_time]
    ])  # shape (16,)

    return features
