# Định nghĩa danh sách khớp và các cặp cần thiết cho các đặc trưng postural
JOINT_NAMES = [
    "root", "spine", "neck", "head",
    "left_shoulder", "left_elbow", "left_hand",
    "right_shoulder", "right_elbow", "right_hand",
    "left_hip", "left_knee", "left_foot",
    "right_hip", "right_knee", "right_foot"
]

# Các xương chính (cặp khớp để tính độ dài)
BONE_PAIRS = [
    ("head", "neck"), ("neck", "spine"), ("spine", "root"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_hand"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_hand"),
    ("left_hip", "left_knee"), ("left_knee", "left_foot"),
    ("right_hip", "right_knee"), ("right_knee", "right_foot"),
    ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip")
]

# Chiều rộng vai, hông
WIDTH_FEATURES = [
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip")
]
import numpy as np

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def angle_between(p1, p2, p3):
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))  # trả về radian

def compute_features(joints):
    # joints: numpy array shape (16, 3) with order [X, Z, Y]
    joints = np.array(joints).reshape(16, 3)

    # Gán tên cho các khớp
    root, spine, neck, head = joints[0], joints[1], joints[2], joints[3]
    l_shoulder, l_elbow, l_hand = joints[4], joints[5], joints[6]
    r_shoulder, r_elbow, r_hand = joints[7], joints[8], joints[9]
    l_hip, l_knee, l_foot = joints[10], joints[11], joints[12]
    r_hip, r_knee, r_foot = joints[13], joints[14], joints[15]

    features = []

    # 1. Euclidean Distance of Two Feet
    features.append(euclidean(l_foot, r_foot))

    # 2. Euclidean Distance of Two Hands
    features.append(euclidean(l_hand, r_hand))

    # 3. Euclidean Distance of Two Elbows
    features.append(euclidean(l_elbow, r_elbow))

    # 4. Euclidean Distance of Left Hand and Head
    features.append(euclidean(l_hand, head))

    # 5. Euclidean Distance of Right Hand and Head
    features.append(euclidean(r_hand, head))

    # 6. Angle of Left Elbow (shoulder - elbow - hand)
    features.append(angle_between(l_shoulder, l_elbow, l_hand))

    # 7. Angle of Right Elbow
    features.append(angle_between(r_shoulder, r_elbow, r_hand))

    # 8. Orientation of Shoulders on X-Y plane
    delta = np.array(r_shoulder[[0,2]]) - np.array(l_shoulder[[0,2]])
    features.append(np.arctan2(delta[1], delta[0]))

    # 9. Orientation of Shoulders on X-Z plane
    delta = np.array(r_shoulder[[0,1]]) - np.array(l_shoulder[[0,1]])
    features.append(np.arctan2(delta[1], delta[0]))

    # 10. Orientation of Feet on X-Y plane
    delta = np.array(r_foot[[0,2]]) - np.array(l_foot[[0,2]])
    features.append(np.arctan2(delta[1], delta[0]))

    # 11. Right Hand - Right Shoulder in Z
    features.append(r_hand[1] - r_shoulder[1])

    # 12. Left Hand - Left Shoulder in Z
    features.append(l_hand[1] - l_shoulder[1])

    # 13. Right Hand - Right Shoulder in Y
    features.append(r_hand[2] - r_shoulder[2])

    # 14. Left Hand - Left Shoulder in Y
    features.append(l_hand[2] - l_shoulder[2])

    # 15. Right Hand - Left Shoulder in X
    features.append(r_hand[0] - l_shoulder[0])

    # 16. Left Hand - Right Shoulder in X
    features.append(l_hand[0] - r_shoulder[0])

    # 17. Right Hand - Right Elbow in X
    features.append(r_hand[0] - r_elbow[0])

    # 18. Left Hand - Left Elbow in X
    features.append(l_hand[0] - l_elbow[0])

    # 19. Right Elbow - Left Shoulder in X
    features.append(r_elbow[0] - l_shoulder[0])

    # 20. Left Elbow - Right Shoulder in X
    features.append(l_elbow[0] - r_shoulder[0])

    # 21. Right Hand - Right Elbow in Z
    features.append(r_hand[1] - r_elbow[1])

    # 22. Left Hand - Left Elbow in Z
    features.append(l_hand[1] - l_elbow[1])

    # 23. Right Hand - Right Elbow in Y
    features.append(r_hand[2] - r_elbow[2])

    # 24. Left Hand - Left Elbow in Y
    features.append(l_hand[2] - l_elbow[2])

    # 25. Right Elbow - Right Shoulder in Y
    features.append(r_elbow[2] - r_shoulder[2])

    # 26. Left Elbow - Left Shoulder in Y
    features.append(l_elbow[2] - l_shoulder[2])

    # 27. Right Elbow - Right Shoulder in Z
    features.append(r_elbow[1] - r_shoulder[1])

    # 28. Left Elbow - Left Shoulder in Z
    features.append(l_elbow[1] - l_shoulder[1])

    return np.array(features)
import numpy as np

# -------------------- UTILITY FUNCTIONS --------------------

def compute_velocity(pos_t, pos_t1, delta_t=1/30):
    """Compute velocity vector between two time frames."""
    return (np.array(pos_t1) - np.array(pos_t)) / delta_t

def compute_acceleration(vel_t, vel_t1, delta_t=1.0):
    """Compute acceleration vector between two time frames."""
    return (np.array(vel_t1) - np.array(vel_t)) / delta_t

def kinetic_energy(mass, velocity):
    """Compute Kinetic Energy: 0.5 * m * v^2"""
    return 0.5 * mass * np.sum(np.square(velocity))

def momentum(mass, velocity):
    """Compute Momentum: m * v"""
    return mass * np.linalg.norm(velocity)

def force(mass, acceleration):
    """Compute Force: m * a"""
    return mass * np.linalg.norm(acceleration)

# -------------------- HIGH-LEVEL FEATURES --------------------

def compute_spatial_expansion(joints):
    """Compute SEXY, SEYZ, SEXZ from joints: shape (N, 3)"""
    joints = np.array(joints)
    x_max, y_max, z_max = np.max(joints[:, 0]), np.max(joints[:, 1]), np.max(joints[:, 2])
    x_min, y_min, z_min = np.min(joints[:, 0]), np.min(joints[:, 1]), np.min(joints[:, 2])
    SEXY = x_max - x_min
    SEYZ = y_max - y_min
    SEXZ = x_max - x_min if z_max == z_min else (x_max - x_min) / (z_max - z_min)
    return SEXY, SEYZ, SEXZ

def compute_symmetry(joints):
    """Compute symmetry metrics (X, Y, Z) using hand and hip joints"""
    joints = np.array(joints)
    l_hand = joints[5]
    r_hand = joints[8]
    l_hip = joints[11]
    r_hip = joints[14]
    body_center = (l_hip + r_hip) / 2

    symmetry_x = (l_hand[0] - body_center[0]) - (body_center[0] - r_hand[0])
    symmetry_y = l_hand[1] - r_hand[1]
    symmetry_z = l_hand[2] - r_hand[2]
    return symmetry_x, symmetry_y, symmetry_z

def compute_body_bending(joints):
    """Compute bending metric (Y.Head - Y.BodyCenter)"""
    joints = np.array(joints)
    head_y = joints[3][1]
    body_center_y = (joints[11][1] + joints[14][1]) / 2
    return head_y - body_center_y

# -------------------- MASS ESTIMATION --------------------

def segment_mass(segment, M):
    if segment == "head":
        return 0.0307 * M + 2.46
    elif segment == "neck_and_torso":
        return 0.75 * (0.5640 * M - 4.66)
    elif segment == "upper_arm":
        return 0.0274 * M - 0.01
    elif segment == "lower_arm_hand":
        return 0.85 * (0.0233 * M - 0.01)
    elif segment == "hand":
        return 0.15 * (0.0233 * M - 0.01)
    elif segment == "thigh":
        return 0.1159 * M - 1.02
    elif segment == "shank":
        return 0.0452 * M + 0.82
    elif segment == "foot":
        return 0.0069 * M + 0.47
    elif segment == "waist":
        return 0.25 * (0.5640 * M - 4.66)
    else:
        return 0.0

# These functions can now be used with joint positions across frames to compute the 10 high-level features.
def compute_high_level_features(joints_t, joints_t1, body_mass, delta_t=1.0/30):
    """
    Tính 10 đặc trưng cao cấp từ 2 khung hình liên tiếp:
    - joints_t: frame hiện tại (shape: 16 x 3)
    - joints_t1: frame tiếp theo (shape: 16 x 3)
    - body_mass: khối lượng cơ thể (kg)
    """
    joints_t = np.array(joints_t).reshape(16, 3)
    joints_t1 = np.array(joints_t1).reshape(16, 3)

    # Vận tốc và gia tốc toàn cục
    vel = compute_velocity(joints_t, joints_t1, delta_t)
    acc = compute_acceleration(joints_t, joints_t1, delta_t)

    # Chọn 1 số segment chính (có thể mở rộng nếu cần)
    segments = {
        "head": joints_t[3],
        "torso": joints_t[1],
        "upper_arm": joints_t[4],  # left shoulder
        "lower_arm_hand": joints_t[6],  # left hand
        "hand": joints_t[6],
        "thigh": joints_t[11],  # left hip
        "shank": joints_t[12],  # left knee
        "foot": joints_t[13],  # left foot
        "waist": joints_t[1],  # spine
    }

    energy = 0
    momentum_total = 0
    force_total = 0

    for name, pos in segments.items():
        idx = np.where((joints_t == pos).all(axis=1))[0][0]
        m = segment_mass(name, body_mass)
        v = vel[idx]
        a = acc[idx]
        energy += kinetic_energy(m, v)
        momentum_total += momentum(m, v)
        force_total += force(m, a)

    # Spatial expansion
    sexy, seyz, sexz = compute_spatial_expansion(joints_t)

    # Symmetry (tay và hông)
    sym_x, sym_y, sym_z = compute_symmetry(joints_t)

    # Body bending
    bending = compute_body_bending(joints_t)

    return [
        force_total,
        energy,
        momentum_total,
        sexy,
        seyz,
        sexz,
        sym_x,
        sym_y,
        sym_z,
        bending
    ]
def compute_all_features(frames,nxt_frames):
    # Tính đặc trưng
    features = compute_features(frames)
    features_10 = compute_high_level_features(frames, nxt_frames, body_mass=60.0)
    features_10 = np.array(features_10)
    features = np.append(features, features_10)   # returns a new array with 4 appended
    return features
