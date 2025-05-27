import compute38features as cp
import h5py
import numpy as np
import pandas as pd

positions = h5py.File("features.h5", 'r')
keys = positions.keys()

for key in keys:
    frame_data = positions[key]
    features = []
    for i in range(1, 240):
        feature = cp.compute_all_features(frame_data[i - 1], frame_data[i])
        features.append(feature)

    features_array = np.array(features)
    df = pd.DataFrame(features_array)
    df.to_csv(f"data_process/{key}.csv", index=False)
    print("done")
