import numpy as np
from extract_posture_movement import extract_posture_feature, compute_movement_features
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import get_data

def deepfeatures(model, gait):
    lstm_layer = model.lstm
    sequence = torch.tensor(gait, dtype=torch.float32)  # sequence_np là (30, 48)
    sequence = sequence.unsqueeze(1)  # Thêm batch dimension: (30, 1, 48)
    output, (h_n, c_n) = lstm_layer(sequence)
    num_frames = gait.shape[0]  # nếu Gait shape là (seq_len, features)
    final_hidden = h_n.squeeze(0).squeeze(0)  # (hidden_size,) tensor có grad
    lstm_out = final_hidden[num_frames-1].detach().cpu().numpy()  # convert sang numpy
    return lstm_out

data_path = "data/Data.h5"
label_path = "data/Labels_My.h5"  # chỉnh đúng đường dẫn
Gait = get_data.get_data(data_path)
Labels = get_data.get_labels(label_path)

model = torch.load("lstm_model_full.pth", weights_only=False)
model = model.to("cpu")
Fm = []
Fp = []
DeepFeatures = []
for gait in Gait:
    temp_fm = extract_posture_feature(gait)
    temp_fp = compute_movement_features(gait)
    temp_deep = deepfeatures(model, gait)
    Fm.append(temp_fm)
    Fp.append(temp_fp)
    DeepFeatures.append(temp_deep)
Fm = np.array(Fm)
Fp = np.array(Fp)
DeepFeatures = np.array(DeepFeatures)
full_features = np.concatenate([Fm, Fp, DeepFeatures], axis = 1)



X = full_features
y = Labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True)

rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Huấn luyện
rf.fit(X_train, y_train)

# Dự đoán
y_pred = rf.predict(X_test)

# Đánh giá
print("🎯 Accuracy:", accuracy_score(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))