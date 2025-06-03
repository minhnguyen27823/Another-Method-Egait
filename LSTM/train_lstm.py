from lstm_model import VanillaLSTM
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import get_data

data_path = "data/Data.h5"
label_path = "data/Labels_My.h5"  # chỉnh đúng đường dẫn
Gait = get_data.get_data(data_path)
Labels = get_data.get_labels(label_path)

X = Gait
y = Labels

# Chia train-test (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# 4. Chuyển dữ liệu sang tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# 5. Tạo DataLoader
batch_size = 16
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

input_size = 48
hidden_size = 64
output_size = 4
num_layers = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = VanillaLSTM(input_size, hidden_size, output_size, num_layers).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


num_epochs = 500

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)               # (batch_size, 4)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

model.eval()
with torch.no_grad():
    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)
    outputs = model(X_test_tensor)
    predicted = torch.argmax(outputs, dim=1)
    accuracy = (predicted == y_test_tensor).float().mean()
    print(f"Test accuracy: {accuracy:.4f}")

torch.save(model, "lstm_model_full.pth")