import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mednet_lite import MedNetLite

# Load preprocessed data
data = pd.read_csv('data/mitbih_beats_binary.csv')  # Ensure this file exists
X = np.array(data.iloc[:, :-1])  # ECG samples
y = np.array(data.iloc[:, -1])   # Labels: 0 = Normal, 1 = Abnormal

# Reshape for PyTorch: [batch_size, channels, length]
X = X.reshape(-1, 1, 300)
y = y.astype(np.float32)

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# Convert to tensors
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Initialize model
model = MedNetLite()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs, _ = model(batch_x.float())
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Evaluation
model.eval()
y_true = []
y_pred = []
y_prob = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs, _ = model(batch_x.float())
        probs = outputs.squeeze().numpy()
        preds = (probs > 0.5).astype(int)
        y_true.extend(batch_y.numpy())
        y_pred.extend(preds)
        y_prob.extend(probs)

# Metrics
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_prob)
precision, recall, _ = precision_recall_curve(y_true, y_prob)
pr_auc = auc(recall, precision)

print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"PR AUC: {pr_auc:.4f}")
print("Confusion Matrix:")
print(cm)

# Save model
torch.save(model.state_dict(), 'mednet_lite_weights.pth')
