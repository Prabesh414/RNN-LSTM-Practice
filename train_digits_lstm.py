"""
TECH 405 — RNN Sequential Classification
Dataset: Digits (sklearn) as Sequential Data
Model: nn.LSTM(8, hidden_dim) -> nn.Linear(hidden_dim, 10)
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 1. Load Data
digits = load_digits()
# Images are 8x8 pixels with values in 0..16
X = digits.images.astype(np.float32) / 16.0  # Normalize to [0, 1]
y = digits.target.astype(np.int64)

# 2. Fixed 80/20 Train-Test Split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Majority class baseline on the test set
test_counts = np.bincount(y_test)
majority_class = np.argmax(test_counts)
baseline_acc = (test_counts[majority_class] / len(y_test)) * 100.0

print("="*60)
print("SNAPSHOT 1: DATA & SHAPES")
print("="*60)
print(f"Total dataset samples: {len(X)}")
print(f"Input image shape per sample: {digits.images[0].shape} (8 rows x 8 columns)")
print(f"Sequence interpretation: 8 time steps, 8 features per step")
print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
print(f"X_test shape:  {X_test.shape}  | y_test shape:  {y_test.shape}")
print(f"Class distribution in test set (10 classes): {test_counts.tolist()}")
print(f"Baseline (majority class) test accuracy: {baseline_acc:.2f}%\n")

# PyTorch DataLoaders
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 3. Model Definition: nn.LSTM -> nn.Linear
class SequentialDigitsLSTM(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_classes=10):
        super().__init__()
        # Input features per step = 8, hidden state dimension = 64
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        # Final classification linear head
        self.fc = nn.Linear(in_features=hidden_dim, out_features=num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len=8, input_dim=8)
        out, (h_n, c_n) = self.lstm(x)
        # out shape: (batch_size, seq_len=8, hidden_dim)
        # Use representation from the final time step (step 8)
        last_step_out = out[:, -1, :]
        logits = self.fc(last_step_out)
        return logits

model = SequentialDigitsLSTM(input_dim=8, hidden_dim=64, num_classes=10)

print("="*60)
print("SNAPSHOT 2: MODEL DEFINITION")
print("="*60)
print(model)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}\n")

# 4. Training
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

EPOCHS = 25
loss_history = []

print("="*60)
print("SNAPSHOT 3: TRAINING PROGRESSION")
print("="*60)

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * len(batch_y)
        
    epoch_loss = running_loss / len(train_dataset)
    loss_history.append(epoch_loss)
    
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] - Training Loss: {epoch_loss:.5f}")

# 5. Final Evaluation on Test Set Only
model.eval()
all_preds = []
all_targets = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(batch_y.cpu().numpy())

all_preds = np.array(all_preds)
all_targets = np.array(all_targets)
test_accuracy = accuracy_score(all_targets, all_preds) * 100.0

print("\n" + "="*60)
print("SNAPSHOT 4: FINAL TEST ACCURACY & EVALUATION")
print("="*60)
print(f"Baseline Accuracy (Majority Class): 10.00% (exact: {baseline_acc:.2f}%)")
print(f"Target Accuracy (Full Marks):       93.00%")
print(f"Achieved Model Test Accuracy:       {test_accuracy:.2f}%\n")
print("Classification Report:")
print(classification_report(all_targets, all_preds, digits=4))

# 6. Generate and Save Visualizations
os.makedirs("plots", exist_ok=True)

# Plot 1: Training Loss Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), loss_history, marker='o', color='#2563EB', linewidth=2, label='Training Loss (Cross-Entropy)')
plt.title("Snapshot 3: Training Loss Progression Across Epochs", fontsize=14, fontweight='bold', pad=12)
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(range(1, EPOCHS + 1, 2))
plt.legend(fontsize=11)
plt.tight_layout()
loss_plot_path = os.path.join("plots", "training_loss.png")
plt.savefig(loss_plot_path, dpi=300)
plt.close()
print(f"Saved: {loss_plot_path}")

# Plot 2: Benchmark Comparison & Confusion Matrix
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Benchmark Comparison Bar Chart
bars = ax1.bar(['Baseline (Guess)', 'Target (Full Marks)', 'Our LSTM Model'], 
               [baseline_acc, 93.0, test_accuracy], 
               color=['#94A3B8', '#F59E0B', '#10B981'],
               width=0.55)
ax1.set_ylim(0, 110)
ax1.set_ylabel("Accuracy (%)", fontsize=12)
ax1.set_title("Benchmark Comparison: Baseline vs Target vs Model", fontsize=13, fontweight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.5)

for bar in bars:
    height = bar.get_height()
    ax1.annotate(f"{height:.2f}%",
                 xy=(bar.get_x() + bar.get_width() / 2, height),
                 xytext=(0, 5), textcoords="offset points",
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

# Subplot 2: Confusion Matrix Heatmap
cm = confusion_matrix(all_targets, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax2,
            xticklabels=range(10), yticklabels=range(10))
ax2.set_xlabel("Predicted Digit", fontsize=12)
ax2.set_ylabel("True Digit", fontsize=12)
ax2.set_title("Test Set Confusion Matrix", fontsize=13, fontweight='bold')

plt.tight_layout()
benchmark_plot_path = os.path.join("plots", "benchmark_and_confusion_matrix.png")
plt.savefig(benchmark_plot_path, dpi=300)
plt.close()
print(f"Saved: {benchmark_plot_path}")
