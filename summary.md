# TECH 405 — RNN: Executive Summary

**Student:** Prabesh  

## 1. Task & Dataset Information
- **Dataset Chosen:** **Dataset #8: Digits (`sklearn.datasets.load_digits`)**
- **Task:** 10-Class Digit Classification (`0` through `9`)
- **Model Architecture:** Lecture-specified pure PyTorch LSTM:
  $$\text{nn.LSTM}(\text{input\_size}=8, \text{hidden\_size}=64, \text{batch\_first}=\text{True}) \longrightarrow \text{nn.Linear}(64, 10)$$
  *(No pretrained models, no external backbones)*
- **Data Partitioning:** Fixed **80/20 train/test split** with stratification (`random_state=42`).
- **Evaluation Rule:** Evaluated strictly on the test set once after training; no hyperparameter tuning on the test set.

---

## 2. Why Digits is a Sequence Problem
In sequential image classification, each $8 \times 8$ grayscale digit image is treated as a temporal sequence of horizontal raster scanlines:
- **Sequence Length ($T$):** 8 time steps (the 8 rows from top to bottom).
- **Feature Dimension ($D$):** 8 features per time step (the 8 horizontal pixel values per row).

Just like speech audio signals unfolding across time or an optical scanner scanning from top to bottom, handwritten digits exhibit distinct spatial stroke trajectories across rows (e.g., top bars for `7`, upper and lower closed loops for `8` and `0`, vertical stems for `1` and `4`). The LSTM updates its recurrent hidden state $h_t$ and cell state $c_t$ sequentially at each row, capturing stroke context over time until the final 8th time step produces a comprehensive sequence representation for classification.

---

## 3. Required Report Snapshots

### Snapshot 1: Data + Shapes
```text
Total dataset samples: 1797
Input image shape per sample: (8, 8) (8 rows x 8 columns)
Sequence interpretation: 8 time steps, 8 features per step
X_train shape: (1437, 8, 8) | y_train shape: (1437,)
X_test shape:  (360, 8, 8)  | y_test shape:  (360,)
Class distribution in test set (10 classes): [36, 36, 35, 37, 36, 37, 36, 36, 35, 36]
Baseline (majority class) test accuracy: 10.28%
```
> **Caption (Snapshot 1):** Scikit-learn digits dataset split into 1,437 training sequences (80%) and 360 test sequences (20%) using stratified sampling. Pixel intensities are normalized to the range $[0.0, 1.0]$. The majority class baseline on the test set is 10.28% (37/360).

---

### Snapshot 2: Model Definition
```python
class SequentialDigitsLSTM(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=num_classes)
        
    def forward(self, x):
        # x: (batch_size, seq_len=8, input_dim=8)
        out, (h_n, c_n) = self.lstm(x)
        # out[:, -1, :] corresponds to the final time step's hidden state
        logits = self.fc(out[:, -1, :])
        return logits

# Total trainable parameters: 19,594
```
> **Caption (Snapshot 2):** Standard PyTorch LSTM recurrent model conforming to lecture requirements: an input layer accepting 8 features per step, a 64-dimensional recurrent hidden layer, and a linear projection head yielding logits for the 10 digit classes. Total parameters: 19,594.

---

### Snapshot 3: Training Loss Decreasing
```text
Epoch [01/25] - Training Loss: 1.84513
Epoch [05/25] - Training Loss: 0.27209
Epoch [10/25] - Training Loss: 0.09212
Epoch [15/25] - Training Loss: 0.02627
Epoch [20/25] - Training Loss: 0.03455
Epoch [25/25] - Training Loss: 0.00321
```
> **Caption (Snapshot 3):** Cross-entropy training loss steadily decreased over 25 epochs from $1.84513$ down to $0.00321$ using the Adam optimizer ($\text{lr} = 0.005$, batch size $= 32$). The visual loss curve is saved in `plots/training_loss.png`.

---

### Snapshot 4: Final Test Accuracy + Evaluation Plot
```text
Baseline Accuracy (Majority Class): 10.00% (exact: 10.28%)
Target Accuracy (Full Marks):       93.00%
Achieved Model Test Accuracy:       98.33% (354 / 360 correct)

Classification Report (Test Set):
              precision    recall  f1-score   support
           0     1.0000    0.9722    0.9859        36
           1     0.9444    0.9444    0.9444        36
           2     0.9722    1.0000    0.9859        35
           3     1.0000    1.0000    1.0000        37
           4     0.9730    1.0000    0.9863        36
           5     0.9737    1.0000    0.9867        37
           6     1.0000    0.9722    0.9859        36
           7     1.0000    1.0000    1.0000        36
           8     0.9706    0.9429    0.9565        35
           9     1.0000    1.0000    1.0000        36
    accuracy                         0.9833       360
   macro avg     0.9834    0.9832    0.9832       360
```
> **Caption (Snapshot 4):** Final test evaluation on the unseen test set yielded 98.33% accuracy (354 correct out of 360). The side-by-side benchmark comparison bar chart and 10-class confusion matrix are saved in `plots/benchmark_and_confusion_matrix.png`.

---

## 4. Benchmark Comparison (One Line)

> **Achieved Test Accuracy: 98.33% vs. Baseline (majority-class guess): 10.28% vs. Target (full marks): 93.00%.**

*(Exceeds full-marks target by **+5.33%**)*

---

## 5. Honest Reflection (One Honest Sentence)

> **Feeding normalized 8-pixel horizontal scanlines sequentially into `nn.LSTM(8, 64)` and classifying from the final hidden state cleanly captured spatial stroke transitions from top to bottom (reaching 98.33% test accuracy), whereas unnormalized raw pixel values or time-step mean pooling degraded convergence speed and discriminability.**

---

## 6. Verification and Project Files
All scripts, notebooks, and plots have been generated, executed, and verified:
1. `train_digits_lstm.py`: Standalone Python script that trains the LSTM, prints all snapshot outputs, and writes plot images.
2. `TECH_405_RNN_Classification.ipynb`: Jupyter Notebook with all cells pre-executed and visualization plots embedded.
3. `plots/training_loss.png`: Plot showing cross-entropy loss decreasing over epochs.
4. `plots/benchmark_and_confusion_matrix.png`: Bar chart comparing Baseline vs. Target vs. Model, paired with the test confusion matrix.
5. `README.md` & `summary.md`: Full project documentation and submission summaries.
