import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: Logistic Regression
# ─────────────────────────────────────────────────────────────────────────────
# Simplest model — draws a straight line between accept and reject.
# Good baseline. Fast to train. Easy to interpret.
# Input:  X of shape (N, 352)
# Output: scores of shape (N,)  — higher = more likely genuine

class LogisticFusion:
    def __init__(self):
        # StandardScaler normalizes each feature to mean=0, std=1
        # Important because our 352 features may have very different ranges
        self.scaler = StandardScaler()

        # max_iter=1000 gives it enough steps to converge
        # C=1.0 controls regularization (higher C = less regularization)
        self.model  = LogisticRegression(max_iter=1000, C=1.0)

    def fit(self, X, y):
        """
        Train the model.
        X : (N, 352) embeddings
        y : (N,)     labels  1=accept, 0=reject
        """
        X_scaled = self.scaler.fit_transform(X)  # learn mean/std then scale
        self.model.fit(X_scaled, y)
        print("[LogisticFusion] Training complete.")

    def predict_scores(self, X):
        """
        Return probability of being genuine (bonafide) for each trial.
        Higher score = more likely to accept.
        Output shape: (N,)
        """
        X_scaled = self.scaler.transform(X)      # use the same mean/std from fit
        # predict_proba returns [[prob_reject, prob_accept], ...]
        # we take column 1 = probability of class 1 (accept)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X, threshold=0.5):
        """
        Return binary decisions: 1=accept, 0=reject.
        """
        scores = self.predict_scores(X)
        return (scores >= threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: MLP (Multi-Layer Perceptron)
# ─────────────────────────────────────────────────────────────────────────────
# Neural network with two hidden layers.
# Can learn non-linear patterns that logistic regression misses.
#
# Architecture:
#   Input (352) → Linear → ReLU → Dropout
#               → Linear → ReLU → Dropout
#               → Linear → Sigmoid → Output (1 score)

class MLPNetwork(nn.Module):
    """
    The actual PyTorch neural network definition.
    Separate from MLPFusion so the architecture is clearly visible.
    """
    def __init__(self, input_dim=352, hidden1=128, hidden2=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: input → hidden1
            nn.Linear(input_dim, hidden1),  # 352 → 128
            nn.ReLU(),                       # kill negative values
            nn.Dropout(dropout),             # randomly turn off 30% of neurons

            # Layer 2: hidden1 → hidden2
            nn.Linear(hidden1, hidden2),     # 128 → 64
            nn.ReLU(),
            nn.Dropout(dropout),

            # Output layer: hidden2 → 1 score
            nn.Linear(hidden2, 1),           # 64 → 1
            nn.Sigmoid()                     # squish to (0, 1) probability
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # shape: (N, 1) → (N,)


class MLPFusion:
    def __init__(self, input_dim=352, hidden1=128, hidden2=64,
                 dropout=0.3, lr=0.001, epochs=50, batch_size=256):
        """
        input_dim  : size of input vector (352 = 192 ASV + 160 CM)
        hidden1    : neurons in first hidden layer
        hidden2    : neurons in second hidden layer
        dropout    : fraction of neurons randomly turned off during training
        lr         : learning rate — how big each weight update step is
        epochs     : how many times to go through the full training set
        batch_size : how many samples to process before updating weights
        """
        self.epochs     = epochs
        self.batch_size = batch_size
        self.scaler     = StandardScaler()

        # Use GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MLPFusion] Using device: {self.device}")

        # Create the network and move it to the device
        self.model = MLPNetwork(input_dim, hidden1, hidden2, dropout).to(self.device)

        # Binary Cross Entropy loss — standard for binary classification
        self.criterion = nn.BCELoss()

        # Adam optimizer — handles learning rate adjustments automatically
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, X, y):
        """
        Train the MLP.
        X : (N, 352) numpy array
        y : (N,)     numpy array of 0/1 labels
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Convert numpy arrays to PyTorch tensors
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y,        dtype=torch.float32).to(self.device)

        # Switch model to training mode (activates dropout)
        self.model.train()

        for epoch in range(self.epochs):
            # Shuffle data at the start of each epoch
            perm    = torch.randperm(len(X_tensor))
            X_shuf  = X_tensor[perm]
            y_shuf  = y_tensor[perm]

            epoch_loss = 0.0
            num_batches = 0

            # Process data in batches
            for i in range(0, len(X_shuf), self.batch_size):
                X_batch = X_shuf[i : i + self.batch_size]
                y_batch = y_shuf[i : i + self.batch_size]

                # Forward pass: make predictions
                predictions = self.model(X_batch)

                # Compute loss (how wrong are we?)
                loss = self.criterion(predictions, y_batch)

                # Backward pass: compute gradients
                self.optimizer.zero_grad()  # clear old gradients
                loss.backward()             # compute new gradients

                # Update weights
                self.optimizer.step()

                epoch_loss  += loss.item()
                num_batches += 1

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"  Epoch {epoch+1}/{self.epochs}  loss: {avg_loss:.4f}")

        print("[MLPFusion] Training complete.")

    def predict_scores(self, X):
        """
        Return probability of being genuine for each trial.
        Output shape: (N,)
        """
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)

        # Switch to eval mode (disables dropout for consistent predictions)
        self.model.eval()
        with torch.no_grad():  # no need to track gradients during inference
            scores = self.model(X_tensor)

        return scores.cpu().numpy()  # move back to CPU and convert to numpy

    def predict(self, X, threshold=0.5):
        """
        Return binary decisions: 1=accept, 0=reject.
        """
        scores = self.predict_scores(X)
        return (scores >= threshold).astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3: CatBoost (Gradient Boosted Trees)
# ─────────────────────────────────────────────────────────────────────────────
# Builds hundreds of decision trees, each one fixing the mistakes of the last.
# Often the strongest model on tabular data.
# Requires almost no tuning to get good results.

class CatBoostFusion:
    def __init__(self, iterations=500, depth=6, learning_rate=0.05):
        """
        iterations    : number of trees to build (more = stronger but slower)
        depth         : how many yes/no questions each tree can ask
        learning_rate : how much each new tree corrects the previous ones
        """
        self.scaler = StandardScaler()
        self.model  = CatBoostClassifier(
            iterations    = iterations,
            depth         = depth,
            learning_rate = learning_rate,
            loss_function = "Logloss",   # standard for binary classification
            verbose       = 100,         # print progress every 100 trees
            random_seed   = 42
        )

    def fit(self, X, y):
        """
        Train CatBoost.
        X : (N, 352) numpy array
        y : (N,)     numpy array of 0/1 labels
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        print("[CatBoostFusion] Training complete.")

    def predict_scores(self, X):
        """
        Return probability of being genuine for each trial.
        Output shape: (N,)
        """
        X_scaled = self.scaler.transform(X)
        # predict_proba returns [[prob_reject, prob_accept], ...]
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict(self, X, threshold=0.5):
        """
        Return binary decisions: 1=accept, 0=reject.
        """
        scores = self.predict_scores(X)
        return (scores >= threshold).astype(int)