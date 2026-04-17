import pickle
import numpy as np
from pathlib import Path
from fusion.utils import load_data
from fusion.models import LogisticFusion, MLPFusion, CatBoostFusion

# ── where trained models will be saved ───────────────────────────────────────
SAVE_DIR = Path(__file__).parent / "saved_models"
SAVE_DIR.mkdir(exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────
def save_model(model, name):
    """Save a trained model to disk using pickle."""
    path = SAVE_DIR / f"{name}.pk"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved -> {path}")


def evaluate_on_dev(model, X_dev, y_dev):
    """Quick accuracy check on the dev set."""
    scores    = model.predict_scores(X_dev)
    predicted = (scores >= 0.5).astype(int)
    accuracy  = np.mean(predicted == y_dev) * 100
    print(f"  Dev accuracy: {accuracy:.2f}%")


# ── main training function ────────────────────────────────────────────────────
def train_all():
    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    X_trn, y_trn, _ = load_data("trn")
    X_dev, y_dev, _ = load_data("dev")

    # Detect the actual combined embedding dimension from loaded data
    input_dim = X_trn.shape[1]
    print(f"\n  Detected input dimension: {input_dim}")

    # ── 2. Define all models to train ─────────────────────────────────────────
    models = [
        ("logistic",  LogisticFusion()),
        ("mlp",       MLPFusion(input_dim=input_dim, hidden1=128, hidden2=64,
                                dropout=0.3, lr=0.001, epochs=50)),
        ("catboost",  CatBoostFusion(iterations=500, depth=6,
                                     learning_rate=0.05)),
    ]

    # ── 3. Train, evaluate, and save each model ───────────────────────────────
    for name, model in models:
        print()
        print("=" * 60)
        print(f"Training: {name.upper()}")
        print("=" * 60)

        model.fit(X_trn, y_trn)
        evaluate_on_dev(model, X_dev, y_dev)
        save_model(model, name)

    print()
    print("=" * 60)
    print("All models trained and saved to fusion/saved_models/")
    print("Next step: run fusion/evaluate.py for full EER metrics")
    print("=" * 60)


if __name__ == "__main__":
    train_all()
