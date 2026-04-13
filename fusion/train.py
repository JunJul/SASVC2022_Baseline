import pickle
import numpy as np
from pathlib import Path
from fusion.utils import load_data
from fusion.models import LogisticFusion, MLPFusion, CatBoostFusion

# ── where trained models will be saved ───────────────────────────────────────
# This creates a 'saved_models' folder inside 'fusion/'
SAVE_DIR = Path(__file__).parent / "saved_models"
SAVE_DIR.mkdir(exist_ok=True)  # create the folder if it doesn't exist yet


# ── helpers ───────────────────────────────────────────────────────────────────
def save_model(model, name):
    """
    Save a trained model to disk using pickle.
    The file will be at fusion/saved_models/<name>.pk
    """
    path = SAVE_DIR / f"{name}.pk"
    with open(path, "wb") as f:   # wb = write binary
        pickle.dump(model, f)
    print(f"  Saved → {path}")


def evaluate_on_dev(model, X_dev, y_dev):
    """
    Quick sanity check during training:
    Print accuracy on the dev set so we can compare models.
    (Full EER evaluation happens in evaluate.py)
    """
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

    # ── 2. Define all models to train ─────────────────────────────────────────
    # Each entry is a (name, model) pair
    # Adding them to a list means we can loop over all three identically
    models = [
        ("logistic",  LogisticFusion()),
        ("mlp",       MLPFusion(input_dim=352, hidden1=128, hidden2=64,
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

        # Train on training set
        model.fit(X_trn, y_trn)

        # Quick accuracy check on dev set
        evaluate_on_dev(model, X_dev, y_dev)

        # Save trained model to disk
        save_model(model, name)

    print()
    print("=" * 60)
    print("All models trained and saved to fusion/saved_models/")
    print("Next step: run evaluate.py for full EER metrics")
    print("=" * 60)


# ── entry point ───────────────────────────────────────────────────────────────
# This block only runs when you execute train.py directly
# It does NOT run when another file imports from train.py
if __name__ == "__main__":
    train_all()