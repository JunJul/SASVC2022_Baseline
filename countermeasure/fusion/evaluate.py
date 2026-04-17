import sys
import pickle
import numpy as np
from pathlib import Path

# Add project root to path so imports work
sys.path.append(str(Path(__file__).parent.parent))

from fusion.utils import load_data

# ── paths ─────────────────────────────────────────────────────────────────────
SAVE_DIR = Path(__file__).parent / "saved_models"


# ── EER computation ───────────────────────────────────────────────────────────


def compute_eer(y_true, scores):
    """
    Compute Equal Error Rate (EER).

    EER = the threshold point where:
        False Accept Rate (FAR) == False Reject Rate (FRR)

    Args:
        y_true : numpy array of true labels (1=bonafide, 0=spoof/impostor)
        scores : numpy array of model scores (higher = more likely bonafide)

    Returns:
        eer       : EER value as a percentage (lower is better)
        threshold : the threshold value where EER occurs
    """
    thresholds = np.linspace(scores.min(), scores.max(), 1000)

    best_diff = float("inf")
    best_eer = 1.0
    best_threshold = 0.5

    for threshold in thresholds:
        predictions = (scores >= threshold).astype(int)

        spoof_mask = (y_true == 0)
        genuine_mask = (y_true == 1)

        false_accept_rate = (predictions[spoof_mask] == 1).mean()
        false_reject_rate = (predictions[genuine_mask] == 0).mean()

        diff = abs(false_accept_rate - false_reject_rate)

        if diff < best_diff:
            best_diff = diff
            best_eer = (false_accept_rate + false_reject_rate) / 2
            best_threshold = threshold

    return best_eer * 100, best_threshold

# ── load a saved model from disk ──────────────────────────────────────────────
def load_model(name):
    """Load a previously trained model from disk."""
    path = SAVE_DIR / f"{name}.pk"
    if not path.exists():
        print(f"  [!] Model file not found: {path}")
        print(f"      Run fusion/train.py first to generate it.")
        return None

    with open(path, "rb") as f:
        model = pickle.load(f)

    print(f"  Loaded {name} from {path}")
    return model


# ── evaluate one model ────────────────────────────────────────────────────────
def evaluate_model(model, name, X, y):
    """Run evaluation on a single model and return its EER."""
    scores = model.predict_scores(X)
    eer, threshold = compute_eer(y, scores)

    predictions = (scores >= threshold).astype(int)
    accuracy    = (predictions == y).mean() * 100

    genuine_mask      = (y == 1)
    spoof_mask        = (y == 0)
    false_accept_rate = (predictions[spoof_mask]   == 1).mean() * 100
    false_reject_rate = (predictions[genuine_mask] == 0).mean() * 100

    print(f"\n  [{name}]")
    print(f"    EER            : {eer:.2f}%   (lower is better)")
    print(f"    Threshold      : {threshold:.4f}")
    print(f"    Accuracy       : {accuracy:.2f}%")
    print(f"    False Accept   : {false_accept_rate:.2f}%  (fakes let through)")
    print(f"    False Reject   : {false_reject_rate:.2f}%  (real blocked)")

    return eer


# ── main evaluation function ──────────────────────────────────────────────────
def evaluate_all():
    """Load all saved models and evaluate on dev and eval sets."""

    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)
    X_dev,  y_dev,  _ = load_data("dev")
    X_eval, y_eval, _ = load_data("eval")
    print(f"Dev  set: {X_dev.shape[0]} samples, dim={X_dev.shape[1]}")
    print(f"Eval set: {X_eval.shape[0]} samples, dim={X_eval.shape[1]}")

    print("\n" + "=" * 60)
    print("STEP 2: Loading trained models")
    print("=" * 60)
    models = {
        "Logistic" : load_model("logistic"),
        "MLP"      : load_model("mlp"),
        "CatBoost" : load_model("catboost"),
    }
    models = {k: v for k, v in models.items() if v is not None}

    print("\n" + "=" * 60)
    print("STEP 3: Evaluation on DEV set")
    print("=" * 60)
    dev_eers = {}
    for name, model in models.items():
        dev_eers[name] = evaluate_model(model, name, X_dev, y_dev)

    print("\n" + "=" * 60)
    print("STEP 4: Evaluation on EVAL set (final)")
    print("=" * 60)
    eval_eers = {}
    for name, model in models.items():
        eval_eers[name] = evaluate_model(model, name, X_eval, y_eval)

    print("\n" + "=" * 60)
    print("SUMMARY -- SASV-EER Comparison (lower is better)")
    print("=" * 60)
    print(f"  {'Model':<12}  {'Dev EER':>10}  {'Eval EER':>10}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}")

    for name in models:
        dev_eer  = f"{dev_eers.get(name,  float('nan')):.2f}%"
        eval_eer = f"{eval_eers.get(name, float('nan')):.2f}%"
        print(f"  {name:<12}  {dev_eer:>10}  {eval_eer:>10}")

    if eval_eers:
        best_name = min(eval_eers, key=eval_eers.get)
        print(f"\n  Best model: {best_name} ({eval_eers[best_name]:.2f}% eval EER)")


if __name__ == "__main__":
    evaluate_all()
