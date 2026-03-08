import sys
import joblib
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
    # Try 1000 evenly spaced thresholds between min and max score
    thresholds = np.linspace(scores.min(), scores.max(), 1000)

    best_eer       = 1.0   # start with worst possible EER (100%)
    best_threshold = 0.5

    for threshold in thresholds:
        # Predictions at this threshold
        predictions = (scores >= threshold).astype(int)

        # False Accept: spoof/impostor that we wrongly accepted
        # These are the samples where y_true=0 but we predicted 1
        spoof_mask       = (y_true == 0)
        false_accept_rate = (predictions[spoof_mask] == 1).mean()

        # False Reject: genuine that we wrongly rejected
        # These are the samples where y_true=1 but we predicted 0
        genuine_mask      = (y_true == 1)
        false_reject_rate = (predictions[genuine_mask] == 0).mean()

        # EER is where FAR and FRR are closest to each other
        # We measure this as the absolute difference between them
        diff = abs(false_accept_rate - false_reject_rate)

        if diff < abs(best_eer - 0.5):
            best_eer       = (false_accept_rate + false_reject_rate) / 2
            best_threshold = threshold

    return best_eer * 100, best_threshold  # return as percentage


# ── load a saved model from disk ──────────────────────────────────────────────
def load_model(name):
    """
    Load a previously trained model from disk.

    Args:
        name : model name string e.g. "logistic", "mlp", "catboost"

    Returns:
        loaded model object
    """
    path = SAVE_DIR / f"{name}.pkl"
    if not path.exists():
        print(f"  [!] Model file not found: {path}")
        print(f"      Run train.py first to generate it.")
        return None
    model = joblib.load(path)
    print(f"  Loaded {name} from {path}")
    return model


# ── evaluate one model ────────────────────────────────────────────────────────
def evaluate_model(model, name, X, y):
    """
    Run evaluation on a single model and return its EER.

    Args:
        model : trained model object
        name  : model name for display
        X     : embeddings (N, 352)
        y     : labels (N,)

    Returns:
        eer : EER percentage
    """
    # Get probability scores from the model
    scores = model.predict_scores(X)

    # Compute EER
    eer, threshold = compute_eer(y, scores)

    # Also compute quick accuracy at the EER threshold
    predictions = (scores >= threshold).astype(int)
    accuracy    = (predictions == y).mean() * 100

    # Break down by genuine vs spoof
    genuine_mask      = (y == 1)
    spoof_mask        = (y == 0)
    false_accept_rate = (predictions[spoof_mask]   == 1).mean() * 100
    false_reject_rate = (predictions[genuine_mask] == 0).mean() * 100

    print(f"\n  [{name}]")
    print(f"    EER            : {eer:.2f}%   (lower is better)")
    print(f"    Threshold      : {threshold:.4f}")
    print(f"    Accuracy       : {accuracy:.2f}%")
    print(f"    False Accept   : {false_accept_rate:.2f}%  (fakes let through)")
    print(f"    False Reject   : {false_reject_rate:.2f}%  (real Alice blocked)")

    return eer


# ── main evaluation function ──────────────────────────────────────────────────
def evaluate_all():
    """
    Load all saved models and evaluate them on the dev and eval sets.
    Prints a comparison table at the end.
    """

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading data")
    print("=" * 60)
    X_dev,  y_dev,  _ = load_data("dev")
    X_eval, y_eval, _ = load_data("eval")
    print(f"Dev  set: {X_dev.shape[0]} samples")
    print(f"Eval set: {X_eval.shape[0]} samples")

    # ── Step 2: Load models ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2: Loading trained models")
    print("=" * 60)
    models = {
        "Logistic" : load_model("logistic"),
        "MLP"      : load_model("mlp"),
        "CatBoost" : load_model("catboost"),
    }

    # Remove any models that failed to load
    models = {k: v for k, v in models.items() if v is not None}

    # ── Step 3: Evaluate on dev set ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3: Evaluation on DEV set")
    print("=" * 60)
    dev_eers = {}
    for name, model in models.items():
        dev_eers[name] = evaluate_model(model, name, X_dev, y_dev)

    # ── Step 4: Evaluate on eval set ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: Evaluation on EVAL set (final)")
    print("=" * 60)
    eval_eers = {}
    for name, model in models.items():
        eval_eers[name] = evaluate_model(model, name, X_eval, y_eval)

    # ── Step 5: Summary comparison table ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY — SASV-EER Comparison (lower is better)")
    print("=" * 60)
    print(f"  {'Model':<12}  {'Dev EER':>10}  {'Eval EER':>10}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}")

    # Baseline from the repo for comparison
    print(f"  {'Baseline':<12}  {'5.12%':>10}  {'7.67%':>10}  ← from main.py")

    for name in models:
        dev_eer  = f"{dev_eers.get(name,  float('nan')):.2f}%"
        eval_eer = f"{eval_eers.get(name, float('nan')):.2f}%"
        print(f"  {name:<12}  {dev_eer:>10}  {eval_eer:>10}")

    # Find best model
    if eval_eers:
        best_name = min(eval_eers, key=eval_eers.get)
        print(f"\n  Best model: {best_name} ({eval_eers[best_name]:.2f}% eval EER)")


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    evaluate_all()