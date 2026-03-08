import sys
import joblib
import numpy as np
from pathlib import Path

# Add project root to path so imports work
sys.path.append(str(Path(__file__).parent.parent))

# ── paths ─────────────────────────────────────────────────────────────────────
SAVE_DIR = Path(__file__).parent / "saved_models"


# ── load a saved model from disk ──────────────────────────────────────────────
def load_model(name):
    """
    Load a previously trained model from disk.

    Args:
        name : one of "logistic", "mlp", "catboost"

    Returns:
        loaded model object, or None if not found
    """
    path = SAVE_DIR / f"{name}.pkl"
    if not path.exists():
        print(f"[!] Model not found: {path}")
        print(f"    Run train.py first.")
        return None
    return joblib.load(path)


# ── predict a single trial ────────────────────────────────────────────────────
def predict_trial(asv_embedding, cm_embedding, model_name="logistic"):
    """
    Make an accept/reject decision for a single audio trial.

    This is what your teammates (Task 4 pipeline) will call after they
    extract embeddings from a new audio file.

    Args:
        asv_embedding : numpy array of shape (192,) from ECAPA-TDNN
        cm_embedding  : numpy array of shape (160,) from AASIST
        model_name    : which fusion model to use ("logistic", "mlp", "catboost")

    Returns:
        result : dict with decision, score, and confidence label
    """
    # Load the trained model
    model = load_model(model_name)
    if model is None:
        return None

    # Validate input shapes
    if asv_embedding.shape != (192,):
        raise ValueError(f"ASV embedding must be shape (192,), got {asv_embedding.shape}")
    if cm_embedding.shape != (160,):
        raise ValueError(f"CM embedding must be shape (160,), got {cm_embedding.shape}")

    # Concatenate embeddings into one vector (352,)
    combined = np.concatenate([asv_embedding, cm_embedding])  # (352,)

    # Model expects shape (N, 352) — reshape single sample to (1, 352)
    X = combined.reshape(1, -1)

    # Get probability score (0 to 1, higher = more likely genuine)
    score      = model.predict_scores(X)[0]
    decision   = model.predict(X)[0]

    # Human readable confidence label
    if score >= 0.8:
        confidence = "High confidence"
    elif score >= 0.6:
        confidence = "Medium confidence"
    elif score >= 0.4:
        confidence = "Uncertain"
    else:
        confidence = "High confidence"  # confidently rejecting

    result = {
        "decision"   : "ACCEPT" if decision == 1 else "REJECT",
        "score"      : round(float(score), 4),
        "confidence" : confidence,
        "model_used" : model_name,
    }

    return result


# ── predict using all three models ───────────────────────────────────────────
def predict_all_models(asv_embedding, cm_embedding):
    """
    Run the same trial through all three fusion models and compare results.
    Useful for seeing whether models agree or disagree on a trial.

    Args:
        asv_embedding : numpy array of shape (192,)
        cm_embedding  : numpy array of shape (160,)

    Returns:
        results : dict of { model_name -> result dict }
    """
    results = {}
    for model_name in ["logistic", "mlp", "catboost"]:
        result = predict_trial(asv_embedding, cm_embedding, model_name)
        if result is not None:
            results[model_name] = result

    # Print comparison table
    print("\n  Model Comparison for this trial:")
    print(f"  {'Model':<12}  {'Decision':>8}  {'Score':>8}  {'Confidence'}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*20}")
    for name, r in results.items():
        print(f"  {name:<12}  {r['decision']:>8}  {r['score']:>8.4f}  {r['confidence']}")

    # Check if all models agree
    decisions = [r["decision"] for r in results.values()]
    if len(set(decisions)) == 1:
        print(f"\n  ✅ All models agree: {decisions[0]}")
    else:
        print(f"\n  ⚠️  Models disagree — check individual scores above")

    return results


# ── entry point ───────────────────────────────────────────────────────────────
# When run directly, demonstrate with a simulated trial
if __name__ == "__main__":

    print("=" * 60)
    print("FUSION PREDICTION DEMO")
    print("=" * 60)
    print("(Using simulated embeddings — swap in real ones from teammates)")

    # ── Simulate a GENUINE trial (high ASV score, high CM score) ─────────────
    print("\n--- Trial 1: Simulated GENUINE (real Alice) ---")
    fake_asv_genuine = np.random.randn(192).astype(np.float32) * 0.3 + 1.0
    fake_cm_genuine  = np.random.randn(160).astype(np.float32) * 0.3 + 1.0
    predict_all_models(fake_asv_genuine, fake_cm_genuine)

    # ── Simulate a SPOOF trial (low ASV score, low CM score) ─────────────────
    print("\n--- Trial 2: Simulated SPOOF (AI-generated audio) ---")
    fake_asv_spoof = np.random.randn(192).astype(np.float32) * 0.3 - 1.0
    fake_cm_spoof  = np.random.randn(160).astype(np.float32) * 0.3 - 1.0
    predict_all_models(fake_asv_spoof, fake_cm_spoof)

    # ── Simulate an IMPOSTOR trial (low ASV, high CM — real but wrong person) ─
    print("\n--- Trial 3: Simulated IMPOSTOR (real human, wrong person) ---")
    fake_asv_impostor = np.random.randn(192).astype(np.float32) * 0.3 - 1.0
    fake_cm_impostor  = np.random.randn(160).astype(np.float32) * 0.3 + 1.0
    predict_all_models(fake_asv_impostor, fake_cm_impostor)

    print("\n" + "=" * 60)
    print("To use with real embeddings from teammates:")
    print("""
    from fusion.predict import predict_trial

    result = predict_trial(
        asv_embedding = asv_emb,   # numpy (192,) from ECAPA-TDNN
        cm_embedding  = cm_emb,    # numpy (160,) from AASIST
        model_name    = "catboost" # or "logistic" or "mlp"
    )
    print(result)
    # {'decision': 'ACCEPT', 'score': 0.87, 'confidence': 'High confidence', ...}
    """)
    