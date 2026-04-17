import sys
import pickle
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
    path = SAVE_DIR / f"{name}.pk"
    if not path.exists():
        print(f"[!] Model not found: {path}")
        print(f"    Run fusion/train.py first.")
        return None

    with open(path, "rb") as f:
        return pickle.load(f)


# ── predict a single trial ────────────────────────────────────────────────────
def predict_trial(asv_embedding, cm_embedding, model_name="logistic"):
    """
    Make an accept/reject decision for a single audio trial.

    Args:
        asv_embedding : numpy array of shape (asv_dim,) -- typically (192,) from ECAPA-TDNN
        cm_embedding  : numpy array of shape (cm_dim,)  -- depends on AASIST variant
        model_name    : which fusion model to use ("logistic", "mlp", "catboost")

    Returns:
        result : dict with decision, score, and confidence label
    """
    model = load_model(model_name)
    if model is None:
        return None

    # Validate inputs are 1-D vectors
    if asv_embedding.ndim != 1:
        raise ValueError(f"ASV embedding must be 1-D, got shape {asv_embedding.shape}")
    if cm_embedding.ndim != 1:
        raise ValueError(f"CM embedding must be 1-D, got shape {cm_embedding.shape}")

    # Concatenate embeddings
    combined = np.concatenate([asv_embedding, cm_embedding])
    X = combined.reshape(1, -1)

    score      = model.predict_scores(X)[0]
    decision   = model.predict(X)[0]

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

    Args:
        asv_embedding : numpy array of shape (asv_dim,)
        cm_embedding  : numpy array of shape (cm_dim,)

    Returns:
        results : dict of { model_name -> result dict }
    """
    results = {}
    for model_name in ["logistic", "mlp", "catboost"]:
        result = predict_trial(asv_embedding, cm_embedding, model_name)
        if result is not None:
            results[model_name] = result

    print("\n  Model Comparison for this trial:")
    print(f"  {'Model':<12}  {'Decision':>8}  {'Score':>8}  {'Confidence'}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*20}")
    for name, r in results.items():
        print(f"  {name:<12}  {r['decision']:>8}  {r['score']:>8.4f}  {r['confidence']}")

    decisions = [r["decision"] for r in results.values()]
    if len(set(decisions)) == 1:
        print(f"\n  All models agree: {decisions[0]}")
    else:
        print(f"\n  Models disagree -- check individual scores above")

    return results


# ── load real embeddings from .pk files and run inference ────────────────────
def predict_from_pk(split="dev", model_name="logistic", trial_id=None):
    """
    Run fusion inference using REAL embeddings stored in embeddings/*.pk

    Args:
        split      : "dev" or "eval"
        model_name : "logistic", "mlp", or "catboost"
        trial_id   : optional specific trial id; if None, use the first matched one

    Returns:
        result : dict with trial_id and prediction result
    """
    base_dir = Path(__file__).parent.parent / "embeddings"

    asv_path = base_dir / f"asv_embd_{split}.pk"
    cm_path  = base_dir / f"cm_embd_{split}.pk"

    if not asv_path.exists():
        raise FileNotFoundError(f"ASV embedding file not found: {asv_path}")
    if not cm_path.exists():
        raise FileNotFoundError(f"CM embedding file not found: {cm_path}")

    with open(asv_path, "rb") as f:
        asv_dict = pickle.load(f)

    with open(cm_path, "rb") as f:
        cm_dict = pickle.load(f)

    valid_ids = sorted(set(asv_dict.keys()) & set(cm_dict.keys()))
    if not valid_ids:
        raise ValueError(f"No matched trial_ids found between {asv_path.name} and {cm_path.name}")

    if trial_id is None:
        trial_id = valid_ids[0]
    elif trial_id not in asv_dict or trial_id not in cm_dict:
        raise ValueError(f"trial_id '{trial_id}' not found in both embedding files")

    asv_emb = np.asarray(asv_dict[trial_id], dtype=np.float32).reshape(-1)
    cm_emb  = np.asarray(cm_dict[trial_id], dtype=np.float32).reshape(-1)

    print(f"Using split   : {split}")
    print(f"Using trial_id: {trial_id}")
    print(f"ASV shape     : {asv_emb.shape}")
    print(f"CM shape      : {cm_emb.shape}")

    pred = predict_trial(
        asv_embedding=asv_emb,
        cm_embedding=cm_emb,
        model_name=model_name
    )

    return {
        "trial_id": trial_id,
        "result": pred
    }


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("FUSION PREDICTION DEMO (REAL EMBEDDINGS)")
    print("=" * 60)

    out = predict_from_pk(split="dev", model_name="logistic")
    print("\nSingle-model result:")
    print(out)

    print("\n" + "=" * 60)
    print("COMPARE ALL MODELS ON THE SAME REAL TRIAL")
    print("=" * 60)

    base_dir = Path(__file__).parent.parent / "embeddings"
    with open(base_dir / "asv_embd_dev.pk", "rb") as f:
        asv_dict = pickle.load(f)
    with open(base_dir / "cm_embd_dev.pk", "rb") as f:
        cm_dict = pickle.load(f)

    trial_id = sorted(set(asv_dict.keys()) & set(cm_dict.keys()))[0]
    asv_emb = np.asarray(asv_dict[trial_id], dtype=np.float32).reshape(-1)
    cm_emb  = np.asarray(cm_dict[trial_id], dtype=np.float32).reshape(-1)

    print(f"trial_id: {trial_id}")
    predict_all_models(asv_emb, cm_emb)
