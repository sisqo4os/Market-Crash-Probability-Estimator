#!/usr/bin/env python3
# =========================================================
# Model  105: Market Crash Probability Estimator
# Domain : Finance & Crypto
# File   : market_crash_probability_estimator.onnx
# Output : crash_score
# =========================================================
# Estimates near-term stock market crash probability from macro and technical warning signals.
#
# Input features (shape [1, 5]):
#   [0] yield_curve_inv                — Yield curve inversion depth 0–1
#   [1] pe_ratio_norm                  — Market P/E / 40
#   [2] credit_spread_norm             — High-yield credit spread / 10%
#   [3] vix_norm                       — VIX / 50
#   [4] momentum_negative              — Negative 6-month momentum 0 or 1
#
# Score < 0.5 → LOW CRASH RISK ✅
# Score ≥ 0.5 → CRASH RISK ELEVATED ⚠️
#
# Run : py run_market_crash_probability_estimator.py
# Need: pip install onnxruntime numpy
# =========================================================

import numpy as np
import onnxruntime as rt
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "market_crash_probability_estimator.onnx")

print(f"\nLoading market_crash_probability_estimator.onnx ...")
session = rt.InferenceSession(MODEL_PATH)
print("Model ready!\n")

def predict(values: list) -> dict:
    """Run inference. Pass a list of 5 floats."""
    x = np.array([values], dtype=np.float32)
    score = float(session.run(None, {"features": x})[0][0][0])
    label = "CRASH RISK ELEVATED ⚠️" if score >= 0.5 else "LOW CRASH RISK ✅"
    conf  = score if score >= 0.5 else 1 - score
    bar   = "█" * int(conf * 20) + "░" * (20 - int(conf * 20))
    return {"score": score, "label": label, "confidence": conf, "bar": bar}

def show(result, values, label=""):
    if label: print(f"  Scenario   : {label}")
    print(f"  Input      : {values}")
    print(f"  Result     : {result['label']}")
    print(f"  Confidence : [{result['bar']}] {result['confidence']*100:.1f}%")
    print(f"  Raw score  : {result['score']:.4f}")
    print()

# ── Demo ──────────────────────────────────────────────────
print("=" * 58)
print(f"  Market Crash Probability Estimator — Demo")
print("=" * 58 + "\n")

samples = [
    {"label": "Calm bull market", "values": [0.0, 0.4, 0.2, 0.2, 0.0], "expected": "LOW RISK"},
    {"label": "2008-style conditions", "values": [0.9, 0.9, 0.9, 0.9, 1.0], "expected": "CRASH RISK"},
    {"label": "Overvalued but stable", "values": [0.0, 0.8, 0.2, 0.3, 0.0], "expected": "LOW RISK"},
]
for s in samples:
    show(predict(s["values"]), s["values"], s["label"])

# ── Interactive ───────────────────────────────────────────
print("✏️  Type 5 comma-separated values (or 'quit'):")
print(f"   Features: yield_curve_inv, pe_ratio_norm, credit_spread_norm, vix_norm, momentum_negative\n")
while True:
    raw = input("   > ").strip()
    if raw.lower() in ("quit","exit","q"): break
    if not raw: continue
    try:
        vals = [float(x) for x in raw.split(",")]
        if len(vals) != 5: print(f"   Need exactly 5 values\n"); continue
        show(predict(vals), vals)
    except ValueError:
        print("   Numbers only, please\n")
