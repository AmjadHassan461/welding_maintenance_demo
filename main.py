import argparse
from pathlib import Path
import subprocess
import shutil
import glob
import sys, os
from torch.optim.adam import Adam
import platform
import re
import pandas as pd 
from src.utils.paths import normalize_path
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from joblib import dump, load

from src.config import (
    MODELS_DIR, DEVICE, VISION_EPOCHS, SENSOR_EPOCHS, SENSOR_LR, RANDOM_STATE,
    YOLOV5_DIR, YOLO_DATA_YAML, YOLO_PROJECT_NAME
)
from src.data.load_sensors import load_sensor_windows
from src.models.sensor_rf import train_sensor_rf
from src.utils.plotting import plot_confusion_matrix


# -----------------------------
# Vision training (YOLOv5)
# -----------------------------
def train_vision(args):
    print(f"[Vision] Training YOLOv5 for {VISION_EPOCHS} epochs...")
    weights = "yolov5s.pt"
    run_name = YOLO_PROJECT_NAME

    cmd = [
        sys.executable, str(YOLOV5_DIR / "train.py"),
        "--img", "640",
        "--batch", "16",
        "--epochs", str(VISION_EPOCHS),
        "--data", str(YOLO_DATA_YAML),
        "--weights", weights,
        "--project", str(MODELS_DIR.resolve()),
        "--name", run_name,
        "--exist-ok",
    ]
    print("[Vision][Train] Running:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, cwd=str(Path.cwd()))
    except subprocess.CalledProcessError as e:
        print(f"[Vision][Train] YOLOv5 train failed: {e}")
        return 0
    print(f"[OK] YOLOv5 training complete. Weights under {MODELS_DIR}/{run_name}/weights")
    if not labels_dir.exists():
        alt_labels_dir = YOLOV5_DIR / "docs" / "vision_preds" / "labels"
    if alt_labels_dir.exists():
        labels_dir = alt_labels_dir


# -----------------------------
# Vision prediction (YOLOv5)
# -----------------------------
def predict_vision(args):
    # CHANGED: respect args.conf, args.iou, and optionally keep outputs unless --clear-outputs is set
    image_path = normalize_path(args.image)
    if not image_path or not image_path.exists():
        print(f"Image not found: {image_path}")
        return 0

    weights_path = MODELS_DIR / YOLO_PROJECT_NAME / "weights" / "best.pt"
    if not weights_path.exists():
        print(f"Trained weights not found at {weights_path}. Train vision first.")
        return 0

    # Use absolute project path so YOLO writes outputs where we expect them
    project_dir = Path("docs").resolve()
    out_dir = project_dir / "vision_preds"
    labels_dir = out_dir / "labels"

    # CHANGED: clear outputs only when requested
    if getattr(args, "clear_outputs", False) and out_dir.exists():
        shutil.rmtree(out_dir)

    # CHANGED: pick up thresholds from args with defaults
    conf_str = f"{getattr(args, 'conf', 0.25):.6f}"
    iou_str = f"{getattr(args, 'iou', 0.45):.6f}"

    cmd = [
        sys.executable, str(YOLOV5_DIR / "detect.py"),
        "--weights", str(weights_path),
        "--img", "640",
        "--conf", conf_str,
        "--iou", iou_str,  # CHANGED: pass IoU
        "--source", str(image_path),
        "--project", str(project_dir),
        "--name", "vision_preds",
        "--exist-ok",
        "--save-txt",
        "--save-conf",
    ]
    print("[Vision][Detect] Running:", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True, cwd=str(YOLOV5_DIR))
    except subprocess.CalledProcessError as e:
        print(f"[Vision][Detect] YOLOv5 detect failed: {e}")
        return 0

    print(f"[OK] Predictions expected under {out_dir}")

    # Count seams from YOLO label files (be robust: search recursively)
    seams_detected = 0
    if labels_dir.exists():
        files = list(labels_dir.glob("*.txt"))
        print(f"[Vision] Label files found: {len(files)} in {labels_dir}")
        for f in files:
            with open(f, "r") as fh:
                lines = sum(1 for _ in fh)
                seams_detected += lines
                print(f"[Vision] {f.name}: {lines} detections")
    else:
        # Try to find any labels file created elsewhere under the project dir (fallback)
        files = list(project_dir.rglob("labels/*.txt"))
        if files:
            print(f"[Vision] Found labels elsewhere: {len(files)} files")
            for f in files:
                with open(f, "r") as fh:
                    lines = sum(1 for _ in fh)
                    seams_detected += lines
                    print(f"[Vision] {Path(f).name}: {lines} detections")
        else:
            print(f"[Vision] No labels directory found at {labels_dir} or under {project_dir}")

    print(f"{seams_detected} seam{'s' if seams_detected != 1 else ''} detected")
    return seams_detected


# -----------------------------
# Sensor training
# -----------------------------
def train_sensor(args):
    Xw, yw = load_sensor_windows()

    if args.model == "rf":
        clf, report, cm = train_sensor_rf(Xw, yw)
        out = MODELS_DIR / "sensor_rf.joblib"
        dump(clf, out)
        print(report)
        plot_confusion_matrix(
            cm, classes=["0", "1"],
            title="Sensor RF Confusion",
            save_path="docs/confusion_rf.png",
            show=True,
        )

    elif args.model == "lstm":
        from src.models.sensor_lstm import SensorLSTM
        import numpy as np

        yw_np = np.array(yw)
        X_train, X_test, y_train, y_test = train_test_split(
            Xw, yw_np, test_size=0.2, random_state=RANDOM_STATE, stratify=yw_np
        )

        # Ensure shapes and dtypes are correct for PyTorch/CrossEntropyLoss
        X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

        # CrossEntropyLoss expects target of shape (N,) with dtype long
        y_train = torch.tensor(y_train, dtype=torch.long).squeeze().to(DEVICE)
        y_test = torch.tensor(y_test, dtype=torch.long).squeeze().to(DEVICE)

        # If squeeze removed too much dimensions, ensure 1D
        if y_train.dim() > 1:
            y_train = y_train.view(-1)
        if y_test.dim() > 1:
            y_test = y_test.view(-1)

        n_features = X_train.shape[2]
        model = SensorLSTM(n_features=n_features, n_classes=2).to(DEVICE)

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(model.parameters(), lr=SENSOR_LR)
        for epoch in range(1, SENSOR_EPOCHS + 1):
            model.train()
            optimizer.zero_grad()
            logits = model(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = logits.argmax(1)
                train_acc = (preds == y_train).float().mean().item()

                model.eval()
                val_logits = model(X_test)
                val_loss = criterion(val_logits, y_test).item()
                val_acc = (val_logits.argmax(1) == y_test).float().mean().item()

            print(
                f"[LSTM][Epoch {epoch}/{SENSOR_EPOCHS}] "
                f"loss={loss.item():.4f} acc={train_acc:.3f} | "
                f"val_loss={val_loss:.4f} acc={val_acc:.3f}"
            )

        out = MODELS_DIR / "sensor_lstm.pth"
        torch.save(model.state_dict(), out)
        print(f"[OK] Saved LSTM model: {out}")


# -----------------------------
# Sensor prediction
# -----------------------------
def predict_sensor(args):
    """Predict sensor status using RF or LSTM, with optional forced fail for testing and optional CSV input."""
    # CHANGED: Allow forcing a fail for testing
    if getattr(args, "force_fail", False):
        print("[Sensor] Forced fail mode enabled")
        return 1, 0.95

    # CHANGED: Support CSV input if provided; else fall back to existing window loader
    X_flat = None
    if getattr(args, "sensor_csv", None):
        csv_path = normalize_path(args.sensor_csv)
        if not csv_path or not csv_path.exists():
            print(f"[Sensor] CSV not found: {csv_path}")
            return 0, 0.0
        df = pd.read_csv(csv_path)
        if "Defect" in df.columns:
            df = df.drop(columns=["Defect"])
        # If multi-row, use last row; use all features
        if df.shape[0] > 1:
            df = df.tail(1)
        X_flat = df.values.reshape(1, -1)

    if args.model == "rf":
        rf_path = MODELS_DIR / "sensor_rf.joblib"
        if not rf_path.exists():
            print("RF model not found. Train first.")
            return 0, 0.0

        clf = load(rf_path)
        if X_flat is None:
            # Fallback to your existing loader for one window
            Xw, _ = load_sensor_windows()
            X_flat = Xw[:1].reshape(1, -1)

        pred = int(clf.predict(X_flat)[0])
        proba = clf.predict_proba(X_flat)[0]
        conf = float(proba[pred])
        print(f"RF prediction: {pred} | confidence: {conf:.3f}")
        return pred, conf

    elif args.model == "lstm":
        from src.models.sensor_lstm import SensorLSTM

        lstm_path = MODELS_DIR / "sensor_lstm.pth"
        if not lstm_path.exists():
            print("LSTM model not found. Train first.")
            return 0, 0.0

        if X_flat is None:
            # Fallback to one window from your loader
            Xw, _ = load_sensor_windows()
            X_seq = torch.tensor(Xw[:1], dtype=torch.float32).to(DEVICE)
        else:
            # CHANGED: reshape flat row to (batch=1, time=1, features=n)
            X_seq = torch.tensor(X_flat.reshape(1, 1, -1), dtype=torch.float32).to(DEVICE)

        model = SensorLSTM(n_features=X_seq.shape[2], n_classes=2).to(DEVICE)
        model.load_state_dict(torch.load(lstm_path, map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            logits = model(X_seq)
            pred = int(logits.argmax(1)[0].item())
            conf = float(torch.softmax(logits, dim=1).max().item())
        print(f"LSTM prediction: {pred} | confidence: {conf:.3f}")
        return pred, conf

    # Fallback
    return 0, 0.0


# -----------------------------
# Combined Pipeline prediction
# -----------------------------
def run_pipeline(args):
    print("=== Running End-to-End Pipeline ===")

    print("\n[Sensor Prediction]")
    sensor_label, sensor_conf = predict_sensor(args)

    print("\n[Vision Prediction]")
    seams_detected = predict_vision(args)

    print("\n[Final Verdict]")
    if sensor_label == 1 or seams_detected == 0:
        print(
            f"⚠️ Maintenance Required (Sensor={sensor_label} @ {sensor_conf:.2f}, Seams={seams_detected})"
        )
    else:
        print(
            f"✅ System Healthy (Sensor={sensor_label} @ {sensor_conf:.2f}, Seams={seams_detected})"
        )

    print("\n=== Pipeline Complete ===")


# -----------------------------
# CLI entrypoint
# -----------------------------
def main():
    p = argparse.ArgumentParser(description="Welding Maintenance Demo (YOLOv5 + RF/LSTM)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Vision training
    tv = sub.add_parser("train-vision", help="Train YOLOv5 vision model")
    tv.set_defaults(func=train_vision)

    # Sensor training
    ts = sub.add_parser("train-sensor", help="Train sensor model (RF or LSTM)")
    ts.add_argument("--model", choices=["rf", "lstm"], default="rf")
    ts.set_defaults(func=train_sensor)

    # Vision prediction
    pv = sub.add_parser("predict-vision", help="Run YOLOv5 vision inference on an image")
    pv.add_argument("--image", type=str, default=None, help="Path to image")
    # CHANGED: add thresholds and output control
    pv.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    pv.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    pv.add_argument("--clear-outputs", dest="clear_outputs", action="store_true", help="Clear previous YOLO outputs")
    pv.set_defaults(func=predict_vision)

    # Sensor prediction
    ps = sub.add_parser("predict-sensor", help="Run sensor prediction")
    ps.add_argument("--model", choices=["rf", "lstm"], default="rf")

    ps.add_argument("--sensor-csv", dest="sensor_csv", type=str, help="Path to sensor CSV file")
    ps.add_argument("--force-fail", dest="force_fail", action="store_true", help="Force sensor to fail for testing")
    ps.set_defaults(func=predict_sensor)

    # Combined pipeline
    pl = sub.add_parser("pipeline", help="Run combined sensor + vision pipeline")
    pl.add_argument("--model", choices=["rf", "lstm"], default="rf", help="Sensor model to use")
    pl.add_argument("--image", type=str, required=True, help="Path to test image for vision model")
    pl.add_argument("--sensor-csv", dest="sensor_csv", type=str, help="Path to sensor CSV file")
    pl.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    pl.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    pl.add_argument("--clear-outputs", dest="clear_outputs", action="store_true", help="Clear previous YOLO outputs")
    pl.add_argument("--force-fail", dest="force_fail", action="store_true", help="Force sensor to fail for testing")
    pl.set_defaults(func=run_pipeline)

    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()