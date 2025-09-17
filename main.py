import argparse
from pathlib import Path
import subprocess
import shutil
import glob
import sys, os
from torch.optim.adam import Adam


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
        subprocess.run(cmd, check=True, cwd=str(YOLOV5_DIR))
    except subprocess.CalledProcessError as e:
        print(f"[Vision][Train] YOLOv5 train failed: {e}")
        return
    print(f"[OK] YOLOv5 training complete. Weights under {MODELS_DIR}/{run_name}/weights")


# -----------------------------
# Vision prediction (YOLOv5)
# -----------------------------
def predict_vision(args):
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

    # Clear the previous run entirely (avoid stale detections)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    cmd = [
        sys.executable, str(YOLOV5_DIR / "detect.py"),
        "--weights", str(weights_path),
        "--img", "640",
        "--conf", "0.25",
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
    """Predict sensor status using RF or LSTM, with optional forced fail for testing."""
    # Allow forcing a fail for testing Case 3/4
    if getattr(args, "force_fail", False):
        print("[Sensor] Forced fail mode enabled")
        return 1, 0.95

    Xw, _ = load_sensor_windows()

    if args.model == "rf":
        rf_path = MODELS_DIR / "sensor_rf.joblib"
        if not rf_path.exists():
            print("RF model not found. Train first.")
            return 0, 0.0

        clf = load(rf_path)
        X = Xw[:1].reshape(1, -1)
        pred = int(clf.predict(X)[0])
        proba = clf.predict_proba(X)[0]
        conf = float(proba[pred])
        print(f"RF prediction: {pred} | confidence: {conf:.3f}")
        return pred, conf

    elif args.model == "lstm":
        from src.models.sensor_lstm import SensorLSTM

        lstm_path = MODELS_DIR / "sensor_lstm.pth"
        if not lstm_path.exists():
            print("LSTM model not found. Train first.")
            return 0, 0.0

        X = torch.tensor(Xw[:1], dtype=torch.float32).to(DEVICE)
        model = SensorLSTM(n_features=X.shape[2], n_classes=2).to(DEVICE)
        model.load_state_dict(torch.load(lstm_path, map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            logits = model(X)
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
    pv.set_defaults(func=predict_vision)

    # Sensor prediction
    ps = sub.add_parser("predict-sensor", help="Run sensor prediction")
    ps.add_argument("--model", choices=["rf", "lstm"], default="rf")
    ps.set_defaults(func=predict_sensor)
    ps.add_argument("--force-fail", action="store_true", help="Force sensor to fail for testing")

    # Combined pipeline
    pl = sub.add_parser("pipeline", help="Run combined sensor + vision pipeline")
    pl.add_argument("--model", choices=["rf", "lstm"], default="rf", help="Sensor model to use")
    pl.add_argument("--image", type=str, required=True, help="Path to test image for vision model")
    pl.set_defaults(func=run_pipeline)
    pl.add_argument("--force-fail", action="store_true", help="Force sensor to fail for testing")

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
