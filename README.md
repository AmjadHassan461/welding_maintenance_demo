# Predictive Maintenance for Robotic Welding Arm

This project implements a predictive maintenance system for robotic welding arms.  
It combines **sensor data analysis** (Random Forest, LSTM) with **computer vision seam detection** (YOLOv5) to predict when maintenance is required.  
The system includes both a **CLI pipeline** and a **Streamlit web UI** for operators.

---

## 👥 Team Members

| AC.NO     | Name                     | Role            | Contributions |
|-----------|--------------------------|-----------------|---------------|
| 202274070 | Amjad Hassan             | Lead Developer  | Project architecture, pipeline integration, Streamlit UI |
| 202274031 | Ali Al-Quladi            | Vision Training | YOLOv5 training, seam detection integration |
| 202274093 | Hashim Nabeel Al-Motwakel| Sensor Training | Sensor dataset simulation, RF/LSTM models |
| 202274089 | Yousef Al-Noaimi         | Data Analysis  | Data preprocessing, EDA, visualization |
| 202274037 | Riyadh Al-Kibsi          | ML Models   | Model optimization, testing, deployment setup |

---

## ⚙️ Installation and Setup

### Prerequisites
- Python 3.12 or higher
- [UV package manager](https://docs.astral.sh/uv/)

### Steps
```bash
# Clone repository
git clone <https://github.com/AmjadHassan461/welding_maintenance_demo/tree/master3.0>
cd welding_maintenance_demo

# Install dependencies
uv sync

# Run CLI
uv run python main.py --help

# Run Streamlit UI
uv run streamlit run app.py

# CLI Examples
# Sensor only
uv run python main.py predict-sensor --model rf --sensor-csv data/welding_dataset.csv

# Vision only
uv run python main.py predict-vision --image data/test.jpg

# Full pipeline
uv run python main.py pipeline --model rf \
  --sensor-csv data/welding_dataset.csv \
  --image data/test.jpg


