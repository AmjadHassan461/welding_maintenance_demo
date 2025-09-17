import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from src.config import RANDOM_STATE, SENSOR_TEST_SIZE

def _flatten_windows(Xw):
    # Supports both [N, W, F] and [N, 1, F] (tabular wrapped as 1-step window)
    return Xw.reshape(Xw.shape[0], -1)

def train_sensor_rf(Xw, yw):
    X = _flatten_windows(Xw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, yw, test_size=SENSOR_TEST_SIZE, random_state=RANDOM_STATE, stratify=yw
    )

    classes = np.unique(y_train)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    ypred = clf.predict(X_test)

    report = classification_report(y_test, ypred, digits=3)
    cm = confusion_matrix(y_test, ypred, labels=classes)
    return clf, report, cm
