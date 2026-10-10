
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config import MODELS_DIR, OUTPUTS_DIR

metric_files = {
    "CNN": "cnn_metrics.json",
    "MobileNetV2": "mobilenet_metrics.json",
    "Random Forest": "random_forest_metrics.json",
    "SVM": "svm_metrics.json",
}

rows = []

for model_name, filename in metric_files.items():
    path = MODELS_DIR / filename

    with open(path, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    # CNN uses different metric key names
    accuracy = metrics.get("accuracy", metrics.get("test_accuracy"))
    
    report = metrics.get("classification_report", {})
    macro_avg = report.get("macro avg", {})

    precision = metrics.get(
        "precision_macro",
        macro_avg.get("precision"),
    )
    recall = metrics.get(
        "recall_macro",
        macro_avg.get("recall"),
    )
    f1 = metrics.get("f1_macro", metrics.get("macro_f1"))
    training_time = metrics.get(
        "training_time_seconds",
        metrics.get("training_seconds"),
    )

    rows.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1": f1,
        "Training Time (seconds)": training_time,
    })

df = pd.DataFrame(rows)

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
graph_dir = OUTPUTS_DIR / "graphs"
graph_dir.mkdir(parents=True, exist_ok=True)

csv_path = OUTPUTS_DIR / "model_comparison.csv"
graph_path = graph_dir / "model_comparison.png"

df.to_csv(csv_path, index=False)

ax = df.plot(
    x="Model",
    y=["Accuracy", "Macro F1"],
    kind="bar",
    figsize=(10, 6),
    ylim=(0, 1),
    rot=0,
)

ax.set_title("FER2013 Emotion Classification Model Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.legend(loc="best")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(graph_path, dpi=150, bbox_inches="tight")
plt.close()

print("\nMODEL COMPARISON")
print("=" * 65)
print(df.round(4).to_string(index=False))
print(f"\nCSV saved to: {csv_path}")
print(f"Graph saved to: {graph_path}")