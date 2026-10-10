# AI-Based Interview Emotion & Confidence Analysis

## Abstract

This project implements a complete Python and deep-learning system for analyzing interview video frames, detecting facial emotions with a trained convolutional model, and estimating an observable behavioral confidence proxy. FER2013 provides the seven emotion classes for model training. Confidence is calculated separately from emotion detection using visible behavior indicators and is explicitly not a clinical or psychological measurement.

## Problem statement

Interview evaluators need a repeatable, computer-assisted way to inspect emotionally salient facial behavior. The system must distinguish emotion recognition from confidence estimation. A model can predict the probability of a facial emotion, but that probability does not establish the candidate's internal confidence. The project therefore tracks face visibility, eye-contact proxy, head stability, expression stability, emotion stability, and positive/neutral expression proportion, then converts these factors into a transparent 0–100 behavioral score.

## Objectives

- Train and compare CNN, SVM+HOG+PCA, Random Forest+HOG, and MobileNetV2 transfer learning on FER2013.
- Evaluate all models with measured accuracy, precision, recall, macro F1, confusion matrix, training time, and inference time where practical.
- Save real trained artifacts and model metadata.
- Analyze uploaded MP4/AVI/MOV interview video without requiring a face in every frame.
- Provide a Streamlit dashboard and downloadable JSON/HTML reports.
- Keep emotion probability and behavioral confidence conceptually separate.

## System architecture

1. Dataset audit validates the seven class directories and counts readable images.
2. Preprocessing creates normalized image tensors and reusable augmentation.
3. Model training runs four separate approaches and writes evaluation results.
4. Model comparison reads the saved metrics and produces tables, confusion matrices, and graphs.
5. Video processing detects faces, predicts emotions, extracts behavioral indicators, and aggregates the score.
6. Report generation writes JSON and HTML artifacts.
7. Streamlit presents the dataset, model, analysis, comparison, and report views.

## Technologies

Python 3.10 or 3.11, TensorFlow/Keras, OpenCV, NumPy, Pandas, scikit-learn, scikit-image, Matplotlib, Seaborn, Pillow, tqdm, Streamlit, MediaPipe optional, and joblib.

## Dataset

FER2013 has seven classes: angry, disgust, fear, happy, neutral, sad, and surprise. It is not a confidence-labeled dataset. The expected structure is:

- data/fer2013/train/{class}/
- data/fer2013/test/{class}/

Place the official FER2013 images in these folders. The dataset audit verifies all seven class folders in both splits, counts supported image files
separately for training and test, and reports missing or empty class folders.

## Installation

1. Create a Python 3.10/3.11 virtual environment.
2. Install dependencies: `python -m pip install -r requirements.txt`.
3. Install TensorFlow for the selected environment using the version supported by the target platform.
4. Install the optional MediaPipe package from the same environment if landmark-based eye/head analysis is desired.
5. In VS Code, select the environment interpreter and open the project root.

## How to run

- Dataset audit: `python src/dataset_audit.py --data_dir data/fer2013`
- CNN: `python src/train_cnn.py --data_dir data/fer2013 --epochs 50`
- SVM: `python src/train_svm.py --data_dir data/fer2013`
- Random Forest: `python src/train_random_forest.py --data_dir data/fer2013`
- MobileNetV2: `python src/train_mobilenet.py --data_dir data/fer2013 --epochs 15`
- Compare models: `python src/compare_models.py --data_dir data/fer2013`
- Analyze video: `python src/interview_analyzer.py --video interview_videos/interview.mp4`
- Run UI: `streamlit run app.py`

The application should not train or analyze if the required dataset or model file is missing. User-friendly messages are logged and displayed instead of raw tracebacks.

## Expected outputs

- models/cnn_emotion.keras, cnn_emotion_best.keras, cnn_metrics.json
- models/svm_hog_pca.joblib, svm_metrics.json
- models/random_forest_hog.joblib, random_forest_metrics.json
- models/mobilenetv2_emotion.keras, class_names.json, model_metadata.json, mobilenet_metrics.json
- outputs/model_comparison.csv
- outputs/graphs/model_comparison.png and confusion matrices
- outputs/interview_reports/interview_report.json and .html
- logs/{module}.log

## Model selection and justification

Selection is based on measured results, not a predetermined preference. The comparison considers macro F1, accuracy, generalization, training cost, inference speed, model size, suitability for video processing, and deployment feasibility. MobileNetV2 is proposed as the final model only if its measured metrics and operational characteristics justify it. If a different model performs better, the report must state the actual result and the experimental rationale.

## Limitations

- Emotion recognition is not perfect.
- Facial expressions do not necessarily represent internal emotional states.
- FER2013 contains no interview-confidence labels.
- The confidence score is a behavioral proxy and cannot measure psychological confidence.
- Lighting, camera angle, cultural expression, occlusion, video quality, and candidate movement affect results.
- The default face detector is practical, not medically or scientifically validated.

## Ethical considerations

Use only with human review. The system must not be used for hiring decisions. Results may contain false positives and false negatives. Candidates should be informed of the system's purpose, limitations, and data handling. Do not use the score as a clinical assessment.

## Future scope

- Add MediaPipe landmarks, gaze direction, and head-pose estimation.
- Add temporal modeling and per-frame tracking.
- Validate on diverse, consented interview datasets.
- Add uncertainty calibration and per-class metrics.
- Add privacy-preserving retention and data governance.
- Build an explainable interface for evaluator review.
