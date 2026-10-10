
"""Streamlit UI for AI-Based Interview Emotion & Behavioral Analysis."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    DATA_DIR,
    EMOTION_CLASSES,
    GRAPHS_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    REPORTS_DIR,
)
from src.dataset_audit import audit_dataset
from src.interview_analyzer import InterviewAnalyzer
from src.model_utils import configure_logging


logger = configure_logging("app")

st.set_page_config(
    page_title="Interview Emotion & Behavioral Analysis",
    page_icon="🧑‍💼",
    layout="wide",
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def st_show_ethical_notice() -> None:
    """Display limitations of automated facial analysis."""

    st.warning(
        "Emotion recognition is imperfect. Facial expressions do not "
        "necessarily reveal internal emotional states. The behavioral "
        "proxy score is experimental and is not a validated measure "
        "of psychological confidence, competence, or job suitability. "
        "Do not use these results alone for hiring decisions."
    )


def load_comparison() -> pd.DataFrame:
    """Load model comparison results if available."""

    path = OUTPUTS_DIR / "model_comparison.csv"

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        logger.exception("Could not read model comparison CSV")
        st.error(f"Could not load model comparison: {error}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def dataset_summary() -> dict:
    """Audit the FER2013 dataset."""

    try:
        return audit_dataset(DATA_DIR, GRAPHS_DIR)
    except (FileNotFoundError, ValueError, OSError) as error:
        return {
            "error": str(error),
            "total_images": 0,
            "class_counts": {},
        }


def save_report(report: dict, video_name: str) -> Path:
    """Save analysis output as JSON."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = (
        REPORTS_DIR / f"{Path(video_name).stem}_report.json"
    )

    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return report_path


def metric_value(report: dict, key: str, default=None):
    """Safely retrieve a report value."""

    return report.get(key, default)


def show_interview_analysis() -> None:
    """Render video upload, progress, and analysis results."""

    st.header("Interview analysis")
    st_show_ethical_notice()

    uploaded = st.file_uploader(
        "Upload an interview video",
        type=["mp4", "avi", "mov"],
        help=(
            "For initial testing, use a short video with a clearly "
            "visible face and good lighting."
        ),
        key="interview_video_upload",
    )

    if uploaded is not None:
        size_mb = uploaded.size / (1024 * 1024)

        st.caption(
            f"Selected file: {uploaded.name} · {size_mb:.1f} MB"
        )

        if size_mb > 200:
            st.error(
                "The video is larger than the recommended 200 MB limit. "
                "Please choose a smaller video."
            )
            return

    if not st.button(
        "Analyze Interview",
        type="primary",
        disabled=uploaded is None,
    ):
        return

    video_path: Path | None = None

    try:
        suffix = Path(uploaded.name).suffix.lower() or ".mp4"

        # Save uploaded video temporarily for OpenCV.
        with tempfile.NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temporary:
            temporary.write(uploaded.getvalue())
            video_path = Path(temporary.name)

        status = st.empty()
        progress_bar = st.progress(0)
        progress_text = st.empty()

        status.info("Loading the emotion model. Please wait...")

        analyzer = InterviewAnalyzer(video_path)

        def update_progress(
            current_frame: int,
            total_frames: int,
            analyzed_frames: int,
        ) -> None:
            if total_frames > 0:
                percent = min(
                    100,
                    int(current_frame * 100 / total_frames),
                )
            else:
                percent = 0

            progress_bar.progress(percent)

            if total_frames > 0:
                progress_text.text(
                    f"Video progress: {current_frame:,} / "
                    f"{total_frames:,} frames read · "
                    f"{analyzed_frames:,} faces analyzed"
                )
            else:
                progress_text.text(
                    f"{current_frame:,} frames read · "
                    f"{analyzed_frames:,} faces analyzed"
                )

            status.info("Analyzing sampled video frames...")

        # Requires the updated InterviewAnalyzer.
        report = analyzer.analyze(
            frame_interval=10,
            progress_callback=update_progress,
        )

        progress_bar.progress(100)
        status.success("Interview analysis completed!")

        # Store the original filename instead of a temporary path.
        report["video"] = Path(uploaded.name).name

        report_path = save_report(report, uploaded.name)

        # -------------------------------------------------
        # Analysis results
        # -------------------------------------------------

        st.subheader("Analysis results")

        score = metric_value(report, "behavioral_proxy_score")
        quality = metric_value(report, "face_image_quality_proxy")
        dominant_emotion = metric_value(
            report,
            "dominant_emotion",
            "Not available",
        )

        col1, col2, col3 = st.columns(3)

        score_display = (
            f"{score:.2f}/100"
            if isinstance(score, (int, float))
            else "Not available"
        )

        col1.metric(
            "Behavioral proxy score",
            score_display,
            help=(
                "An experimental summary of observable video indicators. "
                "It is not a validated measure of psychological confidence."
            ),
        )

        quality_display = (
            f"{quality:.2f}/100"
            if isinstance(quality, (int, float))
            else "Not available"
        )

        col2.metric(
            "Face image quality proxy",
            quality_display,
            help=(
                "An image-quality estimate from analyzed frames. "
                "It is not the percentage of frames containing a face."
            ),
        )

        col3.metric(
            "Dominant emotion",
            str(dominant_emotion).title(),
        )

        # -------------------------------------------------
        # Video summary
        # -------------------------------------------------

        st.subheader("Video summary")

        summary_col1, summary_col2, summary_col3 = st.columns(3)

        duration = metric_value(report, "duration_seconds")
        analyzed_frames = metric_value(report, "analyzed_frames")
        frames_without_face = metric_value(
            report,
            "frames_without_face",
        )

        summary_col1.metric(
            "Duration",
            (
                f"{duration:.1f} sec"
                if isinstance(duration, (int, float))
                else "Not available"
            ),
        )

        summary_col2.metric(
            "Frames analyzed",
            (
                f"{analyzed_frames:,}"
                if isinstance(analyzed_frames, (int, float))
                else "Not available"
            ),
        )

        summary_col3.metric(
            "Sampled frames without face",
            (
                f"{frames_without_face:,}"
                if isinstance(frames_without_face, (int, float))
                else "Not available"
            ),
        )

        # -------------------------------------------------
        # Emotion distribution
        # -------------------------------------------------

        st.subheader("Emotion distribution")

        emotion_distribution = metric_value(
            report,
            "emotion_distribution",
            {},
        )

        if isinstance(emotion_distribution, dict) and emotion_distribution:
            distribution = pd.DataFrame(
                [
                    {
                        "Emotion": emotion.title(),
                        "Percent": percent,
                    }
                    for emotion, percent
                    in emotion_distribution.items()
                ]
            )

            distribution["Percent"] = pd.to_numeric(
                distribution["Percent"],
                errors="coerce",
            )
            distribution = distribution.dropna(
                subset=["Percent"]
            )

            st.dataframe(
                distribution,
                use_container_width=True,
                hide_index=True,
            )

            if not distribution.empty:
                st.bar_chart(
                    distribution.set_index("Emotion")["Percent"]
                )
        else:
            st.info("No emotion distribution is available.")

        # -------------------------------------------------
        # Emotion timeline graph
        # -------------------------------------------------

        st.subheader("Emotion timeline")

        timeline = metric_value(
            report,
            "emotion_timeline",
            [],
        )

        if isinstance(timeline, list) and timeline:
            timeline_df = pd.DataFrame(timeline)

            required_columns = {"frame", "emotion", "probability"}
            if required_columns.issubset(timeline_df.columns):
                fps = metric_value(report, "fps", 0)
                fps = (
                    float(fps)
                    if isinstance(fps, (int, float)) and fps > 0
                    else 0
                )

                timeline_df["frame"] = pd.to_numeric(
                    timeline_df["frame"],
                    errors="coerce",
                )
                timeline_df["probability"] = pd.to_numeric(
                    timeline_df["probability"],
                    errors="coerce",
                )

                timeline_df = timeline_df.dropna(
                    subset=["frame", "probability", "emotion"]
                )

                if fps > 0:
                    timeline_df["Time (seconds)"] = (
                        timeline_df["frame"] - 1
                    ) / fps
                    timeline_df["Time (seconds)"] = (
                        timeline_df["Time (seconds)"].round(2)
                    )
                    x_column = "Time (seconds)"
                else:
                    x_column = "frame"

                timeline_df["Probability (%)"] = (
                    timeline_df["probability"] * 100
                )

                if not timeline_df.empty:
                    timeline_chart = px.line(
                        timeline_df,
                        x=x_column,
                        y="Probability (%)",
                        color="emotion",
                        markers=True,
                        title=(
                            "Predicted emotion probability "
                            "over analyzed video frames"
                        ),
                        labels={
                            "emotion": "Predicted emotion",
                            x_column: x_column,
                        },
                    )

                    timeline_chart.update_layout(
                        legend_title_text="Predicted emotion",
                        yaxis_range=[0, 100],
                    )

                    st.plotly_chart(
                        timeline_chart,
                        use_container_width=True,
                    )

                    st.caption(
                        "Only analyzed frames are shown. Probabilities "
                        "represent model outputs, not proof of a person's "
                        "actual emotional state."
                    )

                    st.dataframe(
                        timeline_df[
                            [
                                x_column,
                                "frame",
                                "emotion",
                                "Probability (%)",
                            ]
                        ].rename(
                            columns={
                                "frame": "Frame number",
                                "emotion": "Predicted emotion",
                            }
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info(
                        "The emotion timeline contains no valid data points."
                    )
            else:
                st.info(
                    "The emotion timeline is missing expected fields: "
                    "frame, emotion, or probability."
                )
        else:
            st.info("No emotion timeline data is available.")

        # -------------------------------------------------
        # Observable behavioral indicators
        # -------------------------------------------------

        st.subheader("Observable behavioral indicators")

        indicator_fields = [
            (
                "Face image quality proxy",
                "face_image_quality_proxy",
            ),
            (
                "Head stability proxy",
                "head_stability_proxy",
            ),
            (
                "Emotion stability proxy",
                "emotion_stability_proxy",
            ),
        ]

        indicator_data = []

        for label, field in indicator_fields:
            value = metric_value(report, field)

            indicator_data.append(
                {
                    "Indicator": label,
                    "Score (0–100)": (
                        round(value, 2)
                        if isinstance(value, (int, float))
                        else "Not available"
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(indicator_data),
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "These are experimental video-derived proxies. They do not "
            "establish eye contact, actual confidence, competence, or "
            "internal emotional states."
        )

        # -------------------------------------------------
        # Interpretation
        # -------------------------------------------------

        st.subheader("Interpretation")

        interpretation = metric_value(
            report,
            "overall_interpretation",
            "No interpretation is available.",
        )

        warning = metric_value(
            report,
            "warning",
            (
                "Facial expressions and predicted emotion labels do not "
                "necessarily reveal internal emotional states."
            ),
        )

        st.write(interpretation)
        st.info(warning)

        # -------------------------------------------------
        # Download report
        # -------------------------------------------------

        report_json = json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )

        st.download_button(
            "Download JSON report",
            data=report_json,
            file_name=f"{Path(uploaded.name).stem}_report.json",
            mime="application/json",
        )

        st.caption(f"Report saved at: {report_path}")

    except Exception as error:
        logger.exception("Interview analysis failed")
        st.error(f"Interview analysis failed: {error}")
        st.info(
            "Check the VS Code terminal for the full traceback. "
            "Confirm that src/interview_analyzer.py supports "
            "frame_interval and progress_callback."
        )

    finally:
        if video_path is not None:
            try:
                video_path.unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "Could not remove temporary video: %s",
                    video_path,
                )


# ---------------------------------------------------------
# Main application
# ---------------------------------------------------------

st.title("AI-Based Interview Emotion & Behavioral Analysis")

st.caption(
    "FER2013 emotion recognition and experimental behavioral "
    "indicator analysis."
)

st_show_ethical_notice()

nav = st.sidebar.radio(
    "Section",
    [
        "Home",
        "Dataset",
        "Model",
        "Interview Analysis",
        "Model Comparison",
        "Report",
    ],
)


# ---------------------------------------------------------
# Home
# ---------------------------------------------------------

if nav == "Home":
    st.header("Project overview")

    st.markdown(
        """
        This application combines facial emotion classification with
        experimental, observable indicators derived from interview video.

        - **Dataset:** FER2013 facial expression dataset.
        - **Emotion model:** trained model for seven emotion classes.
        - **Interview analysis:** sampled video frames and face analysis.
        - **Emotion timeline:** predicted emotion probabilities over time.
        - **Model comparison:** measured experiment results.
        - **Reports:** saved JSON analysis reports.

        The application estimates facial-expression categories. It does
        not determine a person's true emotions, personality, or suitability
        for employment.
        """
    )

    st.info(
        "Model predictions and behavioral proxies should not be treated "
        "as ground truth about a person's internal emotional state or "
        "suitability for a job."
    )


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

elif nav == "Dataset":
    st.header("Dataset status")

    summary = dataset_summary()

    if "error" in summary:
        st.error(summary["error"])
    else:
        st.success(
            f"Dataset contains {summary.get('total_images', 0):,} "
            "images across the training and test splits."
        )

        train_counts = summary.get("train_class_counts", {})
        test_counts = summary.get("test_class_counts", {})

        counts = pd.DataFrame(
            {
                "Class": EMOTION_CLASSES,
                "Training images": [
                    train_counts.get(name, 0)
                    for name in EMOTION_CLASSES
                ],
                "Test images": [
                    test_counts.get(name, 0)
                    for name in EMOTION_CLASSES
                ],
            }
        )

        st.dataframe(
            counts,
            use_container_width=True,
            hide_index=True,
        )

        graph_path = GRAPHS_DIR / "class_distribution.png"

        if graph_path.exists():
            st.image(
                str(graph_path),
                caption="FER2013 training and test class distribution",
            )


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

elif nav == "Model":
    st.header("Model and training status")

    model_path = MODELS_DIR / "mobilenetv2_emotion.keras"
    metadata_path = MODELS_DIR / "model_metadata.json"

    st.subheader("MobileNetV2 architecture")

    st.markdown(
        """
        - MobileNetV2-based emotion classification
        - 224 × 224 RGB input
        - 128-unit dense layer
        - Dropout: 0.25
        - Seven emotion classes
        """
    )

    st.write("Model file:", str(model_path))
    st.write("Model exists:", model_path.exists())

    if metadata_path.exists():
        try:
            metadata = json.loads(
                metadata_path.read_text(encoding="utf-8")
            )
            st.json(metadata)
        except (OSError, json.JSONDecodeError) as error:
            st.error(f"Could not read model metadata: {error}")
    else:
        st.info(
            "No model_metadata.json file was found. "
            "The trained model file may still be available."
        )


# ---------------------------------------------------------
# Interview Analysis
# ---------------------------------------------------------

elif nav == "Interview Analysis":
    show_interview_analysis()


# ---------------------------------------------------------
# Model Comparison
# ---------------------------------------------------------

elif nav == "Model Comparison":
    st.header("Model experiment comparison")

    comparison = load_comparison()

    if comparison.empty:
        st.warning(
            "No comparison results found. Check "
            "outputs/model_comparison.csv."
        )
    else:
        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )

        column_lookup = {
            str(column).strip().lower().replace("_", " "): column
            for column in comparison.columns
        }

        model_column = (
            column_lookup.get("model")
            or column_lookup.get("name")
        )

        f1_column = (
            column_lookup.get("macro f1")
            or column_lookup.get("f1")
        )

        if model_column and f1_column:
            chart_data = comparison[
                [model_column, f1_column]
            ].copy()

            chart_data.columns = ["Model", "Macro F1"]
            chart_data["Macro F1"] = pd.to_numeric(
                chart_data["Macro F1"],
                errors="coerce",
            )
            chart_data = chart_data.dropna(
                subset=["Macro F1"]
            )

            if not chart_data.empty:
                chart = px.bar(
                    chart_data,
                    x="Model",
                    y="Macro F1",
                    color="Model",
                    title="Measured model macro F1",
                )

                st.plotly_chart(
                    chart,
                    use_container_width=True,
                )
        else:
            st.info(
                "The comparison CSV does not contain recognizable "
                "model and F1 columns, so the chart cannot be drawn."
            )

        st.caption(
            "Metrics are read from the saved experiment comparison CSV."
        )


# ---------------------------------------------------------
# Report
# ---------------------------------------------------------

elif nav == "Report":
    st.header("Report management")

    st.info(
        "Reports contain model predictions and experimental behavioral "
        "indicators, not psychological or clinical assessments."
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    reports = sorted(
        REPORTS_DIR.glob("*_report.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if reports:
        selected = st.selectbox(
            "Choose report",
            reports,
            format_func=lambda path: path.name,
        )

        try:
            report_data = json.loads(
                selected.read_text(encoding="utf-8")
            )

            st.json(report_data)

            st.download_button(
                "Download selected JSON report",
                data=selected.read_bytes(),
                file_name=selected.name,
                mime="application/json",
            )
        except (OSError, json.JSONDecodeError) as error:
            st.error(f"Could not load report: {error}")
    else:
        st.warning(
            "No reports are available yet. Analyze an interview video "
            "to generate your first report."
        )
