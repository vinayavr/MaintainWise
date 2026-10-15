
from pathlib import Path

import json
import warnings

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "model.pkl"
COMPARISON_PATH = ARTIFACTS / "model_comparison.csv"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Trained model not found. Run `python -m src.train` first."
    )

# Load only your own trusted model artifact.
model = joblib.load(MODEL_PATH)

comparison = None
if COMPARISON_PATH.exists():
    comparison = pd.read_csv(COMPARISON_PATH)

# AI4I 2020 input fields. ID columns and failure-mode labels
# must not be provided as prediction inputs.
INPUT_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def make_input(product_type, air_temp, process_temp, rpm, torque, tool_wear):
    """Create the exact input features expected by the trained model."""

    air_temp = float(air_temp)
    process_temp = float(process_temp)
    rpm = float(rpm)
    torque = float(torque)
    tool_wear = float(tool_wear)

    temperature_difference = process_temp - air_temp
    mechanical_power = torque * rpm * (2 * np.pi / 60)

    return pd.DataFrame([{
        "Type": product_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
        "Temperature difference [K]": temperature_difference,
        "Estimated mechanical power [W]": mechanical_power,
    }])



def predict_failure(product_type, air_temp, process_temp, rpm, torque, tool_wear):
    values = [air_temp, process_temp, rpm, torque, tool_wear]
    if not all(np.isfinite(float(v)) for v in values):
        raise gr.Error("Please enter finite numeric sensor readings.")

    if not (295 <= air_temp <= 305):
        raise gr.Error("Air temperature must be between 295 and 305 K.")
    if not (305 <= process_temp <= 315):
        raise gr.Error("Process temperature must be between 305 and 315 K.")
    if not (1000 <= rpm <= 3000):
        raise gr.Error("Rotational speed must be between 1000 and 3000 rpm.")
    if not (3 <= torque <= 80):
        raise gr.Error("Torque must be between 3 and 80 Nm.")
    if not (0 <= tool_wear <= 250):
        raise gr.Error("Tool wear must be between 0 and 250 minutes.")

    row = make_input(
        product_type, air_temp, process_temp, rpm, torque, tool_wear
    )

    try:
        prediction = int(model.predict(row)[0])
    except Exception as exc:
        raise gr.Error(
            "The model could not process these inputs. "
            "Check that the model pipeline and input columns match training. "
            f"Technical detail: {exc}"
        )

    # Do not silently invent a probability if the model lacks predict_proba.
    if not hasattr(model, "predict_proba"):
        probability_text = (
            "Unavailable: this model does not expose predict_proba()."
        )
    else:
        probabilities = np.asarray(model.predict_proba(row))[0]
        classes = list(model.classes_)
        if 1 not in classes:
            probability_text = (
                "Unavailable: positive failure class not found in model."
            )
        else:
            failure_probability = float(probabilities[classes.index(1)])
            probability_text = f"{failure_probability:.2%}"

    classification = (
        "Machine failure predicted"
        if prediction == 1
        else "No failure predicted"
    )

    note = (
        "This is a model estimate based on synthetic AI4I 2020 data. "
        "It is not a calibrated real-world risk guarantee or an automatic "
        "maintenance decision. Consult a qualified technician."
    )

    return classification, probability_text, note


def load_comparison():
    if comparison is None:
        return pd.DataFrame({
            "Status": ["Model comparison CSV not found"],
            "Action": ["Run python -m src.train"],
        })

    # Show the actual values saved by the training script.
    return comparison.copy()



def confusion_matrix_figure():
    candidates = [
        ARTIFACTS / "confusion_matrix_RandomForest.png",
        ARTIFACTS / "confusion_matrix.png",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def feature_importance_figure():
    candidates = [
        ARTIFACTS / "feature_importance_RandomForest.png",
        ARTIFACTS / "feature_importance.png",
        ARTIFACTS / "feature_importance.jpg",
        ARTIFACTS / "feature_importance.svg",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return None



with gr.Blocks(title="MaintainWise | Predictive Maintenance") as demo:
    gr.Markdown(
        """
        # MaintainWise
        ### Predictive maintenance decision-support prototype

        Estimate the probability of machine failure from sensor readings.
        The model was trained on the **synthetic AI4I 2020 dataset**.
        It is an academic demonstration, not a safety-critical system.
        """
    )

    with gr.Tab("Predict"):
        gr.Markdown("### Enter sensor readings")
        with gr.Row():
            product_type = gr.Dropdown(
                choices=["L", "M", "H"],
                value="M",
                label="Product type",
            )
            air_temp = gr.Number(
                value=300.0, label="Air temperature (K)"
            )
            process_temp = gr.Number(
                value=310.0, label="Process temperature (K)"
            )

        with gr.Row():
            rpm = gr.Number(value=1500, label="Rotational speed (rpm)")
            torque = gr.Number(value=40, label="Torque (Nm)")
            tool_wear = gr.Number(value=100, label="Tool wear (min)")

        predict_button = gr.Button("Predict failure risk", variant="primary")

        classification = gr.Textbox(
            label="Model classification", interactive=False
        )
        probability = gr.Textbox(
            label="Estimated failure probability", interactive=False
        )
        note = gr.Textbox(label="Decision-support note", interactive=False)

        predict_button.click(
            fn=predict_failure,
            inputs=[
                product_type, air_temp, process_temp, rpm, torque, tool_wear
            ],
            outputs=[classification, probability, note],
        )

        gr.Markdown(
            """
            **Interpretation:** A predicted class is the model's output at
            its configured decision threshold. A probability is a model
            score and should not be assumed to be calibrated.
            """
        )

    with gr.Tab("Model comparison"):
        gr.Markdown(
            "### Results from your actual training run"
        )
        gr.Dataframe(
            value=load_comparison(),
            label="Saved model-comparison metrics",
            interactive=False,
        )
        gr.Markdown(
            """
            Accuracy can be misleading for imbalanced failure data.
            Consider failure-class precision, recall, F1, and ROC-AUC
            together when interpreting the results.
            """
        )

    with gr.Tab("Evaluation evidence"):
        gr.Markdown("### Confusion matrix")
        cm_path = confusion_matrix_figure()
        if cm_path:
            gr.Image(value=cm_path, label="Saved confusion matrix")
        else:
            gr.Markdown(
                "No confusion-matrix image was found in artifacts/. "
                "Generate it from the held-out test predictions."
            )

        gr.Markdown("### Feature importance")
        fi_path = feature_importance_figure()
        if fi_path:
            gr.Image(value=fi_path, label="Saved feature importance")
        else:
            gr.Markdown(
                "No feature-importance image was found. "
                "Feature importance depends on the selected estimator "
                "and the preprocessing pipeline."
            )

    with gr.Tab("About and limitations"):
        gr.Markdown(
            """
            ### Dataset and limitations

            - Dataset: AI4I 2020 Predictive Maintenance Dataset.
            - The data is synthetic and does not establish real-world
              machine-failure performance.
            - Predictions depend on the training data, preprocessing,
              model version, and decision threshold.
            - This application does not connect to live industrial sensors.
            - Do not use the output as the sole basis for safety,
              maintenance, or operational decisions.

            ### Model governance

            The training script logs experiments and registers a selected
            model with MLflow. The application currently loads the local
            `artifacts/model.pkl` file; this is not itself proof that the
            deployed app is serving the latest registered model version.
            """
        )

if __name__ == "__main__":
        demo.launch(server_name="127.0.0.1", server_port=7860)
