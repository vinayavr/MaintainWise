# MaintainWise — Predictive Maintenance of Industrial Machines

An end-to-end CPPE project using the AI4I 2020 dataset, scikit-learn/XGBoost, MLflow, Gradio and Hugging Face Spaces.

## Project structure
- `data/raw/`: place the downloaded AI4I CSV here
- `data/processed/`: cleaned CSV and Parquet outputs
- `artifacts/`: metrics, plots and saved model pipeline
- `src/data_pipeline.py`: data loading, validation, cleaning and feature engineering
- `src/train.py`: five classifiers, evaluation and MLflow logging
- `app.py`: Gradio app for failure-risk prediction
- `tests/test_pipeline.py`: basic tests
- `report/`: report outline and Jira story plan

## Quick start (Windows PowerShell)
1. Install Python 3.11 or 3.12.
2. Open this folder in VS Code.
3. Create and activate a virtual environment:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
4. Install packages:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Download the AI4I 2020 CSV from the UCI dataset page:
   https://archive.ics.uci.edu/dataset/601/ai4i+2020+
6. Put the CSV in `data/raw/` and rename it `ai4i2020.csv`.
7. Run:
   ```powershell
   python -m src.data_pipeline
   python -m src.train
   ```
8. In a second terminal, start MLflow:
   ```powershell
   mlflow ui --host 127.0.0.1 --port 5000
   ```
   Open http://127.0.0.1:5000
9. Run the app locally:
   ```powershell
   python app.py
   ```

## Important modeling choices
- Target: `Machine failure` (binary classification).
- Failure-mode labels (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) are excluded from features to prevent leakage.
- `UDI` and `Product ID` are excluded from model inputs.
- The split is stratified. Scaling and SMOTE are fit only on training data using an imbalanced-learn pipeline.
- The app loads the full saved preprocessing/model pipeline, so inference uses the same transformations as training.
- Model metrics in the report must be copied from your actual MLflow runs. Do not invent scores.
