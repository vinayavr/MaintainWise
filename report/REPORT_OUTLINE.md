# MaintainWise Report Outline

Replace every placeholder with actual project evidence and results.

1. **Abstract** — problem, dataset, approach, best model, actual metrics.
2. **Introduction** — predictive maintenance, context, motivation.
3. **Problem Statement and Objectives** — target users, problem, scope, success metrics.
4. **Literature Survey** — cite real papers/dataset documentation; summarize approach and limitations.
5. **Dataset Description** — UCI AI4I 2020 source, 10,000 rows, fields, target, synthetic-data limitation.
6. **Methodology** — architecture/flow diagram, cleaning, split, feature engineering, algorithms, SMOTE.
7. **EDA Results** — class balance, distributions, correlation heatmap, box plots.
8. **Model Comparison** — actual table for accuracy, precision, recall, F1, ROC-AUC; explain selection criteria.
9. **MLflow Tracking and Registry** — screenshot of 5+ runs, run comparison, artifacts, registered model.
10. **Jira Scrum Evidence** — epics, stories, points, assignees, sprint boards, burndown, reviews and retrospectives.
11. **Deployment** — Gradio interface, Space URL, repository files, screenshots.
12. **Testing, Challenges and Limitations** — test cases and actual outcomes, class imbalance, synthetic data, leakage controls.
13. **Conclusion, Future Work and References** — truthful summary, improvements, dataset and library citations.

## Test case table (fill actual result)
| ID | Test | Expected outcome | Actual outcome |
|---|---|---|---|
| TC01 | Load official CSV | Dataset loads and target exists | TODO |
| TC02 | Clean dataset | Duplicate/null summary generated | TODO |
| TC03 | Feature engineering | Temperature difference and power created | TODO |
| TC04 | Leakage check | IDs and failure-mode columns excluded | TODO |
| TC05 | Train models | Five algorithm runs logged | TODO |
| TC06 | Model metrics | Accuracy/precision/recall/F1/ROC-AUC logged | TODO |
| TC07 | Gradio valid inputs | Label and probability returned | TODO |
| TC08 | Gradio boundary inputs | App responds without crashing | TODO |
| TC09 | Hugging Face public URL | Space loads and prediction works | TODO |
