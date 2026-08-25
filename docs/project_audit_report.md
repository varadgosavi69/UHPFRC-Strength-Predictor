# Project Audit Report

## Project

**AI-Based Prediction and Optimization of Compressive Strength of Sustainable Polypropylene Fiber Reinforced Concrete Using Recycled Aggregate**

**Audit date:** 25 August 2026  
**Audit basis:** Current repository contents and executable project files

## 1. Executive Summary

The project currently has a partially implemented software and machine-learning framework, but it does not yet have the research evidence required for a completed final-year project.

The repository contains data templates, an exploratory-data-analysis notebook, training scripts for Random Forest, XGBoost, ANN, and Linear Regression, SHAP explainability code, a FastAPI backend, and a React frontend. However, both project CSV files contain headers only and zero observations. No trained model artifacts, experimental results, RSM model, optimization routine, or completed report chapters are present.

The estimated overall completion is **approximately 20-25%**. The immediate priority is to populate, validate, and document the dataset because every downstream phase depends on it.

## 2. Completion by Phase

| Phase | Estimated completion | Current status |
|---|---:|---|
| Dataset compilation | 10% | Schema exists, but the two project CSVs contain no data rows. |
| EDA | 25% | Notebook code exists for summary statistics, missing values, correlations, distributions, and source counts. No stored executed outputs. |
| RSM modelling | 0% | No response-surface implementation found. |
| ML modelling | 35% | Training scripts exist for RF, XGBoost, ANN, and Linear Regression. No fitted artifacts or reported metrics. |
| SHAP analysis | 30% | Global and single-prediction explanation code exists, but it depends on a missing trained model. |
| Mix optimization | 0% | No optimization algorithm, objective, constraints, or optimized mix results. |
| Deployment | 35% | FastAPI and React prediction workflow exists, but it cannot run without a trained model. |
| Report writing | 5% | Only a minimal README exists; no completed technical report is present. |

## 3. Completed Work

- Repository structure and initial project modules created.
- Data schema and source-paper metadata field defined.
- EDA notebook drafted.
- Preprocessing pipeline created for numeric and categorical features.
- Train/test split and R2, RMSE, and MAE evaluation functions created.
- Training scripts created for:
  - Linear Regression
  - Random Forest
  - XGBoost
  - ANN/MLP
- SHAP explanation pipeline drafted for tree, linear, and neural-network models.
- FastAPI endpoints created:
  - `GET /health`
  - `POST /predict`
- React mix-design input form connected to the prediction endpoint.
- Python modules pass compilation.
- Frontend production build passes with Vite.

## 4. Pending Work

### Dataset and EDA

- Extract and enter the observations from the five research papers.
- Confirm the number of rows, columns, and observations per paper.
- Confirm that recycled aggregate and polypropylene-fibre variables are represented consistently.
- Resolve schema inconsistencies, including fields used by the application but absent from the current CSV header.
- Define units, data dictionary, inclusion criteria, and duplicate handling.
- Execute the EDA notebook and save plots and tables.

### Modelling

- Train every model on the populated dataset.
- Use cross-validation or repeated validation in addition to one train/test split.
- Prevent source-paper leakage and assess generalization across papers.
- Save model artifacts and metric tables.
- Report model performance with appropriate engineering interpretation.

### RSM, SHAP, and optimization

- Design the RSM experiment and fit the response-surface model.
- Compare RSM results with ML predictions.
- Run SHAP analysis on a successfully trained model and save global plots.
- Define optimization objectives, such as maximizing compressive strength while respecting material and sustainability constraints.
- Produce optimized mix designs and validate them against realistic ranges.

### Deployment and report

- Test the complete backend/frontend workflow with a real trained artifact.
- Add reproducible setup and run instructions.
- Document model limitations and out-of-distribution inputs.
- Write the methodology, results, discussion, literature comparison, limitations, conclusion, and references.

## 5. Technology Assessment

### Implemented in the repository

- Python: pandas, NumPy, scikit-learn, XGBoost, SHAP, matplotlib, seaborn, joblib.
- Machine learning: Random Forest, XGBoost, MLP/ANN, and Linear Regression.
- Backend: FastAPI, Pydantic, Uvicorn dependencies, and CORS configuration.
- Frontend: React 18, Vite, and Axios.
- Explainability: SHAP TreeExplainer, LinearExplainer, and KernelExplainer paths.

### Not implemented or not demonstrated

- RSM/statistical design of experiments.
- Constrained mix optimization.
- Cost, carbon, or other sustainability objective.
- Experimental validation of predictions or optimized mixes.
- Uncertainty estimates or confidence intervals.
- Cross-paper holdout validation.
- Persisted trained models, plots, metric tables, and experiment logs.
- Production deployment configuration.

The backend is therefore a software shell rather than a functioning deployed predictor: startup requires `models/best_model.pkl`, but that artifact is not present.

## 6. Differentiation from the Three 2025 Papers

The comparison literature covers closely related work involving polypropylene fibres, recycled or plastic aggregate, RSM, machine learning, and GUI-based prediction. The repository's current implementation provides only limited, code-level differentiation.

### Defensible differentiation at the current stage

- A schema is being prepared specifically around compressive strength and concrete mix-design variables.
- The planned model comparison includes RF, XGBoost, and ANN in one pipeline.
- SHAP explanations are designed for individual predictions as well as global interpretation.
- A React/FastAPI application structure is present for user-facing prediction.

### Claims that cannot yet be made

The project cannot currently claim that it:

- outperforms any of the three papers;
- contributes a validated new dataset;
- successfully combines RSM and ML;
- performs mix optimization;
- provides a working predictor on real project data;
- demonstrates improved sustainability;
- produces novel SHAP findings; or
- has experimentally validated optimized mixes.

The honest current contribution statement is:

> The project has established a preliminary framework for compressive-strength prediction of sustainable fibre-reinforced concrete, combining multi-model machine learning, SHAP explainability, and a web interface. Dataset completion, RSM modelling, optimization, validation, and empirical differentiation from prior work remain to be completed.

## 7. Recommended Order of Work

1. Populate and validate the master dataset.
2. Freeze the data dictionary, units, inclusion criteria, and preprocessing rules.
3. Complete and execute the EDA.
4. Train and validate the baseline and advanced ML models.
5. Implement RSM and compare it with ML.
6. Add SHAP analysis using the validated best model.
7. Implement constrained optimization and validate candidate mixes.
8. Connect and test the deployment workflow.
9. Write the report from the recorded results rather than projected claims.

## Conclusion

The project has a credible starting software structure, but it is currently at the framework stage. Its research contribution cannot be assessed until the dataset is populated, models are trained, and RSM, optimization, validation, and reporting are completed. The most important next deliverable is a clean, auditable dataset with documented sources and consistent variables.
