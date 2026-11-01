# Tourism Package Purchase Prediction

## Project overview

This project develops a reproducible model to help Visit with Us prioritize customer outreach for its Wellness Tourism Package. It includes dataset validation, preparation, model tuning and evaluation, an automated GitHub Actions workflow, and a Streamlit application.

## Dataset

The registered dataset is `data/tourism.csv`, containing 4,128 customer records and 20 meaningful columns. The dataframe index column is removed during data registration. The target, `ProdTaken`, contains 797 purchases and 3,331 non-purchases. The source has no missing fields or exact duplicate rows. Obvious label variants in `Gender` and `Occupation` are standardized.

The intended use is to estimate purchase interest before a sales contact. `CustomerID` is excluded as an identifier. The model also excludes `TypeofContact`, `DurationOfPitch`, `NumberOfFollowups`, `ProductPitched`, and `PitchSatisfactionScore` because they describe information recorded during or after contact and could leak future information into a pre-contact prediction.

## Modeling approach

A stratified 80/20 split preserves the positive-class share in the training and test sets. Numeric and categorical preprocessing is fitted within a scikit-learn pipeline. A class-weighted Random Forest is tuned using 3-fold stratified cross-validation, with ROC-AUC as the selection metric. The held-out evaluation reports ROC-AUC, average precision, recall, precision, F1, accuracy, and the confusion matrix. Experiment parameters and results are stored under `reports/`; the fitted preprocessing-and-model pipeline is stored under `models/`.

## Repository structure

```text
data/tourism.csv                 Registered source dataset
src/data_validation.py           Schema checks and data summary
src/prepare_data.py              Cleaning and stratified data splits
src/train_model.py               Tuning, evaluation, and model persistence
tourism_package_prediction.ipynb Analysis and results notebook
models/best_model.joblib         Trained model pipeline
reports/                         Parameters and evaluation records
app.py                           Streamlit application
.github/workflows/pipeline.yml   Automated MLOps workflow
Dockerfile                       Container setup for Hugging Face Spaces
hf_space/README.md               Space metadata
requirements.txt                 Python dependencies
```

## Pipeline and deployment

The GitHub Actions workflow validates the dataset, prepares train/test files as a workflow artifact, trains and evaluates the model, and commits the model and experiment records to `main`. The Streamlit app accepts customer profile fields and displays the estimated purchase likelihood. The same app can be deployed on Streamlit Community Cloud and as a Docker-based Hugging Face Space.

## Responsible use and limitations

The output is a historical-data-based estimate, not a guarantee of customer behavior. The positive class is the minority outcome, and model performance may vary across customer groups or change over time. The score is intended to support human-reviewed outreach, not serve as the sole basis for eligibility or pricing decisions.
