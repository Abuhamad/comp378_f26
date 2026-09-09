# 06 — AI Development Lifecycle & MLOps Infrastructure

## Abstract
Production machine learning requires a repeatable path from data to deployed predictions. This lesson defines the five pipeline stages, then shows how MLflow and DVC preserve experiment, data, and checkpoint history. CI/CD gates model promotion, while batch, real-time API, and edge deployment patterns match different latency and scale needs.

## Objectives
- Articulate every stage of the ML development lifecycle and name at least one tool or practice used at each stage.
- Instrument a training script with MLflow to log parameters, metrics, and artifacts across multiple runs, and compare those runs in the MLflow UI.
- Version a dataset and model checkpoint with DVC, then reproduce a prior training run from its git commit hash.
- Describe the three deployment patterns, batch, real-time API, and edge, and select the appropriate one for a given latency and scale requirement.

## Content

### End-to-end ML pipeline stages

A production pipeline has five stages: data ingestion, feature engineering, training, validation, and deployment. Each stage needs an input contract, an output record, and a repeatable tool or practice.

Data ingestion collects source records in defined formats. A schema validation step checks field names, types, ranges, and required values before downstream work starts. For example, a CSV ingestion job can reject a row with a missing label instead of passing an invalid record to training. A versioned source snapshot provides a fixed input for later reproduction.

Feature engineering transforms raw fields into model inputs. A practice such as a shared feature definition keeps training and serving transformations consistent. A feature store can hold named feature definitions and values for reuse. A common failure occurs when training applies one date encoding and deployment applies another. A transformation test can compare both paths on the same fixture.

Training selects a framework, compute target, and configuration, then writes model checkpoints. A scikit-learn estimator with a recorded configuration gives a concrete baseline. A training record should include the input version, feature definition, code commit, hyperparameters, and checkpoint path. Missing records prevent a later engineer from recreating a result.

Validation measures held-out metrics and runs behavioral tests. Accuracy on a validation split supplies one numeric check. A behavioral test can check predictions for fixed examples and expected output types. Validation should produce a report before deployment. A model with an improved training score but a lower held-out score fails this stage.

Deployment makes the validated model available to users or downstream jobs. A serving configuration identifies the model version, input schema, and output schema. A deployment practice can first release a model to a staging environment, then promote the same registered version after approval. A mismatch between the training schema and serving schema can produce invalid predictions.

### Model versioning and experiment tracking with MLflow

MLflow records the evidence for each training run. A Python script opens a run with `mlflow.start_run()`. The script records configuration with `mlflow.log_param("C", c_value)`, a result with `mlflow.log_metric("validation_accuracy", accuracy)`, and a file with `mlflow.log_artifact("metrics.json")`. A saved model directory or file can serve as the run artifact.

The MLflow Tracking UI lists runs in one experiment. Tables and plots allow comparison of the `C` parameter and validation accuracy across three runs. A run name, input version, and code commit make the comparison readable. The lab uses the Iris dataset and compares `C` values of `0.01`, `1.0`, and `100.0`.

MLflow model registration gives a model artifact a shared name and version. A call such as `mlflow.register_model("runs:/RUN_ID/model", "iris-logreg")` creates a registry entry. `MlflowClient` can inspect versions and move a version through the registry stages `None`, `Staging`, and `Production`. A promotion rule can require a recorded validation accuracy before a version enters `Production`.

### Dataset and model-checkpoint versioning with DVC

DVC tracks large datasets and model checkpoints beside Git history. `dvc init` adds DVC configuration to a repository. `dvc add data/iris.csv` creates a pointer file for the dataset. After training saves a checkpoint at `models/model.pkl`, `dvc add models/model.pkl` creates a second pointer file for the model checkpoint. These two commands version both the training input and the produced model.

`dvc push` sends the dataset and checkpoint contents to configured storage. A `dvc.yaml` file can declare the training command, dataset dependency, and `models/model.pkl` output. `dvc repro` reruns the training pipeline when a dependency or command changes. Git records the DVC pointer files and pipeline definition, while DVC storage holds the larger file contents.

A prior run can be reproduced by checking out a Git commit hash, running `dvc checkout` to restore both the dataset and checkpoint versions recorded by the commit, and running `dvc repro` to execute the recorded pipeline. The restored input, code, command, and checkpoint path provide a precise reproduction target.

### CI/CD for ML

CI/CD applies automated checks to model changes. A GitHub Actions workflow can trigger on a code push and run lint, tests, retraining, evaluation, and a promotion gate in order. A scheduled or data-change trigger can start automated retraining. The workflow stores the new run and its evaluation result before any registry change.

A model registry promotion gate compares the new evaluation result with a required threshold or a reference result. An accuracy drop blocks promotion. A passing model moves from `Staging` to `Production` after the gate succeeds. The gate records the decision with the run and model version.

A staged rollout sends a controlled portion of requests to a new production version before full promotion. A rollback returns service to the prior registered version when the new version fails the defined acceptance check. These patterns make deployment a sequence of recorded decisions rather than a manual file replacement.

### Deployment patterns

Batch inference runs predictions on a complete dataset at scheduled times. A nightly scoring job can read a fixed input, write predictions to a table, and finish before the next schedule. Batch inference fits workloads with high volume and no per-request latency requirement.

Real-time API inference serves one request at a time through an HTTP endpoint. A FastAPI application can load a registered model, validate an input record, and return a prediction. Real-time inference fits interactive clients with a low latency requirement and a service capable of handling concurrent requests.

Edge deployment runs a lightweight model near the data source. An ONNX export transfers a model to an ONNX runtime on a device. Edge inference fits cases with limited connectivity, local response requirements, or data kept near the device. The deployment choice follows the required latency, connectivity, scale, and input location.

The lifecycle connects fixed inputs, named transformations, recorded training runs, held-out validation, and registered deployment versions. MLflow records run evidence, DVC restores data and checkpoint versions, and CI/CD applies repeatable promotion gates. Batch inference suits scheduled full-dataset work, real-time APIs suit interactive requests, and edge deployment suits local response needs.


## Summary
The ML development lifecycle moves through data ingestion, feature engineering, training, validation, and deployment. MLflow tracks run parameters, metrics, artifacts, and registered model versions, while DVC versions datasets and checkpoints for reproduction. CI/CD automates retraining and promotion gates. Batch, real-time API, and edge deployment patterns serve different latency and scale requirements.

## Useful References and Resources

- MLflow documentation, “MLflow Tracking” and “MLflow Model Registry.”
- MLflow Python API reference, `mlflow.start_run`, `mlflow.log_param`, `mlflow.log_metric`, `mlflow.log_artifact`, and `mlflow.register_model`.
- DVC documentation, “Get Started” and “Data and Model Versioning.”
- DVC documentation, “Defining Pipelines” and the `dvc repro` command.
- GitHub Actions documentation, “Understanding GitHub Actions” and workflow syntax.
- ONNX documentation, “ONNX Runtime” and model conversion guides.
- FastAPI documentation, “First Steps” for HTTP model-serving endpoints.
