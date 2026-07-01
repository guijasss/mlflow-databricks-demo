# Databricks MLflow Feature Store Lab — Implementation Checklist

## Goal

Build a small end-to-end lab to test Databricks MLflow, Feature Store concepts, batch inference, online inference, and a simple Champion / Challenger workflow.

Avoid building a full production platform. Keep the implementation minimal and focused on testing the main Databricks ML features.

---

# 1. Project Setup

* Create a simple repository structure.
* Add a short README explaining the project goal.
* Define basic configuration for:

  * catalog/schema names
  * table names
  * model name
  * feature table name
  * batch scoring table name
* Keep configuration centralized and easy to change.

---

# 2. Synthetic Data

* Create a simple synthetic dataset for a binary classification problem.
* Use a simple business scenario, such as:

  * fraud prediction
  * order delay prediction
  * customer churn prediction
* Generate at least:

  * entity ID
  * event timestamp
  * raw input attributes
  * label column
* Make the dataset simple enough to understand and debug.

---

# 3. Raw / Silver Tables

* Create a raw table with the simulated input data.
* Create a cleaned table with basic transformations.
* Apply only simple data cleaning:

  * cast data types
  * remove invalid rows
  * standardize column names
* Do not overengineer a full medallion architecture.

---

# 4. Offline Feature Table

* Create one offline feature table.
* Define a clear primary key, such as:

  * `customer_id`
  * `event_timestamp`
* Add a small set of features, for example:

  * count of previous events
  * average transaction value
  * total value in a time window
  * recency feature
* Keep the number of features small.
* Register or use the table as the main offline feature table.

---

# 5. Labels

* Keep labels separate from the feature table.
* Create a simple label table.
* Join labels with features only when building the training dataset.
* Avoid storing labels directly inside the feature store table.

---

# 6. Training Dataset

* Build a training dataset by joining:

  * offline feature table
  * label table
* Ensure the join is temporally valid.
* Split the dataset into:

  * train
  * validation/test
* Keep the split logic simple and reproducible.

---

# 7. MLflow Training

* Train a simple classification model.
* Use MLflow to log:

  * parameters
  * metrics
  * model artifact
* Register the trained model in the Databricks Model Registry.
* Use a simple metric for comparison, such as:

  * AUC
  * accuracy
  * F1 score

---

# 8. Champion Model

* After the first training run, assign the trained model as the initial Champion.
* Use a model alias or equivalent Databricks mechanism.
* Avoid hardcoding model versions in inference jobs.

---

# 9. Batch Inference

* Create a batch inference job.
* Load the current Champion model.
* Score a set of records using the offline feature table.
* Write predictions to a Delta table.
* Store at least:

  * entity ID
  * prediction timestamp
  * model version
  * prediction score
  * predicted class

---

# 10. Online Serving

* Deploy the Champion model to a Databricks Model Serving endpoint.
* Create a minimal example request payload.
* Validate that the endpoint returns predictions.
* Keep online serving simple.
* Do not build a full external API application unless necessary.

---

# 11. Online Feature Store / Feature Serving

* Add a minimal test for online feature serving if available in the workspace.
* Sync or expose a small subset of features for online lookup.
* Validate that online inference can use recent feature values.
* Keep this as a small proof of concept, not a full serving platform.

---

# 12. Challenger Model

* Create a second training execution that produces a new model version.
* Treat this new version as the Challenger.
* Compare Challenger against the current Champion using the same validation dataset.
* Use a simple promotion rule, such as:

```text
Promote Challenger if validation AUC is greater than Champion AUC.
```

---

# 13. Automatic Promotion

* Implement a simple promotion step.
* If the Challenger is better, update the Champion alias to point to the Challenger version.
* If not, keep the current Champion.
* Log the promotion decision in a small Delta table.

---

# 14. Retraining Trigger

* Implement one simple retraining trigger.
* Prefer a simple rule, such as:

  * retrain when a minimum number of new labeled records is available
* Do not implement multiple trigger types.
* Do not implement complex drift detection unless needed later.

---

# 15. Databricks Workflow

* Create one Databricks Workflow with the main tasks:

  * generate/update synthetic data
  * build feature table
  * build training dataset
  * train challenger
  * evaluate champion vs challenger
  * promote model if better
  * run batch inference
* Keep the workflow linear and easy to inspect.

---

# 16. Minimal Monitoring

* Create a simple monitoring table for:

  * model version used
  * number of predictions
  * average prediction score
  * execution timestamp
* Optionally include basic model performance if labels are available.
* Avoid implementing a complete monitoring framework.

---

# 17. Documentation

* Document how to run the project.
* Document the purpose of each table.
* Document the Champion / Challenger flow.
* Document how to call the serving endpoint.
* Keep the documentation concise.

---

# Out of Scope

Do not implement the following unless explicitly requested later:

* complex feature drift detection
* advanced data quality framework
* multiple models
* multiple feature tables
* full external API application
* complex CI/CD
* advanced monitoring dashboards
* real-time streaming ingestion
* complex permission/governance setup
* production-grade orchestration framework
* full medallion architecture with many layers

---

# Final Expected Result

At the end, the project should demonstrate:

* one synthetic dataset
* one offline feature table
* one label table
* one training pipeline
* one MLflow registered model
* one Champion model
* one Challenger evaluation flow
* automatic Champion promotion
* one batch inference process
* one online serving endpoint
* one simple Databricks Workflow
* minimal monitoring and audit tables
