# Executed validation

Development runtime: Python 3.12, TensorFlow CPU 2.20.0, Keras 3.15.1. Docker targets Python 3.11; container execution was not available and is not claimed.

**10 tests passed**, including a real-model smoke (RUN_MODEL_TESTS=1). Covered negation preprocessing, OOV/truncation/padding, normalized-text split isolation, conflicting label rejection, metrics class ordering, SQL feedback bounds/upserts/no raw-text column, API auth/validation/CSV, static UI assets and real model batch/single inference parity.

The supplied real model loaded successfully. Three fixed smoke examples produced positive/negative/neutral respectively. These are engineering checks, not an accuracy benchmark. Original vs compact re-saved model probabilities matched at atol 1e-6 on those three examples.

A complete training smoke ran on the 36-row synthetic sample (25 train / 5 validation / 6 test), TF-IDF/logistic plus one epoch of BiLSTM. Reports and two MLflow runs were produced, and the newly exported neural model loaded through the Predictor contract. Synthetic metrics are intentionally not advertised as real quality. The original real training dataset is absent, so no credible replacement test accuracy is claimed.

JavaScript syntax checked with Node. Browser visual/interaction testing, Windows execution, GitHub Actions, cloud deployment, Docker builds, load testing and real dataset training were not performed here. Fixture API tests are explicitly fixture-based; real-model smoke is separate. UI is an integration with the same API, not a separately hosted service.

Additional real API smoke loaded the original model through FastAPI lifespan, predicted a CSV row, stored feedback and read SQL monitoring successfully. No mock substituted in this test.
