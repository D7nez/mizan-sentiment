# Mizan — Arabic Sentiment Engineering

TensorFlow/Keras Arabic sentiment classification with a modular inference service, Arabic review UI, SQL audit/feedback, Docker, a clean-split training pipeline and MLflow experiment comparison.

**Model status:** The supplied historical BiLSTM + attention model is runnable. Its original reported 83.62% accuracy / 0.83 macro F1 are **not independently validated**: the original notebook oversampled before splitting and fitted its tokenizer before splitting. Do not present these figures as reliable held-out quality. The source training dataset was not included, so no replacement real-world score is claimed.

## Start on Windows

Extract the delivered ZIP; it contains the actual model and JSON tokenizer. From its root:

```bat
cd "C:\Users\gemar\OneDrive\سطح المكتب\mizan-sentiment"
python scripts/check_artifacts.py
docker compose up --build -d
start http://localhost:8092
```

A CPU deployment; no GPU/CUDA required. First build downloads TensorFlow dependencies. The service loads and warms the model before becoming ready. API docs: http://localhost:8092/docs. Requests remain on localhost by default. No API key is required in this local default; to enable one, copy `.env.example` to `.env`, set `APP_API_KEY`, and recreate the service. Browser keys stay in memory. Input texts are **not** stored in the audit database; only request IDs, labels, model version, review flags and timings are stored. Feedback stores a label correction, not source text, and never retrains automatically.

## Structure

| Module | Responsibility |
|---|---|
| `src/mizan/preprocess.py` | Original-compatible Arabic normalization, bundled stopwords, JSON tokenizer, truncation/OOV diagnostics |
| `src/mizan/layers.py` | TensorFlow custom attention and BiLSTM architecture |
| `src/mizan/predictor.py` | Checksums, safe Keras loading, serialized model access, batch inference |
| `src/mizan/api.py` | FastAPI predict/CSV/feedback/monitoring and Arabic UI |
| `src/mizan/storage.py` | SQLite audit and validated feedback upserts |
| `src/mizan/data.py` | Deduplication, conflicting-label rejection and stratified clean-text split |
| `src/mizan/train.py` | Train-only vocabulary, TF-IDF/logistic baseline, BiLSTM training, reports and MLflow |
| `artifacts/legacy/` | Existing model and preprocessing contract, no inference-time pickle |
| `notebooks/Training.ipynb` | Guided commands for corrected training |
| `notebooks/original_reference.ipynb` | Original notebook with outputs cleared; reference only, contains known leakage |

## API

- `GET /health`, `GET /ready`: availability and artifact evaluation status.
- `POST /predict`: `{"texts":["الخدمة ممتازة"]}`; 1–100 texts, 1–5000 characters each.
- `POST /predict-csv`: JSON `{"csv_content":"text\nالخدمة ممتازة\n","text_column":"text"}`; max 1,000,000 characters, at most 100 rows.
- `POST /feedback`: `{"request_id":"UUID","item_index":0,"label":"neg"}`.
- `GET /monitoring`: SQL request counts, timings, review flags, recent label mix and feedback count. These are operational statistics, **not** accuracy or statistical drift detection.

If configured, use `X-API-Key`. Single and batch inference share one implementation. Empty-after-cleaning inputs are rejected; whole batch fails validation instead of silently guessing on invalid rows.

Outputs include the three softmax scores, argmax label, token count, OOV ratio, truncation and review flags. Scores are **not calibrated correctness probabilities or sentiment intensity**. A low score is flagged for review; it does not silently become neutral. Keyword lists are not presented as model explanations. Attention is not exposed as causal explanation. The original unmasked padding behavior is retained for inference compatibility; changing it requires retraining.

## Train with your dataset

Provide a licensed/private local CSV with `review_description` and `rating` (-1,0,1), or `neg/neu/pos`. Data is not included or automatically downloaded. Conflicting labels for the same cleaned text fail explicitly. Exact cleaned duplicates are removed before a stratified 70/15/15 split; tokenizer/TF-IDF fit only the training partition; class weights are computed only on train. Validation macro F1 selects between the candidates; test data is reporting-only. This prevents exact cleaned-text leakage, not all near-duplicate or author/time leakage. Choose group/time splitting when your domain requires it.

```bat
python -m pip install -r requirements-train.txt
python -m pip install --no-deps -e .
python -m mizan.train --data data/dataset.csv --models both --output reports/experiment-01
mlflow ui --backend-store-uri sqlite:///state/mlflow.db --port 5002
```

Use an empty output directory per experiment. `--text-column`, `--label-column`, `--seed`, `--epochs` are configurable. Reports include per-class metrics, confusion matrix, log loss, split membership hashes, data SHA-256 and MLflow run IDs. Saved baseline joblib is a trusted local training artifact; it is not loaded by the API. New Keras export supports the inference contract; deployment/model promotion remains a manual choice after validation. Baseline and neural candidates use the same splits and preprocessing. No automatic retraining or automatic promotion is implemented.

Docker training alternative:

```bat
docker compose --profile tools run --build --rm train python -m mizan.train --data data/dataset.csv --models both --output reports/experiment-01
```

A **synthetic pipeline smoke**, not a benchmark:

```bat
python -m mizan.train --data data/example_synthetic.csv --models both --epochs 1 --synthetic --output reports/smoke-01
```

Training metadata marks this run `synthetic=true`; never report its scores as performance on real Arabic reviews. The original dataset and license/provenance need to be supplied before credible retraining. Repeated tuning against test results invalidates the held-out claim.

## Tests and validation

```bat
python -m pip install -r requirements-test.txt
python -m pytest -q
```

Full real-model smoke after installing runtime dependencies and ensuring model file exists:

```bat
set RUN_MODEL_TESTS=1
python -m pytest -q
```

See [VALIDATION.md](VALIDATION.md) for exactly what ran. No Docker/cloud execution is claimed. CI tests deterministic logic and API using a clearly labelled fixture; it does not fabricate model predictions or quality claims. A missing/corrupt model makes the real service fail startup rather than substitute a mock.

## GitHub and model files

The ZIP contains the model. `.gitignore` excludes `.keras` by default to keep code commits small. A source-only clone **will not run inference without the artifact**. Share the model via an authorized release/artifact store and restore it to `artifacts/legacy/model.keras`, then run `scripts/check_artifacts.py`. Alternatively use Git LFS with an explicit artifact publication decision. Do not commit real review datasets, audit databases, `.env`, or private vocabularies without permission. Dataset/model license was not supplied; no third-party dataset redistribution rights are asserted.

## Scope

This project demonstrates **TensorFlow, Python, SQL, REST APIs, evaluation, Docker and ML workflows**. It is a supervised classifier. It does not implement an LLM, LangChain, RAG, a vector database, cloud hosting, production identity/roles, calibration, or automated drift detection. Use the companion LLM/RAG projects to demonstrate those skills. See [job mapping](docs/JOB_MATCH_AR.md) and [model card](docs/MODEL_CARD.md).
