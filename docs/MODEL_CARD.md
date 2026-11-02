# Mizan legacy BiLSTM — Model Card

## Model

- Original model supplied by project owner; not downloaded from a third-party hub.
- Arabic 3-class sentiment: `neg`, `neu`, `pos`, extracted from supplied label encoder.
- Input: 150 post-padded/post-truncated token IDs. Vocabulary cap 60,000; embedding dimension **128**; bidirectional 64-unit LSTM; custom temporal SelfAttention; BatchNorm; dense 32; dropout; dense softmax 3.
- Actual artifact created with Keras 3.13.2. Re-saved with TensorFlow 2.20.0 / Keras 3.15.1, `compile=False`, no optimizer state, inference weights retained. Original vs re-saved numerical parity checked at atol 1e-6 on three smoke inputs; this is not proof of arbitrary-data quality.
- Artifact file SHA-256 and preprocessing contract are in `artifacts/legacy/manifest.json`; app refuses changed files.
- Tokenizer converted from original pickle through a restricted allowlist into JSON. No pickle loading occurs in the inference service. Bundled stopwords reproduce original NLTK rules; no network download at service startup.

## Training data and historical results

Original notebook expected a local CSV with `review_description,rating`. Dataset bytes, source, license, authorship/time grouping and held-out membership were not supplied. Reported historical accuracy 83.62% and macro F1 0.83 are not revalidated and may be optimistic. Notebook duplicates neutral samples before train/test splitting, and tokenizer is fitted on all text before splitting. Historical results must not be presented as unbiased held-out quality.

New training pipeline removes duplicate normalized texts and rejects conflicting labels before splitting; feature vocabularies are train-only; train class weights replace pre-split oversampling; validation chooses candidate; test is reporting-only. Exact-text split integrity does not prove author/time isolation or remove all near duplicates. No new real-data model has been trained in this delivery.

## Intended use

Local demonstration and assisted review of Arabic feedback. Threshold `.6` is a configurable review heuristic, not a calibrated decision threshold. Output label remains argmax even when flagged. Full score vector, OOV rate and truncation are visible. Manual corrections are audited without automated retraining.

## Limitations

- Source domain and label quality unknown; do not assume generalization across dialects/platforms.
- Sarcasm, negation, mixed sentiments and domain shift remain difficult.
- Legacy preprocessing removes Latin text, digits and emoji; these may carry sentiment. Empty cleaned input is rejected. No normalization change was silently applied to the existing model.
- Legacy attention does not mask zero padding. Preserved for compatibility; a padding-mask experiment requires retraining/evaluation.
- Softmax scores are uncalibrated; not correctness probabilities or intensity.
- No causal or feature-attribution explanations are claimed.
- Not suitable for adverse decisions about individuals, diagnosis or automatic moderation without further validation/human review.
- Architecture is supervised deep learning, not an LLM or RAG pipeline.

## Artifact publication

Dataset/model license not provided; release permissions require owner confirmation. The ZIP contains model/tokenizer; code-only Git clones omit weights unless separately restored. Do not publish private datasets or identifying vocabulary without appropriate rights.
