# Emotion Counsellor

A Streamlit NLP mini project that classifies English journal entries into seven
emotion labels and offers rule-based reflection prompts. It includes sentence
readings, a session journal, a score heatmap, CSV export and a conservative
keyword screen that routes potentially concerning text to immediate support.

## Run locally

Python 3.10 or newer is recommended. Tested here on Python 3.13 with Streamlit
1.65.0, Transformers 5.15.0, Torch 2.13.0 and Plotly 7.1.0.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m streamlit run app.py
```

The first ordinary analysis downloads the model through Hugging Face; subsequent
runs reuse its local cache. The page and keyword-screened support path can run
without downloading the model. Inference runs on CPU. If weights are already
cached, offline operation can be enabled with `HF_HUB_OFFLINE=1`.

## NLP approach

The classifier is [Jochen Hartmann's Emotion English DistilRoBERTa-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base),
a pretrained English model with anger, disgust, fear, joy, neutral, sadness and
surprise labels. Its weights are reused; this project does not train a model.
The checkpoint revision is pinned in `emotion_model.py` for reproducibility.

Inputs are limited to 20,000 characters. The tokenizer produces non-overlapping
windows of up to 512 tokens, including special tokens. Each window is classified
in batches of at most eight. Overall scores are the content-token-weighted mean
of window softmax scores. This includes the full input instead of discarding
everything after the first window. Window boundaries can lose context; averaging
can dilute a strong feeling expressed briefly. The implementation uses the
tokenizer's [overflow/truncation behavior](https://huggingface.co/docs/transformers/main_classes/tokenizer).

Multiple sentences receive separate readings with the same window handling.
Sentence splitting is a lightweight English heuristic, with handling for common
titles, initials and closing quotes. It is not a full linguistic parser.

A reading is marked uncertain if the top score is below 0.50 or its lead over
the second score is below 0.15. These thresholds are heuristics, not validated
calibration. Model scores concern the writing, not a probability that a person
has a particular emotional or clinical state. Support replies are fixed
templates, not generative counselling. Uncertain entries receive broader prompts.

`reading_policy.py` adds a separate, conservative applicability check. Describing
an event does not necessarily express the author's feelings: "I stole chips",
"I beat cancer" and "I lost my virginity" can support several interpretations.
Such entries now ask "How did that feel for you?" instead of presenting the
classifier's high-scoring guess as a personal reading. Negated feelings and
another person's emotional state also trigger clarification. The raw scores
remain available in a collapsed inspection panel and are not rewritten.

This check uses first-person language, a feeling vocabulary and conservative
negation/attribution rules. It is **not trained semantic understanding** and can
ask for context unnecessarily, especially for implicit emotion. Adding a personal
feeling sentence such as "I feel relieved" supplies useful context without forcing
every recovery, intimate experience or admission into a predetermined emotion.
Joy replies validate the possible feeling without automatically praising the action.

## Verification and evaluation

```powershell
python -m pytest -q
python evaluate.py
```

The default tests mock inference, so they do not download weights. To also run
the actual model checks:

```powershell
$env:RUN_MODEL_TESTS = "1"
$env:HF_HUB_OFFLINE = "1"  # only when the checkpoint is already cached
python -m pytest -q
```

`evaluate.py` reads `text,label` CSV files and writes accuracy, macro-F1,
per-class precision/recall/F1, a confusion matrix (rows = expected, columns =
predicted), misclassified examples, uncertainty counts, latency, model revision
and installed library versions. It also reports a majority-label baseline.
Metrics evaluate the raw transformer predictions on every dataset row;
`context_requests` separately records application abstentions. Abstentions are
not silently dropped from the accuracy calculation.

```powershell
python evaluate.py --dataset data/my_held_out_set.csv --output reports/held_out.json
```

The bundled 35 examples are manually written smoke data, balanced at five per
class. On the assessed run, accuracy was **88.6%** and macro-F1 was **0.873**.
These figures are not a benchmark claim: the examples are small, obvious and
not independently annotated. See `reports/evaluation.json` and `ASSESSMENT.md`.
Custom evaluation reports include the text of misclassified examples, so use
consented, anonymized data when writing reports to disk.

## Support and session behavior

The keyword screen runs before the classifier. Flagged entries skip emotion
classification and show support guidance; no cheerful template is shown even if
the model is unavailable. The screen handles common Unicode punctuation and
whitespace variations. It can miss indirect language and can flag quotations,
negations or third-person reports. It is not a validated risk detector.

Unverified hard-coded phone numbers were removed. The UI links to a directory
with country selection; users should seek local emergency care if in immediate
danger. This project is a reflection prototype, not a diagnostic or clinical
service.

The app retains entries only in Streamlit session memory on the server. It does
not write journal entries to a database or disk. Closing a browser tab need not
delete server state immediately. **Clear journal** clears history, draft, latest
reading and notice. Downloaded CSV files remain on the user's device. CSV export
escapes potential spreadsheet formulas and safely quotes multiline text.
Support-screened entries appear in the journal with blank score cells and do not
contribute to emotion trend calculations. Uncertain entries also do not contribute
to the trend flag, which describes recent entries rather than clinical persistence.
Entries requiring more context have blank score cells as well, so an unconfirmed
model guess does not become an apparent mood history.

## Files

| File | Purpose |
| --- | --- |
| `app.py`, `style.css` | Streamlit interface and theme |
| `emotion_model.py` | Tokenization, inference, score aggregation, sentence readings |
| `counselling.py` | Keyword screen, response templates, trend heuristic |
| `reading_policy.py` | Conservative personal-feeling applicability and clarification rules |
| `journal.py` | CSV export |
| `evaluate.py`, `data/evaluation.csv` | Repeatable evaluation command and smoke examples |
| `test_*.py` | Unit, app-runtime and opt-in model checks |
| `ASSESSMENT.md` | Project readiness, verified defects and remaining work |
