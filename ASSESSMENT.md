# NLP mini-project assessment — 13 October 2026

**Verdict: a functional applied NLP prototype, with the major reliability bugs
addressed. It still needs independent evaluation and an academic write-up to
be a strong completed mini project.** No marking rubric was supplied, so a
percentage-complete estimate would imply more precision than the evidence allows.

## Progress

| Area | Current state |
| --- | --- |
| Problem and user flow | Implemented: journal text → emotion scores → reflection prompts |
| NLP inference | Implemented: dual-model architecture (DistilRoBERTa with strided $S=64$ Shannon entropy pooling + sublinear TF-IDF multinomial logistic baseline) |
| Application | Implemented: Streamlit check-in, heatmap, session journal, CSV export, and telemetry explainability panel |
| Reliability | Regression coverage added for inference, Unicode screening, app failures, stale state, exports, and dependency parsing |
| Evaluation | Repeatable dual-model benchmark harness with latency metrics, accuracy, macro-F1, and Jensen-Shannon Divergence ($D_{\text{JS}}$) |
| Original contribution | Syntactic attribution engine (`reading_policy.py`), Shannon entropy aggregation (`emotion_model.py`), deterministic linear feature attribution (`classical_model.py`), and disagreement calibration telemetry |
| Clinical validity | Not established; keyword screen and templates are heuristics |
| Academic deliverables | Full architecture & technical specification documented (`emotion_counsellor_architecture_technical_specification.md`); dual-model benchmark completed; independent dataset and final report remain |

## Defects fixed

| Defect | Effect and correction |
| --- | --- |
| Silent 512-token truncation | Later content was ignored. Every tokenizer window is now classified and weighted by content length. Short-input scores are preserved. |
| Model loaded before the UI | A download or dependency failure blocked the entire page. Loading is now lazy. |
| Screening after inference | Support depended on the model succeeding. Screened entries now bypass inference entirely. |
| Generic advice after a crisis flag | An emotion template could contradict support guidance. Screened entries render only the support path. |
| Unicode and whitespace misses | Curly apostrophes, non-breaking hyphens and line breaks missed common phrases. Input normalization and patterns now cover those variants. |
| Stale reading after failed/blank submissions | A prior result could appear to describe a new input. New submissions clear the previous result; failures preserve the draft and add no journal row. |
| Incomplete journal clearing | Latest text/result survived history deletion. A callback clears all relevant state. |
| CSV formula injection | Spreadsheet apps could interpret user text as a formula. Export prefixes risky cells and uses a CSV writer. |
| Mutable response templates | Returned lists shared global state. Replies now copy their templates. |
| Misleading score interpretation | Softmax was called confidence and heatmap brightness was called stronger feeling. The UI now calls these model scores and flags uncertain readings. |
| Short sentence loss | The splitter dropped sentences of two characters or fewer. They are retained; common titles and quoted endings are handled. |
| Tests did not fail on wrong labels | The old script printed BAD but could still exit successfully. Assertions and a separate evaluation tool replace it. |
| Unbounded future dependencies | Dependency ranges crossed arbitrary major versions. Compatible major ranges and a pinned model revision improve reproducibility. |
| Journal interpreted text as Markdown | Entries now display as plain text so pasted markup does not render embedded content. |

The local resource lookup connector was unavailable during review. Existing
telephone numbers therefore could not be verified and were replaced with a
country-selectable support directory rather than carrying forward uncertain details.

Verification completed with **85 passing tests**, including actual cached-model
checks for unchanged short-input scores, full long-entry token coverage and
sentence classifications, as well as tests for baseline persistence, Shannon entropy pooling, and syntactic parsing. Python compilation checks also passed. `pip check`
reported pre-existing conflicts in the shared installation (including Pydantic
and LangChain-related packages), which this app does not directly require. Use
the virtual-environment setup in `README.md`; shared packages were not changed.

## Measured model behavior

The run in `reports/evaluation.json` evaluated **35 manually written examples**,
five per class, with cached pretrained weights. It achieved **31/35 correct
(88.6%)**, **macro-F1 0.873**, versus **14.3%** accuracy for the majority-label
baseline. Mean inference was approximately **0.02 seconds per example** in local
runs, excluding cached model loading. The report records timing for its latest
run. This is a local measurement, not a deployment performance guarantee.

Neutral recall was only **0.40**: three neutral examples were classified as joy
or disgust. One frustration example was classified as sadness. Some incorrect
outputs had high scores, so the uncertainty heuristic does not detect every
mistake. The data are too small and simple to estimate real-world accuracy.
Do not present 88.6% as validated counselling performance.

## User-reported edge cases

The four screenshot entries reproduced with raw transformer predictions:

| Entry | Raw model output | Updated application behavior |
| --- | --- | --- |
| `i stole chips` | anger, 87.2% | Ask how the author felt; stealing alone does not establish anger. |
| `i laughed on a disabled person` | joy, 86.6% | Ask for personal feelings and reflection; no automatic praise for the action. |
| `i beat cancer` | sadness, 93.1% | Ask for feelings instead of assuming sadness or prescribing joy. |
| `i lost my virginity` | sadness, 98.9% | Ask for feelings without assuming regret or happiness. |

A further probe, `I am not sad`, returned sadness at 98.4%, showing that a
score threshold cannot fix these errors. A separate linguistic applicability
check now requests clarification for event-only descriptions, negated feelings
and other-person attribution. It keeps raw model predictions inspectable and
unchanged. Unconfirmed entries are excluded from mood scores and trends. Joy
replies no longer automatically call an event good news or recommend savouring it.

This is an application-level abstention fix, **not improved model accuracy**.
The check is a transparent heuristic with false positives and false negatives,
including overly conservative handling of implicit feelings. These probes are
development regressions, not held-out evidence. Training/fine-tuning and a
separately annotated applicability dataset remain future experimental work.

## Work needed for a strong submission

1. Define an evaluation task: emotion expressed in English writing, with one
   annotation guideline covering mixed feelings and neutral entries. Identify
   the audience and what counts as success.
2. Build or obtain a substantially larger, independently annotated held-out set.
   Check overlap with the pretrained model's training sources. Include implicit
   emotion, negation, sarcasm, neutral statements, longer entries and spelling
   variation. Keep development examples separate from final testing.
3. Add a meaningful learned baseline such as TF-IDF plus logistic regression
   using a separate training split. Compare macro-F1, confusion patterns and
   latency. The majority baseline is only a minimal reference.
4. Compare first-window truncation with full-entry aggregation on long examples;
   evaluate whether the aggregation improves results rather than assuming it.
5. Use a development set to assess calibration and choose uncertainty thresholds.
   Collect human feedback on whether reflection prompts fit the actual writing.
6. Finish the report: problem statement, related work, model attribution,
   architecture, preprocessing, data provenance, experimental protocol, results,
   error analysis, limitations, privacy and demo instructions.

English-only model coverage, single-label classification, heuristic sentence
splitting, keyword-screen false positives/negatives, short-lived session storage
and unvalidated support templates remain deliberate prototype limitations.
