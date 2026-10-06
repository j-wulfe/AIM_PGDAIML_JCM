# Bias & Fairness Analysis

> **Model under analysis:** the overall champion is **XGBoost** (test AUC-PR 0.817; see the
> Final Report). The explainability, fairness, and overfitting analyses below were run on the
> **Random Forest** (test AUC-PR 0.793) as a representative tree model; repeating them on the
> XGBoost champion is a recommended next step, though the two tree ensembles are expected to
> behave similarly.

## 1. Explaining model decisions

Because a fraud model triggers real-world actions — blocking a payment, opening an
investigation — its decisions must be explainable rather than opaque. We used model-based
feature importances together with SHAP, PDP, ICE, and LIME.

The model's **feature importances** rank **V14, V10, and V12** as the strongest drivers, and
these were carried into the partial-dependence analysis. **Partial Dependence Plots (PDP)**
show the *average* effect of each: as **V14** falls below roughly −0.5, the predicted fraud
probability jumps sharply (from ~0.004 to ~0.011), with the same step-like pattern for V10 and
V12 — low/negative values of these PCA components signal fraud. **Individual Conditional
Expectation (ICE)** plots one line per transaction for V14; the lines spread widely, with a
subset of transactions far more sensitive to V14 than the average, confirming heterogeneous
behaviour. **LIME** on a single flagged fraud case attributed the decision mainly to
**V14 ≤ −0.43, V4 > 0.74, V12 ≤ −0.40, and V10 ≤ −0.54** — the kind of per-case rationale an
investigator needs. **SHAP** confirms this globally: the beeswarm ranks **V14, V12, and V4**
as the highest-impact features (followed by V3, V11, and V10) and shows the *direction* of each
effect — low values of V14 and V12 push a transaction strongly toward "fraud."

Together these move the model from a black box to an auditable system: global behaviour
(importances, SHAP, PDP), heterogeneity (ICE), and case-level justification (LIME). The
consistency across independent methods — SHAP, importances, PDP, and LIME all surface V14, V12,
V4, and V10 — gives confidence that the model relies on stable, genuine signal.

## 2. Limitations

**Class imbalance.** Fraud is roughly 0.17% of transactions. Left unaddressed, a model can
score ~99.8% accuracy while catching no fraud, so accuracy was rejected as a metric. We
mitigated imbalance with balanced class weights and by tuning the decision threshold for F2
(favouring recall), and evaluated on AUC-PR, the appropriate summary metric for rare
positives.

**Data leakage.** Leakage is the most common cause of inflated, non-reproducible results. We
prevented it structurally: the data was split before any scaling, all transformations were
fit on the training set only, and the operating threshold was selected using training
cross-validation — never the test set. Any oversampling (SMOTE) is confined to
cross-validation folds.

**Overfitting.** We quantified it by comparing train and test AUC-PR: **0.980** vs **0.793**,
a gap of **0.187**. This is a **sizeable** gap, indicating the model overfits the training data
to some degree — unsurprising for a depth-12 Random Forest on highly imbalanced data. Stronger
regularisation (shallower trees, higher `min_samples_leaf`) or more data would likely narrow
it. Importantly, the test AUC-PR of 0.79 still sits far above the ~0.0017 random baseline, so
the model remains highly useful despite the gap.

**Anonymised features.** `V1`–`V28` are PCA components, so we cannot attach real-world meaning
to the most important features or rule out that the model relies on something ethically
problematic encoded within them. This is a transparency limitation inherent to the dataset.

## 3. Bias detection & fairness audit

**The central ethical finding: this dataset contains no sensitive attributes.** There are no
gender, race, age, or socioeconomic fields — only `Time`, `Amount`, and anonymised PCA
components. A direct fairness audit across protected groups is therefore **not possible on
this data**, and we report that honestly rather than fabricating demographic breakdowns. In a
production setting this is itself a governance gap: without demographic data (collected
lawfully and with consent) an institution *cannot verify* that its fraud model treats
protected groups equitably.

To still demonstrate the audit methodology, we used a **declared proxy**: transaction
`Amount`, split at the median (€21.59) into high-value and low-value groups of 28,373 each, as
a rough stand-in for socioeconomic status. Results are illustrative of method only, not
evidence of demographic fairness. Note that here the "positive" prediction (a flag) is the
*adverse* customer outcome, since it means friction or a blocked transaction.

| Fairness metric | What it measures | Result | Interpretation |
|---|---|---|---|
| Demographic parity (flag rate per group) | Equal flag rates across groups | high-value 0.18% · low-value 0.16% | broadly similar |
| Disparate impact ratio | min/max flag rate; < 0.80 = red flag | **0.863** | **PASS** (≥ 0.80) |
| Equal opportunity (ΔTPR) | Equal fraud-catch rate across groups | 0.062 | small gap (82.2% vs 76.0%) |
| Equalised odds (ΔFPR) | Equal false-alarm rate across groups | 0.000 | effectively equal |

On the proxy groups the model **passes the 80% disparate-impact rule (ratio 0.863)** and
raises false alarms at effectively equal rates across groups (ΔFPR ≈ 0.000). There is a small
equal-opportunity gap (ΔTPR = 0.062): fraud on higher-value transactions is caught somewhat
more often (82.2%) than on lower-value transactions (76.0%) — worth monitoring, but not a
major disparity. Crucially, this proxy is **not** a substitute for a true demographic audit —
it only demonstrates the method; a real assessment requires lawfully collected, consented
sensitive attributes.

## 4. Proposed mitigations

- **Reweighting / resampling** — balanced class weights (already applied); SMOTE or
  undersampling confined to training folds.
- **Regularisation** — to narrow the train/test gap, use shallower trees, a higher
  `min_samples_leaf`, or a well-tuned gradient-boosted model.
- **Threshold adjustment** — set a per-segment operating point if any group is systematically
  over-flagged, trading a little overall recall for more equal treatment.
- **Data augmentation & richer data** — collect lawful, consented demographic data to enable a
  genuine fairness audit; synthesise rare fraud patterns to strengthen the minority class.
- **Post-processing** — apply an equalised-odds post-processor (e.g. `fairlearn`'s
  `ThresholdOptimizer`) to equalise error rates across groups after training.
- **Governance & monitoring** — require human review for high-value flagged transactions, and
  continuously monitor group-wise flag rates in production to catch fairness drift.

## 5. Conclusion

The model is explainable (importances / SHAP / PDP / ICE / LIME), with independent methods
agreeing on a small set of genuine drivers (V14, V12, V4, V10). Its principal risks were
addressed and quantified: imbalance via class weights and F2 thresholding, leakage via strict
train-only fitting, and overfitting measured directly (a sizeable 0.19 AUC-PR train/test gap
to be narrowed with regularisation). On the declared `Amount` proxy the model shows no major
disparity (disparate-impact 0.863, equal false-alarm rates, a small catch-rate gap worth
monitoring). The key ethical conclusion is that a rigorous demographic fairness audit is
impossible on this dataset because it holds no sensitive attributes; we therefore demonstrated
the methodology on a transparent proxy and set out the data-governance steps a real deployment
would require to audit and mitigate bias responsibly.
