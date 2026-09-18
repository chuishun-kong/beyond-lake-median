# Jasień identity-boundary sensitivity

This directory provides a targeted post hoc
robustness check, not a replacement estimand and not a strict-other-23 run.

`Lake Jasień Południowy` and `Lake Jasień Północny` remain distinct published
lake identities. Because their GLORIA records share a rounded coordinate,
dataset, and provider, this sensitivity removes the four northern groups from
every source-side fit only when the southern lake is the outer target.

The script reruns the southern fold's full inner-LOLO hyperparameter selection,
then H1, all three H2 estimand layers, and paired H3. It builds hybrid 24-lake
tables by replacing only the southern rows in copies held under this directory.
Primary frozen CSVs are hashed before execution and checked again at the end.

The target group order and random support/query trajectories remain fixed.
H2 deterministic selectors are rerun from the sensitivity source
representation; their supports may therefore change and are explicitly
compared in the results. Ordered-sequence changes and support-set replacements
are reported separately.

The final impact class is
`NUMERICALLY_SENSITIVE_BUT_INTERPRETIVELY_STABLE`: no H1/H2-own/H3 headline
classification changes, but the southern fold's MLP hyperparameter choice and
several single-fold values change materially. One secondary pairwise
common-query cell moves from `random_better` to `unresolved` at a CI endpoint
within 0.001 of zero; it does not alter the own-pool headline conclusion.

Run the full targeted fold:

```powershell
python experiments/jasien_identity_sensitivity/run_jasien_identity_sensitivity.py
```

After a full run, rebuild only the hybrid summaries from its checkpoints
(checkpoints are not bundled in this repository):

```powershell
python experiments/jasien_identity_sensitivity/run_jasien_identity_sensitivity.py --aggregate-only
```

Outputs are written within this experiment directory; primary results remain
separate from the sensitivity summaries.
