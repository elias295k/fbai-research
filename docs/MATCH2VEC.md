# Match2Vec experimental candidate

## Research question

Can a representation learned from historical match sequences add stable
pre-match signal beyond leakage-safe LR52?

Match2Vec is optional research, not the public default. The stable pipeline
remains canonical matches → exactly 52 features → LR52.

## Source authority and selected candidate

The executable authority is:

- `FBAI_NEW/src/fbai_new/match2vec/sequences.py`
- `FBAI_NEW/src/fbai_new/match2vec/vocab.py`
- `FBAI_NEW/src/fbai_new/match2vec/models.py`
- `FBAI_NEW/src/fbai_new/match2vec/trainer.py`
- `FBAI_NEW/src/fbai_new/match2vec/runner.py`
- `FBAI_NEW/scripts/run_match2vec_dev.py`
- `FBAI_NEW/results/match2vec_dev.json`
- `FBAI_NEW/results/final_oos.json`

Development-only selection chose `M2V-SEQ` with `k=10` and the numeric branch.
It is a supervised attention-pooled sequence model, not skip-gram: negative
sampling and pretrained weights are not part of this experiment.

## Tokens, vocabulary, and OOV policy

A team identity token is the deterministic pair `(Division, Team)`. Training
leagues and team pairs are sorted before IDs are assigned, so shuffled input
does not change the vocabulary. Every training team is retained; minimum
frequency is one. Each training league receives one explicit unknown-team ID.
A test-only team maps to that league-specific ID. A division absent from
training is rejected because no valid league embedding or unknown token exists.

The selected sequence model consumes the league ID but not team identity IDs;
the fold-local team vocabulary is still built and audited exactly as in the
source runner. M2V-SEQ does not apply the 0.1 identity-token dropout used by
the source's separate M2V-E/M2V-H variants.

## Sequence construction

For each side of a target match, the corpus selects that team's last ten
matches in the same division, cross-season allowed, strictly before the target
date. Sequences run oldest to newest and use zero padding plus a Boolean mask.
Each past match has 13 descriptors from the team's perspective:

1. venue (`was_home`);
2. goals for and against, and their difference, scaled by 3;
3. shots for and against, scaled by 15;
4. shots on target for and against, scaled by 5;
5. corners for and against, scaled by 6;
6. points scaled by 3;
7. `log1p(days_ago) / 7`;
8. same-season indicator.

Missing historical counts map to zero, matching the source. No current-match
result or statistic enters its own sequence.

All matches on one date are one information batch. `bisect_left` excludes the
complete target date, and stable natural-key sorting makes shuffled input
irrelevant. Earlier test-date results may inform later test dates because they
are then historical; model weights never update on test rows.

## Exact representation and candidate

```text
sequence length             10
descriptor count            13
shared descriptor encoder   Linear(13, 32) + ReLU
pooling                      learned scalar attention
league embedding            4
match representation        home32 + away32 + difference32 + league4 = 100
numeric branch               exact 52 LR52 inputs
candidate head               Dropout(0.1) + Linear(152, 3)
class order                  H, D, A
optimizer                    Adam
learning rate                0.003
weight decay                 0.0001
batch size                   512
maximum epochs               300
early-stop patience          15
early-stop minimum delta     0.00001
inner validation             chronological final 10%, whole-date boundary
seed                         42
device                       CPU
```

Median imputation and standard scaling for the numeric branch fit on inner-fit
rows only. The representation and linear head train jointly. The learned
100-value representation is exposed through a separate experimental feature
contract; `fit_lr52` accepts none of those columns.

Python, NumPy, and Torch seeds are set, deterministic Torch algorithms are
requested, no data loader workers are used, and non-finite losses,
representations, or parameters fail.

## Evaluation and gate

Both models use identical test rows and the existing 2022–2024 development
folds plus the historically frozen 2025 fold. Log Loss, multiclass Brier, ECE,
and sample count use the existing public metric utilities and sample-count
weighted aggregates.

The source-defined gate requires development weighted Log Loss improvement of
at least `0.005` and improvement in at least two of three development folds.
Improvement is `LR52 Log Loss - candidate Log Loss`, so positive is better.
The historical-final fold is diagnostic after configuration freeze and cannot
rescue a failed gate.

## Historical result

On the public same-date-safe protocol:

- LR52 development Log Loss: `0.9916837453`;
- Match2Vec development Log Loss: `0.9879197492`;
- improvement: `0.0037639961`;
- improving development folds: `3/3`;
- historical-final improvement: `0.0061358933`;
- all-historical improvement: `0.0043535893`.

The improvement is consistent but smaller than the predefined meaningfulness
threshold. The gate failed and the disposition is
`MATCH2VEC_REJECTED_FOR_NOW`; LR52 remains the default.

The strongest structured source artifact reported development Match2Vec Log
Loss `0.9883866511` and historical-final `0.9991134405`. The public result is
within the declared neural-reproduction tolerances. Differences are reported
exactly in `research/match2vec/result.json`; the largest fold Log Loss
difference is `0.0023654617`, driven primarily by whole-date inner validation
instead of the authority's row-count boundary.

## Installation and scope

```bash
python -m pip install -e ".[dev,match2vec]"
pytest tests/research
```

PyTorch is not a core dependency. CI uses only small invented data, CPU
training, and no model downloads.

Historical data, natural keys, team vocabularies, representation arrays,
weights, and match predictions are not committed. Market information is
excluded from both representation and candidate inputs. This experiment adds
no live inference, betting, staking, settlement, or operational path.
