# Player-availability audit design

## Research question

Can genuinely pre-kickoff squad continuity and expected-availability proxies
add stable signal beyond LR52? This is an isolated historical experiment;
LR52 is unchanged and remains the default model.

## Authority and timing correction

The executable authorities are `availability.py` and
`run_availability_eval.py`, followed by the source test, structured result,
and three availability reports. The authority defines 63 strictly-prior
features and a separate 48-feature target-lineup branch.

The raw lineup artifact does not record publication timestamps. Consequently,
actual target starters and bench membership are timing-unknown and forbidden
in this public predictive audit. Target participation, minutes, and
substitutions are post-match facts and are also forbidden. No clean historical
injury/suspension interval table exists, so missing lineup rows are never
relabelled as injuries or absences.

## Strict-prior feature contract

The 21 source semantics are emitted for home, away, and home-minus-away:

- match count in 30 days and total player-minutes in 7/14/21/30 days;
- days since the previous match;
- unique players over 5/10 matches and unique starters over 5;
- top-5 and top-11 minute shares over 5 matches, plus top-11 over 10;
- average minutes per observed player over 5 matches;
- last-two-XI overlap, Jaccard similarity, rotation index, and average starter
  changes over the last 5 lineups;
- strictly earlier valuation sums for recent players, recent top 11, and last
  starters, plus recent valuation coverage.

All histories are keyed by division and team, reset at each season, and use
only dates strictly earlier than the target. A whole calendar date is
transformed before any row on that date enters history. Valuations use a
left-bound lookup, so same-date and future values cannot enter a target.
Missing history remains missing and is handled by fold-local median imputation.

## Alignment, model, and gate

Availability data stays separate from canonical matches. Natural match keys
and game IDs must be unique; player relations and valuations must have unique
identifiers; minutes, lineup roles, and valuations are validated. Only exact
natural-key coverage is allowed. Candidate and LR52 are fitted and scored on
the same covered rows.

The candidate has the exact stable 52 plus 63 availability fields. It uses
training-fold median imputation, standard scaling, and L2 `lbfgs` Logistic
Regression with `C=1`, `max_iter=2000`, tolerance `1e-4`, seed 42, and explicit
`H, D, A` output ordering. There is one frozen candidate and no search.

```text
candidate_improvement_log_loss =
    lr52_aligned_log_loss - availability_candidate_log_loss

required:
    development improvement >= 0.005
    and at least 2 of 3 development folds improve
```

The 2025 historical-final fold is evaluated once after development. It cannot
rescue a failed gate.

## Retained record

Only aggregate input-quality, coverage, fold, metric, gate, timing, and
reproduction metadata is retained. Player identifiers, lineup rows, raw data,
predictions, learned parameters, models, and machine paths are excluded.
