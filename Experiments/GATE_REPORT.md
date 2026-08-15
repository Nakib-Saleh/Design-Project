# Gate report

Items evaluated: **700**


## Q1 - already settled

Measure validation was completed by the instrument check: `new_letter` AUC **0.995** (vs `old_absolute` 0.725) on 173 items. Scoring mode is fixed to `letter`.


_Excluded 2 item(s) with degenerate control scoring._


## Q2 - do the interventions actually cause failures?

Control accuracy: **66.6%**

| factor | acc | acc drop | mean dp | broke | fixed | p | kept |
|---|---|---|---|---|---|---|---|
| irrelevant_plausible_fact | 64.9% | +1.7% | -0.015 | 36 | 24 | 0.155 | no |
| salience_recomposition | 61.3% | +5.3% | -0.058 | 80 | 43 | 0.001 | YES |
| text_overlay_wrong_answer | 33.2% | +33.4% | -0.258 | 238 | 5 | 0.000 | YES |
| wrong_local_entity | 65.5% | +1.1% | -0.002 | 19 | 11 | 0.200 | no |

Factors kept for the attribution run: **text_overlay_wrong_answer, salience_recomposition**


**GATE PASSED** - attribution run launched.


## Launched

- `nakibsaleh/culprit-vqa-attribution-v2-a` (account NakibSaleh) - https://www.kaggle.com/code/nakibsaleh/culprit-vqa-attribution-v2-a
- `gmnoorlabib/culprit-vqa-attribution-v2-b` (account chotonunulabib) - https://www.kaggle.com/code/gmnoorlabib/culprit-vqa-attribution-v2-b

Scoring mode: `letter`  
Factors: text_overlay_wrong_answer, salience_recomposition

## Results pulled

| kernel | status | rows | local file |
|---|---|---|---|
| `nakibsaleh/culprit-vqa-attribution-v2-a` | complete | 500 | `kaggle_kernels\attribution_v2\output\attribution_v2_a.jsonl` |
| `gmnoorlabib/culprit-vqa-attribution-v2-b` | complete | 500 | `kaggle_kernels\attribution_v2\output\attribution_v2_b.jsonl` |

**Total attribution records: 1000**

Analysis note: filter on `control_correct == true` for the headline attribution claim -- on control-wrong items there was no working answer for a factor to break.
