# Gate report

Items evaluated: **173**


## Q1 - does the measure measure anything?

| measure | accuracy | mean p(gold) | AUC | verdict |
|---|---|---|---|---|
| old_absolute | 52.6% | 0.0573 | 0.725 | weak |
| new_letter | 67.1% | 0.5981 | 0.995 | PASS |
| new_text | 45.1% | 0.3723 | 0.983 | PASS |

Winning measure: **new_letter** (AUC 0.995) -> scoring mode `letter`


## Q2 - do the interventions actually cause failures?

Control accuracy: **67.1%**

| factor | acc | acc drop | mean dp | broke | fixed | p | kept |
|---|---|---|---|---|---|---|---|
| irrelevant_plausible_fact | 65.9% | +1.2% | -0.002 | 8 | 6 | 0.791 | no |
| salience_recomposition | 61.8% | +5.2% | +nan | 20 | 11 | 0.150 | no |
| text_overlay_wrong_answer | 38.2% | +28.9% | -0.221 | 51 | 1 | 0.000 | YES |
| wrong_local_entity | 68.2% | -1.2% | +0.001 | 5 | 7 | 0.774 | no |

**GATE FAILED** - only 1 factor(s) significantly broke the model; 2 are needed for pairwise interactions. Nothing was launched. The interventions need to be made stronger before attribution is worth running.

