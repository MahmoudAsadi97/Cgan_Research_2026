# Technical notes

This document holds the full technical record behind the [README](../README.md):
the detailed results, the data preparation, the models and their selection, the
ablation ladder, the metric definitions, the comparison with the published
figures and the protocol notes.  File paths are relative to the repository root.

In the tables and file names, `faithful` and "GitHub cGAN" denote the reference
model (the published code of Li, Dong & Qiu, 2025,
[malvin0130/cGAN_EB](https://github.com/malvin0130/cGAN_EB)) and `final` the
final model.

## Detailed results

Held-out **test households** (354 households, 128,826 days); `final`: five
training seeds, validation-selected checkpoints (`best`) and final iterates
(`last`); `faithful` (the GitHub code): three training seeds.  Accuracy in
percent; ± is the standard deviation over seeds.

| model | statistical accuracy | worst label | labels ≥ 85 % | rFD ↓ | C2ST AUC (0.5 ideal) | coverage ↑ | DCR ratio (≈ 1) | negative load, no PV ↓ |
|---|---|---|---|---|---|---|---|---|
| GitHub cGAN [last] (as the reference) | 48.3 ± 1.3 | 27.4 ± 5.5 | 0 % | 1.40 | 1.00 | 0.04 | 2.00 | 21.9 % |
| GitHub cGAN [best] | 55.1 ± 0.6 | 39.8 ± 4.1 | 0 % | 0.91 | 1.00 | 0.06 | 1.51 | 18.2 % |
| **final [best]** | **87.9 ± 0.2** | 74.6 ± 0.6 | 82.5 % | **0.117** | 0.61 | 0.81 | 1.01 | **0** |
| final [last] | 87.6 ± 0.2 | 70.8 ± 1.1 | 82.2 % | 0.129 | 0.59 | 0.82 | 1.00 | 0 |
| real: validation households | 86.1 | 76.9 | 59.4 % | 0.167 | 0.50 | 0.79 | 1.02 | 0 |
| real: training households | 87.9 | 70.6 | 82.8 % | 0.132 | 0.49 | 0.82 | – | 0 |

* **The 85 % target is met** by every seed (87.72, 88.00, 88.12, 87.61,
  87.88 %).  With the household-level uncertainty (`results/uncertainty.md`)
  the accuracy is 87.9 % (95 % interval 86.4–89.3 %, one-sided 95 % lower
  bound 86.7 %).  The final model matches the test households 1.8 points
  better than real days of the 356 validation households (paired 95 %
  interval +0.6 to +3.1) and as well as real days of the 1,657 training
  households (−0.1, interval −0.8 to +0.7); it is 39.5 points above the
  GitHub model [last] (38.3 to 40.7).
* **Better than the GitHub model on every ranked metric** (the reference
  code's KLD is descriptive only, see *Comparison with the GitHub model*),
  each with complete separation: every final seed beats every GitHub value
  (`results/comparison.md`).  The GitHub code trained on these data
  oscillates and does not converge (`results/figures/validation_curves.png`;
  the dotted line there is the training-household benchmark scored on the
  validation households); its average days zig-zag from hour to hour
  (`results/figures/compare/`).
* **What makes the difference** (`results/ablation.md`): two changes in how
  the data are fed to the networks — minibatches reshuffled every epoch
  (+11.2 points) and the per-hour value transform (+21.1) — lift the
  reference networks with their BCE training from 54.8 % (seed 0; 55.1 ± 0.6
  over three seeds) to 87.2 % (one seed, validation-selected checkpoint).  The
  WGAN-GP recipe with wider networks, EMA and the hour-to-hour critic input
  adds 0.5–0.7 points of accuracy (seed 0: +0.55, within the seed spread of
  the BCE runs) and mainly makes single days more realistic: C2ST AUC 0.70 →
  0.61, rFD 0.137 → 0.117, coverage 0.80 → 0.81.  With the EMA read-out the
  validation accuracy drifts down by at most 0.8 points after its peak,
  against 2.0 points for the BCE model (88.0 % at epoch 1,450, 86.0 % at
  epoch 10,000).  The no-PV floor removes negative load without PV (0.15 % of
  the hours with BCE, 0.39–0.87 % in the WGAN steps before the floor).  The
  BCE model is better on the worst label (78.9 % against 74.6 %), the
  key-parameter FID (0.69 against 0.85), the near-base KL (0.070 against
  0.080) and the key-parameter std (90.7 % against 90.1 %).
* **Classical generators** (*Classical baselines*, `results/baselines.md`): a
  Gaussian copula fitted per label reaches 85.7 ± 0.2 % and a Gaussian mixture
  85.8 ± 0.1 %; drawing every hour independently reaches 78.4 %.  The final model
  is ahead of all of them on the key parameters, the classifier test and
  coverage, with every seed.  In statistical accuracy it is 2.2 points above
  the copula (paired 95 % interval over households 1.5 to 2.8) and 2.0 points
  above the mixture (1.3 to 2.7).
* **Acceptance levels of the paper** (`results/paper_criteria.md`, test
  households, share of label × key-parameter pairs, with the paper's
  definitions and a proper KL; the reference-code definitions are in the same
  table; GitHub averaged over its three seeds): |difference of mean| < 1 —
  GitHub [last] 54 %, final 99 %;
  |difference of std| < 1 — 78 % / 100 %; KL < 0.3 — 7 % / 100 %; FID < 5 —
  71 % / 100 %.  With generated days against 2,000 days per label of the
  training households (`results/paper_protocol.md`) the final model reaches
  93.2 % accuracy and meets every level for 100 % of the pairs with the
  paper's definitions (99 % for KL < 0.3 with the reference code's KLD).
* **Against the published figures of the paper** (*Comparison with the
  published results*; other data, so levels and no ranking): with the reference
  code's definitions, the mean of each time parameter per label is within 1 h of
  the real mean for 100 % of the pairs with the paper's protocol (median 0.18 h;
  paper 79.7 %, 0.50 h) and for 98.8 ± 0.5 % on held-out households (0.20 h), which
  the paper does not test; the GitHub model reaches 39 %.  These levels are
  necessary, not sufficient: real days of other households meet them too, and a
  stronger classifier still separates generated from real days of the test
  households (C2ST AUC 0.70 against 0.51 for real days, `results/reliability.md`).

Figures (`results/figures/`, seed 0): the paper's Figs. 3–12 for the final
model (`final_best/`; files with `_code` use the definitions of the reference
code) and for the GitHub model (`faithful_last/`), plus `temp_dist_<key>.png`
with the distribution of each key parameter in all 64 labels; both models in
one plot (`compare/`); and the validation curves of every finished seed
(`validation_curves.png`).
`results/figures/final_best/*_gen.csv` are 2,000 generated days per label (kWh
per hour), as published by the reference repository.

## Dataset (`data_prep.py`)

`raw.zip` holds eight CSV files of 300 residential meters each, one per
combination of PV (*ZP*), heat pump as main heating (*WP*) and home-charged
electric vehicle (*EV*), with quarter-hourly grid import
(`Volume_Afname_KWh`) and export (`Volume_Injectie_KWh`) for all of 2024 and
the three indicators per meter.  `data_prep.py` aggregates the quarter hours to
hourly energy and builds daily profiles as in the paper (Sec. 2.1–2.2):

| | |
|---|---|
| Target | hourly **net load** = import − export (kWh); negative while PV exports |
| Day type | weekday / weekend, Belgian public holidays counted as weekend |
| Season | spring MAM, summer JJA, fall SON, winter DJF |
| Group | PV, heat-pump and EV indicators: `No PV HP EV`, `PV`, `HP`, `PV+HP`, `EV`, `PV+EV`, `HP+EV`, `PV+HP+EV` |
| Label | `daytype · 32 + group · 4 + season` → 64 labels, the layout of the reference |

**Source.**  The archive holds the two parts of the 2024 sample
(`P6269_1_50_DMK_Sample_Elek_2024_Deel_1` and `_Deel_2`) and the legend of the
Fluvius open data set *Verbruiksprofielen digitale elektriciteitsmeters:
kwartierwaarden voor een volledig jaar*
([opendata.fluvius.be](https://opendata.fluvius.be/explore/dataset/1_50-verbruiksprofielen-dm-elek-kwartierwaarden-voor-een-volledig-jaar/),
[open data licence of Fluvius](https://opendata.fluvius.be/p/licentieopendatafluvius)).
According to the legend, the scope is the residential digital meters whose
users asked Fluvius to read the quarter-hourly values and for which all values
of the year are available.  A meter counts as "with PV" when it registered
injection in all 12 months and as "without PV" when it registered none in the
whole year, and meters with a known storage system are left out.  From this
scope Fluvius took 300 meters for each of the eight technology groups.  The
sample is therefore balanced by design and does not represent the mix of
households in Flanders.

**EV labels.** The reference distinguishes EV households by charger level
(Level 1, Level 2, both).  The Fluvius release has no charger information,
only a yes/no EV indicator (an EV declared by the customer or detected by
Fluvius; 0 does not exclude an EV).  The three charger classes are therefore
replaced by the heat-pump indicator, which keeps eight technology groups and
64 labels.

**Cleaning** (the paper removes "daily profiles with missing or incorrect
recordings"):

* timestamps are local time (CET/CEST, see the legend in the archive): the DST
  days 31 March (92 quarter hours) and 27 October (100) do not form a 24-hour
  day and are removed — every kept hour consists of exactly four quarter hours;
* 33 meters have a full-year import and export series identical to another
  meter of the same group under a different `EAN_ID`; only the first copy is
  kept, so that no household can appear in two splits;
* 866 meter-days without any import or export in all 24 hours (vacancy,
  disconnection or zero-filled data) are removed;
* the reference drops days with values < 0 or > 100 kWh.  The > 100 kWh rule is
  kept (it removes nothing); the < 0 rule is not applied because negative net
  load is genuine PV export.

**Households are split** 70 / 15 / 15 % into train / validation / test,
stratified by group (1,657 / 356 / 354 households; 602,356 / 129,540 /
128,826 daily profiles; 598,249 weekday and 262,473 weekend days in total).
The scaler and the models see only the training households, the validation
households select checkpoints and models, and the test households give the
reported numbers.  `results/table1_labels.md` lists the days and households of
every label and split (the counterpart of the paper's Table 1): 5,355–13,440
training days and 199–210 training households per label.  Unlike the paper's
data (316–9,325 days per label), the labels are nearly balanced, so the
paper's explanation of its weakest labels by class imbalance does not arise
here.

**Checks of a local copy.**  `results/household_split.csv` lists the split of
every meter (and the 33 removed copies), so the split does not depend on the
random-number stream of a NumPy version.  `python data_check.py` compares a
local copy with the published data: the SHA-256 of `raw.zip`
(`934cc871…389bd8`), the numbers of removed meters and meter-days, the
households and days per split, the split of every meter and the days per label.

**Meters without PV.**  The legend of the release states that meters without
the PV indicator registered no injection during the year.  Three of the 1,200
such meters do register a little (0.001, 0.012 and 4.7 kWh in the year), and the
net load of one of them, a training household, is negative in 5 hours (minimum
−0.40 kWh).  The validation and test households hold no negative value for
labels without PV.

**Single households.**  Households differ strongly (the mean net load of a
household ranges from −15 to 103 kWh a day, median 9 kWh), and a label of the
test households holds the days of 42–45 households only.
`results/table1_labels.md` therefore also gives, per label and split, the
largest share of one household in the variance of the label (its part of the
squared deviations from the mean day of the label).  For 10 of the 64 labels of
the test households one household carries more than 25 % of the variance (six
labels of the group without technology and four of the heat-pump group), and
for the group without technology in winter two thirds (66.6 % on weekdays and
64.9 % at weekends, against 17.0 % and 15.9 % for the training households).  The
hourly standard deviation of such a label, one of the four terms of the
statistical accuracy, is then set by one household that no generator can know:
real days of the training households score 70.6 % on the weekday winter label
of the group without technology, their lowest label, and 78.4 % on average over
the eight labels of this group against 85.8–92.1 % for the other groups.

The table also gives the share of the variance of a label that lies between
households (every day replaced by the mean day of its household): 37.0 % on
average over the labels for the training households, from 17.1 to 79.6 %
(33.7 % for the validation and 34.1 % for the test households).  The models of
this study, like the model of the paper, draw single days for a label and know
no household.  They are trained and evaluated on the pooled days of a label;
365 days drawn for one household would not share a level, and a set of such
households would differ far less from each other than real households do.

## Models (`cGAN.py`, `train.py`)

### `faithful` — the GitHub code

`Generator`, `Discriminator` and `cGAN` are the reference classes: 24-d noise
concatenated with the 64-d one-hot label; generator 64-128-64 with LeakyReLU
and linear output; discriminator 64-128-256 with Dropout(0.4) and sigmoid;
binary cross-entropy; Adam with lr 1e-4 (G) and 2e-4 (D); batch 2,000;
10,000 epochs; global min-max scaling; minibatches in a fixed order, that of my
data (by technology group, then meter and day, so 98 % of the batches hold one
group).  The reference does not shuffle either, but it trains in the row order of
its data file, which is not public, so this order is an assumption.  Additions
that do not change the training mathematics: seeds, a CUDA graph that replays the
training step (the same operations, ~6× faster), periodic validation,
checkpointing and resuming.
Both the final iterate (`last`, what the reference uses) and the
validation-selected checkpoint (`best`) are evaluated.

### `final` — the final model

The same conditional set-up as the reference — noise and the one-hot label at
the input of both networks, 64 labels — with these changes:

| change | reference | final | why |
|---|---|---|---|
| minibatches | fixed order, every epoch the same | reshuffled every epoch | ordered batches hold one technology group at a time; the networks then chase one group after another |
| value transform | global min-max over all values | per-hour Gaussianisation (empirical CDF of the training data, then the standard-normal quantile) | net load is heavy-tailed (EV and heat-pump peaks up to 24 kWh/h); min-max squeezes ordinary loads of 0.1–1 kWh into a sliver of [0, 1]; the inverse maps every generated value back into the observed range of its hour |
| loss and training recipe | BCE, sigmoid, dropout; Adam lr 1e-4 (G) / 2e-4 (D), default betas; batch 2,000; 10,000 epochs | Wasserstein loss with gradient penalty (λ = 10), critic without sigmoid and dropout, 5 critic steps per generator step; Adam lr 1e-4 for both, betas (0.5, 0.9); batch 1,024; 400 critic epochs | stable training without mode collapse; the BCE training oscillates strongly (`results/figures/validation_curves.png`) |
| networks | G 64-128-64, D 64-128-256, noise 24 | G and critic 4 × 512 LeakyReLU, noise 64; the critic also receives the hour-to-hour changes x[t+1] − x[t] | capacity for 64 labels with very different shapes; generated days of the smaller models were rougher than real days |
| read-out | final weights | exponential moving average (0.999) of the generator weights; checkpoint with the best validation accuracy | averaging removes the oscillation of the iterates; selection uses only validation households |
| physics | — | net load of labels without PV floored at 0 kWh | meters without PV register no injection; without the floor 0.40 % of the generated hourly values of these labels were negative (validation households) |

The hour-to-hour changes are linear in the profile, so giving them to the
critic adds no information: it reparameterises the critic's first layer, which
gives the changes their own weights and changes how the critic is initialised
and optimised.

**Model selection.**  `final` was chosen on the validation households among
four candidates trained with seed 0 (`results/selection.md`):
`dev_improved_nofloor` (the `improved` configuration — 3 × 256 networks,
noise 32 — trained before the no-PV floor existed; the floor changes its
validation accuracy by about 0.02 point), `improved_diff` (+ hour-to-hour
changes for the critic), `improved_big` (4 × 512, noise 64) and
`improved_bigdiff` (both).  Rule, fixed before the comparison: highest
validation accuracy, the simpler model if two are within 0.1 percentage point.
`improved_bigdiff` won (88.70 % against 88.48 % and 88.38 %), and `final` is its
configuration trained with five seeds (seed 0 is the selection run itself:
training is deterministic, so both runs are identical); `selection.md` also
lists the validation accuracy of the five `final` seeds.  The selection-draw
value of `dev_improved_nofloor` (87.88 %) was recorded by the development
version of the validation code; the published code scores the same checkpoint
with the same draw at 88.07 %, which leaves the ranking unchanged (the other
three candidates re-score exactly).  The rule ranked the
best of 40 validation checkpoints scored with one fixed noise draw; scored
again with five fresh noise draws (`results/selection.md`), the two 4 × 512
candidates are within noise of each other for seed 0.  Seeds 1–4 of
`improved_big`, trained after the selection for the ablation and listed in
`selection.md` together with the selection run (seed 0), give the
hour-to-hour changes a small validation advantage (+0.14 ± 0.16 points over
the five seeds with fresh noise draws, seed-matched; higher for 4 of 5
seeds), and the ablation finds +0.09 ± 0.15 points on the test households:
the change is kept, but its effect is within seed noise.

**Ablation ladder** (`results/ablation.md`, validation-selected checkpoints;
seeds 0–4 for `improved`, `improved_big` and `final`, seeds 0–2 for
`faithful` and the other WGAN steps, seed 0 for `abl_shuffle` and
`abl_quantile`).  Each step adds one change to the previous row, and the table
gives the seed-matched difference of test accuracy to the previous step;
`abl_wgan_refnets` switches to the WGAN-GP training recipe as a whole (loss,
critic without sigmoid and dropout, 5 critic steps, optimiser settings, batch
1,024, 400 critic epochs) with the reference networks:

`faithful` → `abl_shuffle` (+ reshuffled batches) → `abl_quantile` (+ value
transform) → `abl_wgan_refnets` (+ WGAN-GP recipe) → `abl_wide` (+ 3 × 256
networks, noise 32) → `abl_ema` (+ EMA read-out) → `improved` (+ no-PV floor)
→ `improved_big` (+ 4 × 512 networks, noise 64) → `final` (+ hour-to-hour
changes for the critic).

| step | seeds | test accuracy | Δ to previous step (pt) | worst label | C2ST AUC | rFD | negative load, no PV |
|---|---|---|---|---|---|---|---|
| `faithful` [best] | 3 | 55.1 ± 0.6 | | 39.8 | 1.00 | 0.913 | 18.2 % |
| `abl_shuffle` | 1 | 66.1 | +11.2 | 55.3 | 1.00 | 0.469 | 13.6 % |
| `abl_quantile` | 1 | 87.2 | +21.1 | 78.9 | 0.70 | 0.137 | 0.15 % |
| `abl_wgan_refnets` | 3 | 83.7 ± 0.3 | −3.8 (seed 0) | 72.2 | 0.76 | 0.200 | 0.87 % |
| `abl_wide` | 3 | 87.1 ± 0.2 | +3.43 ± 0.54 | 74.8 | 0.68 | 0.119 | 0.48 % |
| `abl_ema` | 3 | 87.5 ± 0.3 | +0.36 ± 0.41 | 75.1 | 0.65 | 0.117 | 0.39 % |
| `improved` | 5 | 87.5 ± 0.2 | −0.00 ± 0.01 (seeds 0–2) | 76.0 | 0.65 | 0.117 | 0 |
| `improved_big` | 5 | 87.8 ± 0.1 | +0.26 ± 0.20 | 75.2 | 0.62 | 0.117 | 0 |
| `final` | 5 | 87.9 ± 0.2 | +0.09 ± 0.15 | 74.6 | 0.61 | 0.117 | 0 |

The data-handling changes carry most of the gain; `abl_shuffle` and
`abl_quantile` have one seed each, and the BCE runs of `faithful` have seed
SDs of 0.6 (best) and 1.3 (last) points (ranges 1.1 and 2.5), so their single
values are less certain than the WGAN rows.  With the reference networks the
WGAN-GP recipe alone scores lower than BCE (within its 400 critic epochs;
seeds 0 and 1 are still at their maximum at epoch 400); with the wider
networks it reaches the BCE level (87.1 ± 0.2 against 87.2 %) and passes it
only with the EMA read-out and the 4 × 512 networks (87.5–87.9 %).  Without
EMA its validation accuracy fluctuates about as much as BCE's and ends
0.5–1.1 points below its peak; with EMA it drifts down by at most 0.8 points
after its peak, against 2.0 points for the BCE model.  The EMA and the
no-PV floor change only how generated days are read out and which checkpoint
is selected: with the same seed, `abl_wide`, `abl_ema` and `improved` follow
the same training trajectory, so their differences are not affected by
training noise.

## Classical baselines (`baselines.py`)

Four generators without a neural network show what the conditional GAN adds.
Each is fitted per label on the days of the training households and then
treated like a trained model: it produces as many days per label as the test
households have, and `evaluate.evaluate` computes the same metrics
(`results/baselines.md` and `.csv`, five seeds; `python baselines.py` runs on a
CPU and needs no trained model).

| generator | fitted per label | reproduces by construction |
|---|---|---|
| independent hours | every hour is drawn on its own from the training days | the distribution of every hour; no relation between the hours of a day |
| multivariate normal | mean day and covariance matrix of the 24 hours, in kWh | means, standard deviations and linear correlations; its values are symmetric and unbounded |
| Gaussian copula | empirical distribution of every hour, joined by the correlation matrix of the normal scores | the distribution of every hour and the rank correlation between the hours |
| Gaussian mixture | K normal components with full covariances in the per-hour Gaussianised space of the final model, with its floor of 0 kWh for labels without PV | several kinds of days within a label |

A seed fixes the sampling noise and, for the mixture, the initialisation of the
fit; the other fits are deterministic, so their spread over seeds is sampling
noise only.  The number of mixture components is chosen on the validation
households among 1, 2, 5, 10, 20, 40, 80 and 160, with the rule of the cGAN
candidates (highest validation accuracy, the smaller mixture if two are within
0.1 point): the validation accuracy is 83.1, 84.2, 85.0, 85.3, 85.7, 86.2, 86.7
and 87.2 %, so the mixture has 160 components.  The accuracy still rises at the
end of the grid.  The grid ends there because a component of the smallest label
then holds 33 training days on average, hardly more than the 24 values of a
day.  On the same validation days the Gaussian copula scores 86.7 % and the
final model 88.5 % (seed 0 with fresh noise draws, `results/selection.md`).

| generator | accuracy | hourly part (of 62.5) | key-parameter part (of 37.5) | worst label | FID (5 KP) ↓ | rFD ↓ | C2ST AUC | precision | recall | coverage ↑ | negative load, no PV |
|---|---|---|---|---|---|---|---|---|---|---|---|
| independent hours | 78.4 ± 0.1 | 52.9 | 25.4 | 59.9 | 11.7 | 0.481 | 0.99 | 0.29 | 0.81 | 0.18 | 0 |
| multivariate normal | 74.7 ± 0.1 | 51.0 | 23.7 | 52.8 | 37.8 | 0.126 | 0.99 | 0.36 | 0.85 | 0.19 | 23.4 % |
| Gaussian copula | 85.7 ± 0.2 | 53.0 | 32.7 | 67.4 | 2.04 | 0.141 | 0.80 | 0.76 | 0.90 | 0.62 | 0 |
| Gaussian mixture (160 components) | 85.8 ± 0.1 | 52.2 | 33.7 | 68.4 | 1.43 | 0.147 | 0.68 | 0.79 | 0.89 | 0.77 | 0 |
| **final [best]** | **87.9 ± 0.2** | 53.3 | 34.6 | 74.6 | 0.85 | 0.117 | 0.61 | 0.82 | 0.88 | 0.81 | 0 |
| real: validation households | 86.1 | 51.6 | 34.5 | 76.9 | 0.75 | 0.167 | 0.50 | 0.83 | 0.83 | 0.79 | 0 |
| real: training households | 87.9 | 53.0 | 35.0 | 70.6 | 0.50 | 0.132 | 0.49 | 0.83 | 0.88 | 0.82 | 0 |

The accuracy is the sum of two parts, given here in points: the hourly part
(mean profile + std profile + hourly 1−KS / 2) / 4, which depends only on the
distribution of each hour, and the key-parameter part
((KP mean + KP std) / 2 + KP 1−KS / 2) / 4, which depends on the shape of the
days.

* **The accuracy has a narrow range between generators that copy the hourly
  distributions.**  Such a generator gets about 53 of the 62.5 points of the
  hourly part, as real days of the training households do (53.0): what is
  missing is the difference between the training and the test households, which
  no generator can know.  The models differ in the key-parameter part, from
  25.4 points without any relation between the hours to 35.0 for real days of
  the training households.  On this data set the accuracy of a generator with
  the right hourly distributions therefore lies between 78.4 % and about 88 %,
  and the copula and the mixture pass the 85 % target.
* **The final model is ahead on the shape of the days.**  `baselines.md`
  compares the final model with every generator metric by metric.  Every seed
  of the final model is better than every seed of the copula and of the
  mixture on the key-parameter terms, the key-parameter FID and KL, the worst
  label, rFD, the C2ST AUC, precision, coverage and the DCR ratio.
  The exceptions: independent hours and the copula reuse the training values
  of every hour and are 0.2 point ahead on the hourly 1−KS (93.6 against
  93.5 %); the copula is within the seed spread of the final model on the mean
  profile and the hourly standard deviation; and the copula and the mixture
  have a higher recall (0.90 and 0.89 against 0.88) with a lower precision
  (0.76 and 0.79 against 0.82), so they spread wider than the real days.  The
  mixture is closer to the final model than the copula on the shape of the
  days (key-parameter FID 1.43 against 2.04, C2ST AUC 0.68 against 0.80,
  coverage 0.77 against 0.62) but loses on the hourly standard deviation
  (76.4 % against 78.8 %).  Its DCR ratio of 1.02 gives no sign that the many
  components copy training days.
* **The acceptance levels of the paper do not separate the models.**  With the
  definitions of the paper the copula meets them for 94, 100, 98 and 100 % of
  the pairs and the mixture for 97, 100, 99 and 100 % (|difference of mean| < 1,
  |difference of std| < 1, KL < 0.3, FID < 5), against 99, 100, 100 and 100 %
  for the final model.
* **rFD and the correlation error see only means and covariances.**  The
  multivariate normal reproduces both by construction: its rFD (0.126) and
  correlation error (0.071) are close to those of the final model (0.117 and
  0.071), while its days are the least realistic of all (C2ST AUC 0.99, 23 % of
  the values of labels without PV negative).
* **Household uncertainty** (`results/uncertainty.md`): with the half-sampling
  of the test households the final model is 2.16 points above the copula
  (standard error 0.34, 95 % interval 1.5 to 2.8) and 2.02 points above the
  mixture (0.37, 1.3 to 2.7).  In 20 other splits of the households the copula
  stays 2.2 ± 0.2 points below real days of the training households
  (`results/split_sensitivity.md`); the final model is level with this
  benchmark in the split of the study (−0.1, interval −0.8 to +0.7).
  `improved`, trained again on eight of these splits, is 0.6 to 2.6 points
  above the copula of its split and between 1.6 points below and 0.6 points
  above the benchmark (see "Trained models on other splits" below).

## Paper, reference code and this implementation

| item | paper (text) | reference code | `faithful` | `final` |
|---|---|---|---|---|
| data | 110 US households with PV and EV, 2013–2019, hourly | — | Fluvius 2024, 2,367 Flemish households | same |
| labels | day type × 8 PV / EV-charger statuses × season | same | day type × 8 PV / HP / EV groups × season | same |
| value filter | "missing or incorrect recordings" removed | days with values < 0 or > 100 kWh dropped | > 100 kWh rule only (net load) | same |
| normalisation | divided by the maximum of the data set | min-max with global min and max | min-max, training households | per-hour Gaussianisation, training households |
| data used for validation | — (same data as training) | training data | held-out test households | same |
| minibatch order | — | file order | data order (group, meter, day) | reshuffled every epoch |
| loss | BCE (Eqs. 1–3) | BCE, D loss = (real + fake) / 2 | as the code | WGAN-GP |
| networks | Table 2 (the 88 units of "layer 1" are the 24 + 64 inputs) | G 64-128-64, D 64-128-256, dropout 0.4 | as the code | 4 × 512, critic with hour-to-hour changes |
| training | Adam 1e-4 / 2e-4, 10,000 epochs, batch 2,000 | same | same | Adam 1e-4, betas (0.5, 0.9), 400 critic epochs, batch 1,024, EMA |
| weights used | — | after the last epoch | `last` and `best` | `best` (EMA) |
| generated days per label | 2,000 | 2,000 | test-day count for the metrics; 2,000 for the figures | same |
| key parameters | near-peak / near-base = 97.5th / 2.5th percentile; rise and fall time after Price (2010) | 2nd-highest / 2nd-lowest hour; rise and fall as \|argmax − argmin\| of hour indices | both reported | both reported |
| KL | KL(P ‖ Q), Eq. 4 | `rel_entr` on two sorted, index-aligned pandas Series (pairs day *i* with day *i*) | both reported | both reported |
| FID | Eq. 5 on the five key parameters; figures per parameter | per parameter, 1-D | both reported | both reported |

"Both reported": the figures and `paper_criteria.md` come in two versions, with
the definitions of the paper's text (and a proper KL) and exactly as the
reference code computes them (`metrics.reference_key_parameters`,
`metrics.reference_kld`); the second uses the definitions of the published code,
which are not in every respect those of the published figures (see *Comparison
with the published results*).

## Comparison with the published results

`compare_paper.py` repeats the validation of the paper's Figs. 5–6 and 9–12 for
all training seeds, with the definitions of the published reference code (see
*Near-peak and near-base* below; `results/paper_comparison.md`, every seed in
`results/paper_comparison.csv`).  The paper's values are digitised from its
heatmaps; the digitised cells and the paper's figures are not part of this
repository.  *Paper protocol*: 2,000 generated days per label against all days
of the training households, as the reference evaluates itself.  *Held-out*:
against the test households, a test the paper does not make.  Each cell gives
the share of (label, key parameter) pairs that meet the level and, in brackets,
the median over the pairs (mean over seeds, rounded); the rows with the
scale-matched bar have no median, because the bar differs per label.  Undefined
"KLD" values fail the level and rank last in the median.  ± is the SD over seeds
(five for `final`, three for the GitHub model) and does not include which
households are in the test set; the real-household columns are one random draw
of days.

| criterion | paper (in-sample, its data) | `final` [best], paper protocol (in-sample) | `final` [best], held-out | GitHub [last], paper protocol (in-sample) | GitHub [last], held-out | real validation households, held-out | real training households, held-out |
|---|---|---|---|---|---|---|---|
| time parameters: \|Δ mean\| < 1 h | 79.7 % (0.50) | 100 % (0.18) | 98.8 ± 0.5 % (0.20) | 38.5 ± 5.9 % (1.44) | 39.1 ± 5.9 % (1.41) | 98.4 % (0.20) | 99.5 % (0.20) |
| time parameters: \|Δ std\| < 1 h | 99.0 % (0.30) | 100 % (0.12) | 100 % (0.18) | 65.6 ± 6.6 % (0.69) | 63.9 ± 6.7 % (0.71) | 100 % (0.17) | 100 % (0.13) |
| time parameters: FID < 5 | 100 % (0.49) | 100 % (0.07) | 100 % (0.11) | 54.5 ± 3.5 % (4.07) | 52.3 ± 4.8 % (4.60) | 100 % (0.10) | 100 % (0.07) |
| loads: \|Δ mean\| < 1 kWh | 99.2 % (0.17) | 100 % (0.02) | 100 % (0.06) | 94.8 ± 2.5 % (0.47) | 94.0 ± 3.2 % (0.49) | 100 % (0.08) | 100 % (0.06) |
| loads: \|Δ std\| < 1 kWh | 99.2 % (0.15) | 100 % (0.05) | 100 % (0.08) | 98.4 ± 2.1 % (0.30) | 99.0 ± 1.2 % (0.30) | 100 % (0.08) | 100 % (0.11) |
| near-peak: \|Δ mean\| below the scale-matched bar | 98.4 % | 100 % | 93.8 ± 2.7 % | 46.4 ± 2.4 % | 39.1 ± 1.6 % | 81.2 % | 92.2 % |
| near-base: \|Δ mean\| below the scale-matched bar | 100 % | 100 % | 90.0 ± 0.9 % | 27.1 ± 10.2 % | 29.2 ± 10.0 % | 84.4 % | 90.6 % |
| near-peak: \|Δ std\| below the scale-matched bar | 100 % | 80.9 ± 3.2 % | 73.8 ± 3.2 % | 50.5 ± 5.0 % | 52.1 ± 10.4 % | 68.8 % | 78.1 % |
| near-base: \|Δ std\| below the scale-matched bar | 100 % | 99.4 ± 0.9 % | 93.8 ± 2.5 % | 46.9 ± 7.8 % | 52.1 ± 8.6 % | 98.4 % | 92.2 % |
| all parameters: FID < 5 | 100 % (0.33) | 100 % (0.03) | 100 % (0.06) | 72.5 ± 2.0 % (0.95) | 71.1 ± 2.7 % (1.02) | 100 % (0.07) | 100 % (0.04) |
| all parameters: code "KLD" ≤ 0.3 (descriptive) | 100 % (0.07) | 99.1 ± 0.2 % (0.10) | 97.7 ± 0.4 % (0.11) | 90.5 ± 3.4 % (0.09) | 90.8 ± 2.9 % (0.10) | 97.8 % (0.10) | 98.8 % (0.11) |

The time parameters do not depend on the load scale, but cells of PV labels
still show a different quantity (see below), so only the group without
technology has the same label and quantity in both data sets, and even there the
households differ (median near-peak std / mean 1.33 against 0.57).  On the time
rows `final` reaches the paper's
levels with both protocols, with a median error of the mean of 0.18–0.20 h
against the paper's 0.50 h, close to real days of other households.  The GitHub
code trained on these data misses them.

`final` is below the paper's share on the load rows with the scale-matched bar
held-out and on the code's "KLD".  The two real pools do not reach these shares
either: real days of the 1,657 training households, what a copy of the training
data would score, stay at the same level (held-out 92.2, 90.6, 78.1 and 92.2 % on
the four load rows against 93.8, 90.0, 73.8 and 93.8 % for `final`; "KLD" 98.8 %
against 97.7 %), and the validation households reach 81.2, 84.4, 68.8 and 98.4 %,
so of the four load rows only on the near-base std does a real pool come close
to the paper.  Every "KLD" failure of every `final` seed is the near-base load of
a PV label (see below).  With the paper protocol the "KLD" stays below the paper
(99.1 ± 0.2 % against 100 %, real days 99.1–99.4 %) and two load gaps remain: the
near-peak std, 80.9 % against 98.4 % for a sample of the training days themselves,
while real days of the validation households reach 71.9 %, and the near-base std,
99.4 ± 0.9 % against 100 %.  The near-peak std compares least well across the data sets: the
near-peak varies twice as much relative to its level here (median std / mean 1.00
against 0.45), so a bar set on the level is stricter for the spread here than in
the paper.

What limits this comparison:

* **Near-peak and near-base.**  The table uses the definitions of the published
  code, like `metrics.reference_key_parameters` and the `_code` figures: the
  second-highest and second-lowest hour of a day.  The paper's text gives the 97.5th and 2.5th percentile, and the
  generated days in its Figs. 7–8 are the third-highest and fourth-lowest hour of
  the published days.  The heatmaps themselves cannot be tested, because the real
  days are not public.  Repeated with those ranks
  (`results/figure_ranks/paper_comparison.md`), `final` reaches 100 % and
  99.9 ± 0.2 % (medians 0.13 and 0.23 h) on the first row of the table, the GitHub
  model 45.5 ± 6.8 % and 44.3 ± 7.3 % (1.16 and 1.20 h), and the conclusions stay
  the same.
* **Different data.**  The paper's data are not public, so the table shows
  whether the levels the paper reports are reached here; it is not a ranking.
  The ranking on common data is the comparison with the GitHub model, trained
  on the same households.
* **Load scale.**  The paper's loads are larger: the median near-peak of a label
  is 5.13 kWh in its real data (the third-highest hour of the generated days
  published with the reference, which are those of its Figs. 3–4 and 7–8, minus the
  digitised difference) against 1.28 kWh here at the same rank (1.59 kWh at the
  second-highest hour used in the table), and the median daily energy 60.9 kWh
  in those generated days against 10.3 kWh net here (about 15 kWh on labels without
  PV, 4 kWh with PV).  1 kWh is therefore a lenient level here: the GitHub model
  stays below it for 94–95 % of the pairs of |Δ mean|, with a median of about
  0.5 kWh (near-peak and near-base pooled).  The scale-matched bar is the paper's
  1 kWh as a share of its median near-peak (19.5 %, about 0.31 kWh here), applied
  per label to the four load rows.  The medians of the load rows and of "all
  parameters: FID" are not scale-matched (FID of loads is in kWh²).
* **Different quantity on PV labels.**  The paper's profiles do not appear to
  show net load, although it describes PV as lowering the demand from the
  utility: the reference code drops days with values below 0, and in the
  published days the PV-only labels use more energy than those without
  technology (median 61 against 44 kWh a day) and peak at 12–13 h (24 of the 32
  PV labels at 12–15 h, the other eight, all with an EV charger, in the evening
  or at night).  Here the target is net load, and most days of PV labels have a
  negative near-base load.  Cells of PV labels are not like-for-like.
* **Labels.**  Only the first group (no technology) has the same label and
  quantity in both data sets.  PV only has the same label but not the same
  quantity (see above);
  the other six are EV charger levels there and heat-pump or EV flags here.  The
  paper's groups are adoption phases of the same 110 homes, mine are different
  households.  On the first group alone, |Δ mean| of the time parameters is below
  1 h for 91.7 % of the pairs in the paper (median 0.42 h), 100 % for `final` with
  both protocols (0.23 h, 0.19 h) and 15.3 ± 15.8 % and 18.1 ± 13.4 % for the
  GitHub model (3.50 h, 3.25 h).  With PV only added: 89.6 %; 100 % and
  96.7 ± 1.9 %; 25.0 ± 13.7 % and 26.4 ± 11.5 %.
* **The code's "KLD" is no divergence.**  It pairs real day *i* with generated
  day *i* (with the paper protocol only the first 2,000 training days of a label,
  in household order, are paired; the others only set the normalising maximum)
  and drops infinite terms.  A negative near-base load makes terms infinite, for
  PV labels in spring and summer nearly all of them, so the value rests on a few
  pairs and can exceed 0.3, or on none: it is then undefined.  The level says
  little about quality.  Identical samples miss 100 % on these data, because a
  negative near-base load leaves few finite pairs: a shuffled copy of the test days
  scores 96.9–99.7 % over 40 shuffles, never 100 % (median 0.11), of the days of
  `final` 97.8–99.1 % over seeds (0.09) and of all training days 98.4–100 %
  (100 % in 7 of 40 shuffles; 0.10), while the published days of the reference,
  which hold no negative load, score 100 % in all 40 (0.08).  Wrong days come close
  to real days, which score 99.1–99.4 % with the paper protocol: flat days (the mean
  of each day plus noise of SD 0.001 kWh, so that the key parameters are defined)
  reach 96.9 % and training days with the heat-pump flag flipped 99.1 %; with the PV
  flag flipped they fall to 88.8 %, close to the GitHub model.  The proper KL of
  `results/paper_criteria.md` does separate the models (held-out, below 0.3 for
  100 % of the pairs for `final`, 7 % for the GitHub model).
* **The paper's own figures.**  The paper states differences below 1 kWh, while
  two of its 256 load cells are above that, and the means of its time parameters
  are within 1 h in 79.7 % of the cells (all three for 51.6 % of the labels; means
  and std together 89.3 % of the cells).  Reaching its level means reaching these
  shares.
* **What the levels see.**  The time rows see the PV flag: with it flipped, |Δ mean|
  is below 1 h for 4.2 % of the pairs, for flat days 34.4 %.  Other wrong labels
  they see far less: with the heat-pump flag flipped 89.6 %, with the EV flag
  flipped 84.9 %, with another season 74.5–82.3 %, around the paper's 79.7 %, and
  |Δ std| is below 1 h for 90.6–95.3 % against its 99.0 %.  They do not see the day
  type: training days of the other day type reach 100 % on the time rows and on the
  1 kWh and FID rows.  Meeting the time, 1 kWh and FID levels, which real days of
  other households also meet, is necessary, not sufficient, and does not make
  generated days indistinguishable from real ones: on the test households a
  gradient-boosting classifier that also sees step-to-step changes and sorted values
  reaches a C2ST AUC of 0.697 (approximate 95 % interval 0.685–0.709) against 0.505
  (0.482–0.528) for real days of the validation households, which are held fixed
  (`results/reliability.md`).
* **The GitHub baseline.**  Its minibatch order is an assumption (see
  *`faithful`*), and reshuffling alone raises its accuracy (+11.2 points, `best`
  against `best`, one seed), so `abl_shuffle` (GitHub model with reshuffled
  batches, one seed) is a second baseline.  With `best` the means of the time
  parameters are within 1 h for 51.0 % and 53.1 % of the pairs, their |Δ std| is
  below 1 h for 56.8 % and 55.7 %, and their FID is below 5 for 63.5 % and 65.1 %
  (paper protocol, held-out); with `last` these are 45.3 % and 44.3 %, 72.4 % and
  72.4 %, and 66.7 % and 72.9 %.  Both miss the paper's levels for the time
  parameters.
* **Digitisation.**  Values are read from the colours of each heatmap to about
  one of the 256 colours of its scale (0.012 for differences, 0.004 for the
  KLD, 0.039 for the FID).  11 of the 1,280 cells lie at the end of a colour
  scale; all are |Δ mean| cells of time parameters above 1 h, so no median or
  share changes.

## Evaluation (`metrics.py`, `evaluate.py`, `report.py`)

For every label the generator produces as many days as the test households
have for that label; all metrics are computed per label and averaged over the
64 labels.

**Statistical accuracy** (the target, ≥ 85 %).  The paper validates with the
average day per label (Figs. 3–4), the mean and standard deviation of five key
parameters (Figs. 5–6) and their distributions (KL, FID; Figs. 9–12).  The
accuracy turns these comparisons into bounded scores (1 = perfect), each
against the held-out real days of the label:

| term | definition |
|---|---|
| mean profile | 1 − Σₜ\|μ_gen(t) − μ_real(t)\| / Σₜ\|μ_real(t)\| |
| std profile | 1 − Σₜ\|σ_gen(t) − σ_real(t)\| / Σₜ σ_real(t) |
| KP mean | 1 − \|mean_gen(k) − mean_real(k)\| / mean_real(\|k\|), averaged over the key parameters k |
| KP std | 1 − \|σ_gen(k) − σ_real(k)\| / σ_real(k), averaged over k |
| hourly 1−KS | 1 − Kolmogorov–Smirnov distance of the hourly values, averaged over the 24 hours |
| KP 1−KS | 1 − KS distance of each key parameter, averaged over k |

accuracy = mean(mean profile, std profile, (KP mean + KP std)/2, (hourly 1−KS + KP 1−KS)/2),
each term clipped to [0, 1].  Level, spread, key-parameter statistics and full
distributions weigh equally, so a model that reproduces only the average day
scores below 50 %.  Key parameters (paper Sec. 2.4, after Price 2010):
near-peak load (97.5th percentile of the day), near-base load (2.5th
percentile), peak load duration (hours above the mean of the two), rise time
(from the last near-base hour to the start of the high-load period), fall time
(from its end to the next near-base hour) and, in addition, daily energy.  The
high-load period is the contiguous run of hours above the midpoint that
contains the daily maximum, and a near-base hour lies within 10 % of the daily
range above the near-base load.  (With the first and last high hour of the
whole day, rise and fall time would be 0 on most PV days, whose night and
evening lie above the midpoint set by the midday export.)

**Paper criteria** (`results/paper_criteria.md`).  The paper accepts its model
because the mean and standard-deviation differences of the load parameters are
below 1 kWh, those of the time parameters below 1 h for most classes, the KL
divergences are about 0.3 and below and most FIDs are below 5.  The table gives the share
of (label, key parameter) pairs that meet each level, for both definition sets.
`results/paper_protocol.md` repeats it with generated days against 2,000 random
days per label of the *training* households, to separate the model from the
stricter held-out protocol used everywhere else; the comparison with the
published figures uses all training days, as the reference does.

**Other metrics.**  The relative Fréchet distance of the 24-h profiles (rFD),
the error of the hour-to-hour correlation matrix, a classifier two-sample test
(C2ST: ROC-AUC of a gradient-boosting classifier separating real from
generated days, with cross-validation folds grouped by household so that it
cannot win by recognising households; 0.5 = indistinguishable), precision /
recall / coverage (Naeem et al. 2020), a memorisation check (DCR ratio:
distance of generated days to their nearest training day relative to the same
distance for real held-out days; ≈ 1 expected, ≪ 1 would mean copying) and the
share of negative values for households without PV (physically impossible).

**Real-data benchmarks.**  Two rows score real days of other households
against the test households with the same metrics: days of the 356
*validation* households and days of the 1,657 *training* households (what a
generator that copied its training data would score).  They show how close
real homes of the same population come to the test homes.  They are not upper
bounds: they depend on the size and composition of the pool, and a generator
that represents the population well can exceed them on some metrics.  They are
single deterministic evaluations (no seeds); the DCR ratio of the
training-household benchmark is not meaningful (those households are the
training data).

**Uncertainty** (`uncertainty.py`, `results/uncertainty.md`).  The ± in the
tables is the standard deviation over training seeds.  The larger source of
uncertainty is which households are in the test set, shared by all rows.  It
is estimated by stratified half-sampling: the test households of every group
are split at random into two halves, every model and benchmark is scored on
both halves, and the standard error of the full-sample accuracy is
SD(half A − half B) / 2 over 100 splits (the same splits for all rows, so
differences between rows are paired).  The statistical accuracy depends on how
many real days a label has — fewer days give a lower score — so it is
comparable only between rows evaluated on the same test set, as all rows here
are, and the 85 % target refers to this test split.  All results are
conditional on the 1,657 training households, and the benchmarks on their pool
of households.  The household halves depend only on the replicate number, so
`uncertainty.py --stored results` adds rows to the table without sampling the
stored sets again, as for the classical baselines; it stops if the stored
scores of the real-data benchmarks differ from the recomputed ones.

**Other household splits** (`split_sensitivity.py`,
`results/split_sensitivity.md`).  The half-sampling keeps the training
households fixed and works with the test households of one split.  As a check
that needs no trained network, the split is drawn again with 20 other seeds
(same fractions and stratification), and the two real-data benchmarks and the
Gaussian copula (one sampling seed) are scored on the test households of every
split:

| | split of the study | 20 other splits, mean ± SD | median | range | household SE, split of the study |
|---|---|---|---|---|---|
| real: training households | 87.95 | 88.44 ± 1.18 | 88.66 | 84.63 to 90.06 | 0.78 |
| real: validation households | 86.06 | 86.70 ± 1.10 | 86.72 | 84.43 to 88.45 | 0.98 |
| Gaussian copula | 85.78 | 86.28 ± 1.12 | 86.54 | 82.65 to 87.76 | 0.77 |
| Gaussian copula − training households | −2.17 | −2.16 ± 0.18 | | −2.46 to −1.67 | |
| validation − training households | −1.89 | −1.75 ± 1.34 | | −3.98 to +0.92 | |

The split of the study is an ordinary one, somewhat below the average (4 of
the 20 other splits give a lower training-household benchmark).  The level of
the accuracy varies more between splits than the half-sampling standard error
of one split indicates (SD 1.1–1.2 points against 0.8–1.0), mostly through single
splits far below the others: the distribution has a tail towards low values
(84.6 and 86.7 % for the training-household benchmark of two splits, all others
87.6–90.1 %), as expected when single households carry much of the variance of
a label.  The 85 % target is therefore not a fixed bar, and the intervals of
`uncertainty.md` describe the split of the study.  The difference between the
copula and the training-household benchmark of the same split is nearly
constant (SD 0.18 points), whereas the validation-household benchmark, a pool
of 356 households, moves by 1.3 points against it.

**Trained models on other splits** (`jobs/splits.txt`,
`results/split_sensitivity.md`).  A trained model has to be trained again for
every split.  `improved` was trained on the first eight of the 20 splits
(`train.py --set split_seed=K`, seed 0, on a CPU with the environment of the
section on other hardware).  Every run sees the training households of its
split, its checkpoint is selected on the validation households of that split,
and it is scored on the test households of that split, next to the benchmark
and the copula of the same split:

| split | `improved` | training households | Gaussian copula | `improved` − training households | `improved` − copula | C2ST AUC, `improved` / copula | coverage, `improved` / copula |
|---|---|---|---|---|---|---|---|
| of the study | 87.67 | 87.95 | 85.78 | −0.28 | +1.89 | 0.646 / 0.804 | 0.792 / 0.618 |
| 0 | 88.92 | 88.81 | 86.35 | +0.11 | +2.57 | 0.639 / 0.806 | 0.806 / 0.646 |
| 1 | 87.69 | 88.11 | 86.44 | −0.42 | +1.25 | 0.676 / 0.804 | 0.794 / 0.647 |
| 2 | 87.70 | 88.75 | 86.56 | −1.05 | +1.14 | 0.638 / 0.800 | 0.795 / 0.629 |
| 3 | 87.50 | 88.80 | 86.65 | −1.31 | +0.85 | 0.649 / 0.796 | 0.787 / 0.637 |
| 4 | 87.01 | 86.67 | 84.68 | +0.34 | +2.33 | 0.668 / 0.809 | 0.788 / 0.623 |
| 5 | 88.87 | 89.81 | 87.48 | −0.94 | +1.40 | 0.660 / 0.801 | 0.801 / 0.642 |
| 6 | 87.43 | 88.99 | 86.79 | −1.56 | +0.64 | 0.677 / 0.802 | 0.794 / 0.640 |
| 7 | 85.25 | 84.63 | 82.65 | +0.62 | +2.60 | 0.658 / 0.809 | 0.792 / 0.635 |
| mean ± SD, 8 other splits | 87.55 ± 1.15 | 88.07 ± 1.66 | 85.95 ± 1.55 | −0.53 ± 0.81 | +1.60 ± 0.79 | 0.658 ± 0.015 / 0.803 ± 0.004 | 0.795 ± 0.006 / 0.638 ± 0.008 |

The row of the split of the study is the CPU run of the section on other
hardware (the five GPU seeds give 87.5 ± 0.2 %).  `improved` is above the
copula in each of the eight splits.  In every split a classifier separates its
days from real days less well than copula days (AUC 0.64–0.68 against
0.80–0.81), its coverage is higher (0.79–0.81 against 0.62–0.65) and its DCR
ratio stays near 1 (0.99–1.07).  The split with the lowest benchmark (seed 7,
84.6 %) is also the one with the lowest accuracy of `improved` (85.3 %).

The distance to the training-household benchmark is not constant: −0.5 ± 0.8
points, from −1.6 to +0.6, against −2.1 ± 0.2 for the copula in the same eight
splits.  The copula takes the distribution of every hour from the training
households, so its mean-profile and standard-deviation terms follow those of
the benchmark (differences of −0.1 ± 0.2 and −0.2 ± 0.3 points over the 20
splits) and its distance comes from the key-parameter terms.  A trained model
has to learn these distributions.  Over the eight splits the term of the hourly
standard deviation of `improved` lies between 2.3 points below and 3.1 points
above that of the benchmark (SD 1.9 points, of which a quarter enters the
accuracy), while the differences in the other terms have an SD of 0.1 to 0.9
points.  `improved` is above the benchmark where the benchmark is low (splits
4 and 7, in which the hourly standard deviation of the training households
scores 76.2 and 72.6 %) and below it where the benchmark is high (correlation
−0.77 between the benchmark and the difference), so its accuracy varies less
between splits (SD 1.2 points) than that of the benchmark (1.7).

Two consequences follow.  A difference of a few tenths of a point between a
trained model and the benchmark, like the −0.1 point of the final model in the
split of the study, belongs to that split and can change sign in another one.
And the difference between `improved` and the copula varies over the splits
with an SD of 0.8 points, about twice the half-sampling standard error of the
difference between the final model and the copula in the split of the study
(0.34).  The half-sampling keeps the training households fixed, and single
runs add the spread of the training seed and of the sampling seed of the
copula (0.2 points each).  The intervals of `uncertainty.md` describe the test
households of the split of the study; for another draw of the training
households they are too narrow.

This check has one run per split.  The configurations were chosen on the
validation households of the split of the study, which are spread over the
training, validation and test households of the other splits, so only the
weights and the checkpoint of a run are free of its test households.

**Reliability check** (`reliability.py`, `results/reliability.md`).  Five
headline metrics with approximate 95 % intervals over the test households (normal
approximation, half-sampling as above, 10 replicates, seed spread included): a
stronger C2ST on days rounded to the meter resolution of 0.001 kWh that also sees
the step-to-step changes, the sorted values and the sorted absolute changes, the
statistical accuracy, the KS distance of the key parameters, coverage and the DCR
ratio.  It samples on the CPU and uses at most 1,000 days per side and label for
the classifier, coverage and DCR (2,000 in `report.py`) and at most 5,000 training
days per label for the DCR (all in `report.py`), so its coverage is up to 0.03
higher than in `report.py`, also for the real-data rows.  Accuracy and DCR ratio
differ only slightly, except the DCR ratio of the training-household benchmark,
which is not meaningful in either table.

**Comparison with the GitHub model** (`results/comparison.md`).  Every metric
has a direction (higher, lower, or closer to its ideal value: 0.5 for the C2ST
AUC, 1 for the DCR ratio); the table marks a metric as better for the final
model when every final seed beats every value of the GitHub model (all seeds,
both checkpoints).  The reference code's "KLD" is shown for comparison with the
published figures only and is not used to rank models.

### Protocol notes

* One development run — the first `improved` run, before the no-PV floor
  existed (`runs/dev_improved_nofloor`) — was scored on the test households
  during development (accuracy 87.2 %, 21 October 2026).  Its test numbers are
  not reported.  The floor added afterwards rests on the physical argument and
  on the validation households, where the same run produced 0.40 % negative
  hourly values for labels without PV.  The three other selection candidates
  (`improved_diff`, `improved_big`, `improved_bigdiff`) were defined after
  this test scoring; the comparison among all four used validation households
  only, with the selection rule fixed beforehand.  Every run in the summary,
  comparison and ablation tables was trained after these decisions; the
  selection table lists the development run itself, scored on validation
  households.
* The metric definitions were finalised after two code reviews and before the
  reported runs: C2ST folds grouped by household, KL bins at the quantiles of
  the real sample, and the high-load period of the rise and fall times.
* Training runs that were interrupted before checkpoints with optimiser state
  existed were restarted from scratch; since then every run writes
  `resume.pt` and continues after an interruption (`run_jobs.py`).

## Re-training on other hardware (`replication.py`)

All runs of the result tables were trained on one CUDA device.  To see how
much the results depend on the hardware and the package versions, the final
model and `improved` (seed 0 each) were trained again on a CPU with the
environment of `requirements-lock.txt` (`jobs/replication.txt`, one thread;
every run writes its environment to `environment.json`).  A seed fixes a run on
one type of device (on the CPU a run repeats exactly, also after an
interruption; `tests/test_train.py`), but the CPU and CUDA generators draw
different random numbers, so a re-trained run is a new, independent run of its
configuration and is compared with the spread of the published seeds
(`results/replication.md`):

| | test accuracy | worst label | FID (5 KP) | rFD | C2ST AUC | coverage | DCR ratio |
|---|---|---|---|---|---|---|---|
| `final` [best], 5 published seeds | 87.9 ± 0.2 | 74.6 ± 0.6 | 0.85 ± 0.03 | 0.117 ± 0.005 | 0.609 ± 0.006 | 0.808 ± 0.002 | 1.01 ± 0.00 |
| `final` [best], re-trained (seed 0) | 88.0 | 75.3 | 0.82 | 0.116 | 0.615 | 0.807 | 1.00 |
| `improved` [best], 5 published seeds | 87.5 ± 0.2 | 76.0 ± 1.5 | 1.03 ± 0.08 | 0.117 ± 0.005 | 0.645 ± 0.011 | 0.791 ± 0.004 | 1.03 ± 0.01 |
| `improved` [best], re-trained (seed 0) | 87.7 | 77.2 | 1.08 | 0.112 | 0.646 | 0.792 | 1.03 |

In accuracy both re-trained runs lie within the seed-to-seed spread of the
published runs: the final model is 0.1 point above their mean (0.5 of their
standard deviations; the five seeds range from 87.6 to 88.1 %) and `improved`
0.2 point (0.9).  No metric of these checkpoints is worse than the published
mean by more than 1.2 standard deviations.  In the other direction the largest
distances are those of the mean profile (87.1 % against 86.7 ± 0.2 % for the
final model, 87.0 % against 86.5 ± 0.2 % for `improved`) and of the hourly
distributions of the final model (93.58 % against 93.47 ± 0.05 %), 1.9 to 2.5
standard deviations.  The last iterate of the re-trained final model scores
88.1 %, above its five published last iterates (87.6 ± 0.2 %, at most 87.9 %).
A standard deviation from five seeds is itself uncertain by about a third, and
one re-trained seed per model cannot separate the spread of independent runs
from a small effect of the hardware.  The published values of `improved` are
read from the rounded cells of `results/ablation.csv`.

## Reproducing

```bash
python -m venv .venv
source .venv/bin/activate                             # Linux/macOS; Windows: .venv\Scripts\activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
mkdir -p data/raw
curl -L -o data/raw/raw.zip https://github.com/MahmoudAsadi97/Cgan_Research_2026/releases/download/data/raw.zip
python data_prep.py                                   # -> data/processed/fluvius_hourly.npz (~1 min)
python data_check.py                                  # checksum, cleaning, split and label counts of the local data
python -m pytest

python run_jobs.py jobs/variants.txt --parallel 3     # model-selection candidates and faithful seed 0
python run_jobs.py jobs/final.txt --parallel 3        # final model, seeds 0-4
python run_jobs.py jobs/ablation.txt --parallel 3     # faithful seeds 1-2 and the ablation ladder
python evaluate.py --real train --split val           # real benchmark on the validation households (curves, dashboard)
python report.py --models faithful final --figures faithful:last final:best \
    --ablation faithful abl_shuffle abl_quantile abl_wgan_refnets abl_wide abl_ema improved improved_big final \
    --selection dev_improved_nofloor improved_diff improved_big improved_bigdiff final \
    --paper-protocol faithful:last final:best --curves faithful final
python baselines.py                                   # classical baselines (CPU; compares with all_runs.csv of report.py)
python uncertainty.py --reps 100 --baselines copula gmm   # household-level uncertainty (needs report.py and baselines.py first)
python split_sensitivity.py                           # real-data benchmarks and copula on 20 other household splits (CPU)
python reliability.py --models final:best faithful:last --reps 10 --workers 6   # five metrics with household intervals
python compare_paper.py --pdf <paper.pdf> --reference-gen "<cGAN_EB>/cGAN Generation"   # needs the PDF, poppler's pdfimages and a clone of the reference
python compare_paper.py --pdf <paper.pdf> --reference-gen "<cGAN_EB>/cGAN Generation" --ranks 21 3 --out results/figure_ranks
python dashboard.py                                   # live training monitor, http://127.0.0.1:8050

python run_jobs.py jobs/replication.txt --runs runs_cpu   # improved and final (seed 0) again, e.g. on other hardware
python replication.py --runs runs_cpu                 # the re-trained runs against the published seeds
python run_jobs.py jobs/splits.txt --runs runs_splits     # improved and final trained on other household splits
python split_sensitivity.py --trained runs_splits/* runs_cpu/improved runs_cpu/final   # ... next to the benchmarks of their splits
```

`run_jobs.py` starts every job with `train.py --resume`: running a job file
again after an interruption continues all unfinished runs from their last
checkpoint and skips finished ones.  Single runs:
`python train.py --config final --seed 0`, `python evaluate.py --run runs/final/seed0 --ckpt best`.

## Files

```
data_prep.py   raw.zip -> hourly arrays; labels, cleaning, household split
data_check.py  checks of a local copy of the data; the split of every meter
cGAN.py        reference Generator / Discriminator / cGAN; cWGAN (improved and final models)
train.py       configurations, value transforms, validation, checkpoints and resuming
baselines.py   classical baseline generators, fitted, selected and scored like the models
run_jobs.py    runs a job file a few jobs at a time, resuming interrupted runs
metrics.py     key parameters (paper text and reference code), accuracy, KL / FID, other metrics
evaluate.py    evaluation of one run (or of the real-data benchmarks)
report.py      result tables and figures for all finished runs
uncertainty.py household-level standard errors and paired differences (half-sampling)
split_sensitivity.py real-data benchmarks, Gaussian copula and trained models on other random household splits
reliability.py five headline metrics with confidence intervals over households
replication.py re-trained runs against the published seeds; additional seeds of the ablation
compare_paper.py comparison with the published figures of the paper, all seeds
utils.py       reference result analysis: per-label CSVs and the paper's Figs. 3-12
dashboard.py   live training monitor
jobs/          job files of all experiments
results/       tables and figures (generated days of the final model in results/figures/final_best)
tests/         unit tests
requirements-lock.txt  exact package versions of a tested CPU environment
LICENSE-cGAN_EB  license of the reference code adapted in cGAN.py and utils.py
```

## References

* Fluvius. Verbruiksprofielen digitale elektriciteitsmeters: kwartierwaarden voor een volledig jaar (sample of 2024). Fluvius Open Data. https://opendata.fluvius.be/explore/dataset/1_50-verbruiksprofielen-dm-elek-kwartierwaarden-voor-een-volledig-jaar/
* Li, Y., Dong, B., Qiu, Y. (2025). Conditional generative adversarial network (cGAN) for generating building load profiles with photovoltaics and electric vehicles. *Energy and Buildings* 335, 115584. https://doi.org/10.1016/j.enbuild.2025.115584
* Price, P. N. (2010). Methods for analyzing electric load shape and its variability. LBNL-3713E.
* Wang, Z., Hong, T. (2020). Generating realistic building electrical load profiles through the Generative Adversarial Network (GAN). *Energy and Buildings* 224, 110299.
* Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., Courville, A. (2017). Improved training of Wasserstein GANs. NeurIPS.
* Yazıcı, Y. et al. (2019). The unusual effectiveness of averaging in GAN training. ICLR.
* Naeem, M. F., Oh, S. J., Uh, Y., Choi, Y., Yoo, J. (2020). Reliable fidelity and diversity metrics for generative models. ICML.
