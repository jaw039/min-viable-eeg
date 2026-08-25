# Conclusion (draft — Week 2)

Owner: Jackie. Structure follows the four research questions. Numbers in
`[brackets]` are placeholders to be filled from the final saved results;
numbers stated plainly are already final (channel-ranking and stability
analyses, commits 34b7ba1 and d2d705e). Do not add claims that are not in
the Results section.

---

This study asked how far the electrode count of a motor-imagery
brain–computer interface can fall before decoding breaks down, and which
electrodes carry the signal. Using the PhysioNet EEG Motor Movement/Imagery
dataset (106 subjects, 64 channels, left- versus right-fist imagery), we
trained a full-montage EEGNet reference, ranked electrodes by a
training-split-only Fisher score of 8–30 Hz band power, and re-trained the
same architecture under channel budgets of 4, 6, 8, 12, 16, and 32
electrodes. Every reduced-channel model used identical subject-wise splits,
preprocessing, and evaluation, so differences in Cohen's κ are attributable
to the electrode set alone.

**Minimum viable montage.** Defining the minimum viable montage in advance
as the smallest budget retaining at least 90% of full-montage κ, we found
that threshold at [k* = __] electrodes, which retained [__%] of the
64-channel κ of [__]. [One sentence on the shape of the curve: where it
plateaus, where it collapses, e.g. "Performance was flat from 64 down to
[__] channels and fell sharply below [__]."] This places the practical
floor for this task [within / above] the 4–8 dry-electrode range of current
consumer headsets.

**Which electrodes.** The selected montages were dominated by the right
sensorimotor region: C4 led the ranking, followed by CP4, C6, CP6, and FC4,
with the canonical left-hemisphere electrode C3 not entering until rank 23.
A bootstrap over training subjects showed that C4 is the only electrode
whose place in the small-budget montages is robust (selected in 85% of
resamples at k=4 and 93% at k=8); its right-centroparietal neighbours were
selected roughly half the time and were interchangeable with frontal and
occipital alternatives. The recommended 4-electrode set should therefore be
read as "C4 plus three supporting channels" rather than as a uniquely
determined layout.

**Selected versus random electrodes.** [Fill from Younes' analysis.]
Selected montages [outperformed / did not reliably outperform] random
subsets of the same size: at k=4 the selected set reached κ = [__] against a
random-subset mean of [__] (SD [__]), and at k=8, [__] against [__]. [One
sentence on where the advantage was largest and whether it vanished at
larger budgets.] Given the bootstrap result above, [most / part] of the
small-budget advantage is plausibly carried by C4 alone.

**Knowledge distillation.** [Fill from Colin's analysis.] Distilling from
the full-montage teacher [improved / did not consistently improve]
low-channel students: at 4, 6, and 8 channels the distilled models scored
κ = [__ / __ / __] versus [__ / __ / __] from scratch, [with / without] a
consistent direction across seeds.

**Shared versus personalised montages.** Ranking each training subject from
their own data alone gave montages that overlapped the shared top-4 in only
12% of slots, with 52 of 74 subjects sharing no electrode at all; occipital
(PO7, O1, PO3) and frontal-temporal (F7, FT7, AF7) sites led the individual
selections. With roughly 45 trials per subject these personal rankings are
noisy, so the disagreement mixes genuine heterogeneity with estimation
error, but it indicates that a single shared low-density layout is unlikely
to be optimal for every user and that the case for per-subject electrode
placement deserves direct testing.

Taken together, the results give a first systematic electrode-budget curve
for two-class motor-imagery decoding at population scale, a concrete
recommended electrode set, and an open, deterministic evaluation harness
(fixed splits, frozen channel sets, and provenance-stamped outputs) with
which the same curve can be produced for other paradigms, datasets, and
architectures. Two qualifications bound these conclusions: the evaluation
is offline, so the κ values are an upper bound on what a user would achieve
in closed-loop control, and the recurrence of non-motor electrodes in the
rankings means part of the discriminative signal may reflect lateralised
gaze or attention rather than motor cortex, a possibility that a
real-headset, gaze-controlled replication would resolve.

---

## Notes for revision (delete before submission)

- Placeholders depend on: Jahari's full-montage κ and budget sweep (k*),
  Younes' selected-vs-random statistics, Colin's distillation table.
- If k* comes out ≤ 8, strengthen the consumer-hardware sentence; if it is
  > 16, reframe the contribution as "characterising the floor" rather than
  "reaching the consumer range".
- The stability and per-subject numbers are from `stability.json`
  (commit d2d705e); the ranking from `channel_ranking.json` (5e0fbda).
- Keep consistent with Discussion (Colin) on the gaze/attention caveat —
  state it once here, argue it there.
- Candidate titles: "Minimum Viable EEG: How Few Electrodes Does Motor
  Imagery Decoding Need?"; "An Electrode-Budget Curve for Motor-Imagery
  BCI".
