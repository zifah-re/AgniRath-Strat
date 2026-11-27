# LSTM state-of-charge estimation (replacement for coulomb counting)

Run (from this folder): `pip install jax pandas scipy matplotlib`, then
`python dataset.py` (builds `data/day*.csv` from the Sasol logs, ~1 min) and
`python soc_lstm.py` (reference fit, leave-one-day-out training of 7 folds, figures, ~6 min on CPU).
JAX is used because it installs CPU-only everywhere (Windows included); the LSTM cell is written out
explicitly (`lstm_forward`) and trained with Adam.

## Why
Coulomb counting integrates current: it needs a known starting SOC and it drifts with any current-sensor
offset or capacity error. LSTM networks can estimate SOC directly from recent voltage/current/temperature
history without an initial SOC or a battery model (Chemali et al., IEEE TIE 65(8), 2018). On the 2026 car
the logger's `SOC_Ah` was stuck and `Pack_Current` read 0, so the strategist had no live SOC at all.

## Data (what the logs actually contain)
- Cell voltages from the BMS (CMU1-CMU5, 6 cells each, mV); corrupt channels (outside 2.5-4.35 V) dropped.
  One CMU per module; modules are in series (CMU5 reads 0 on 4-module days, CMU4-5 on 3-module days).
- Motor-controller bus current; array charge current = MPPT power / bus voltage. Net pack current =
  array current - bus current (auxiliary electronics not logged).
- 10 s bins, Days 1-8 (Day 4 is trailered, charging only: training data only).

## Reference SOC ("ground truth")
No measured SOC exists, so the reference is a fitted equivalent-circuit + coulomb-counting model:
`v_cell = OCV(SOC) + R i_net`, `SOC = SOC0_segment + 100 cumAh / Q`, typical NMC OCV curve, one SOC0 per
continuous log segment (logger gaps lose the current), Q and R fitted over all days.
Result: **Q = 29.3 Ah, R = 1.7 mOhm per series cell group, 41 mV RMS voltage fit** (results/reference_fit.csv).
Q is consistent with the 588 Wh module rating (588 Wh / 6 cells / 3.6 V = 27 Ah).

## Results (leave-one-day-out, SOC percentage points, results/summary.csv)
| estimator | RMSE | mean abs | needs start SOC | drifts with sensor offset |
|---|---|---|---|---|
| Voltage lookup, OCV^-1(v - R i) | 3.7 | 2.7 | no | no |
| Coulomb counting, live start | 4.3 | 3.5 | yes | yes |
| Coulomb counting, 0.5 A / 2 % sensor error | 4.3 | 3.6 | yes | yes |
| **LSTM** (10 min window: v_cell, v - R i, i_net, i_bus, T) | 5.1 | 4.2 | **no** | **no** (5.14 -> 5.19 with corrupted current) |
| Hybrid: lookup + LSTM correction | 5.5 | 4.2 | no | no |

Drift (results/coulomb_drift.csv, fig_soc_drift.png): coulomb counting starts 2.8 pts off when started from a
loaded voltage and drifts 1.7 pts/h per 0.5 A offset, i.e. ~18 pts after a 9 h race day. The LSTM's error does
not grow with time.

## Honest conclusion
- On six training days the LSTM does **not yet beat** a physics voltage lookup; the reference itself is built from
  coulomb counting and the same OCV curve, which favours those two methods structurally.
- What the LSTM does show: no initial SOC needed, no drift, insensitive to current-sensor bias; worst where
  training data is thin (below ~20 % SOC; Day 7 went to 2.85 V/cell, outside the OCV table).
- Recommended for WSC 2027: fix `Pack_Current`/`SOC_Ah` logging (true reference), log a full charge and a rest
  period every night (OCV anchors), measure the cell OCV curve, then retrain on more days. Use the voltage lookup
  + coulomb counting with nightly re-anchoring as the baseline the LSTM must beat.

## References
- E. Chemali, P. J. Kollmeyer, M. Preindl, R. Ahmed, A. Emadi, "Long Short-Term Memory Networks for Accurate
  State-of-Charge Estimation of Li-ion Batteries," IEEE Trans. Ind. Electron., 65(8), 6730-6739, 2018,
  doi:10.1109/TIE.2017.2787586.
- F. Yang, S. Zhang, W. Li, Q. Miao, "State-of-charge estimation of lithium-ion batteries using LSTM and UKF,"
  Energy, 2020.
- K. L. Wong et al., "Li-Ion Batteries State-of-Charge Estimation Using Deep LSTM at Various Battery
  Specifications and Discharge Cycles," GoodIT 2021, doi:10.1145/3462203.3475878.
- "Advancements in the estimation of the state of charge of lithium-ion battery: a comprehensive review of
  traditional and deep learning approaches," J. Mater. Inf., 2025 (OAE, jmi.2024.84).
- Further LSTM/Kalman hybrids and real-vehicle LSTM work: see ../../lstm_soc_scope.md.
