# LSTM for live SOC / voltage-curve estimation: literature and proposed scope (future work, WSC 2027)

Goal: estimate the battery state of charge (and the voltage-vs-SOC curve) live from what the car logs,
so the strategist sees remaining energy during a stage instead of relying on the planned SOC trajectory.

## Papers (from web search; abstracts / first pages read, not full texts)
- Yang, Zhang, Li, Miao, "State-of-charge estimation of lithium-ion batteries using LSTM and UKF" (2020):
  LSTM-RNN estimating SOC from voltage, current and temperature under varying temperature; targets LFP's
  flat OCV-SOC curve. An unscented Kalman filter smooths the LSTM output.
- Wong et al., "Li-Ion Batteries State-of-Charge Estimation Using Deep LSTM at Various Battery
  Specifications and Discharge Cycles", GoodIT 2021, doi:10.1145/3462203.3475878.
- Shu et al., "State of charge estimation of lithium-ion battery packs based on improved cubature Kalman
  filter with LSTM model" (pack level, not cell level).
- "Accurate State of Charge Estimation for Real-World Battery Systems Using a Novel Grid Search and Cross
  Validated Optimised LSTM Neural Network", Energies 2022, 15(24), 9654: trained/validated on operating
  data from three real electric vehicles (closest to our situation: field data, not lab cycles).
- "Combined SOC Estimation Method Using LSTM Network and Unscented Kalman Filter", Energies 2025, 18(5), 1106.
- "SOC Estimation for Lithium-Ion Batteries: An Online Method Combining Deep Neural Network and Adaptive
  Kalman Filter", Processes 2025, 13(11), 3559.

## What our telemetry can and cannot provide (checked on the Sasol logs)
- Usable: Pack_Voltage (logged in mV; Days 2-8, all zero on Day 1), Bus_Voltage x Bus_Current (motor
  controller draw), MPPT Output_Voltage/Output_Current A-D (charge into the pack), timestamps.
- Not usable: SOC_Ah (stuck at a constant, e.g. -0.142, all race), Pack_Current (logged as 0).
- Pack configuration changed during the race (5 -> 4 -> 3 -> 3.5 modules), visible as the pack-voltage
  level dropping from ~86 V (Day 2) to ~65 V (Day 7). A model must be told the module count.

## Proposed scope
1. Ground truth: coulomb counting, net pack current = sum(MPPT output current) - Bus_Current - aux,
   re-anchored at full charge / rest voltages each night. (This is the "software coulomb counting" item.)
2. Baseline: rest-voltage OCV lookup + coulomb counting.
3. Small LSTM (1-2 layers) on the last 5-10 min of pack voltage, net current, and module count,
   predicting SOC; optionally an LSTM + Kalman filter as in the papers above.
4. Evaluation: leave-one-day-out on the Sasol days.
Open item: a reliable SOC reference. Without a working current sensor on the pack the ground truth is
itself a coulomb-count estimate, so WSC 2027 needs Pack_Current / SOC_Ah fixed in the logger.
