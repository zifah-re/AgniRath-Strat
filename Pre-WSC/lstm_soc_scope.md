# LSTM for live SOC / voltage-curve estimation: literature and proposed scope (future work, WSC 2027)

## Papers (from web search; abstracts/first pages read, not full texts)
- Yang, Zhang, Li, Miao, "State-of-charge estimation of lithium-ion batteries using LSTM and UKF" (2020): LSTM-RNN from voltage, current, temperature; targets LFP's flat OCV-SOC curve and temperature.
- Wong et al., "Li-Ion Batteries State-of-Charge Estimation Using Deep LSTM at Various Battery Specifications and Discharge Cycles", GoodIT 2021.
- Shu et al., "State of charge estimation of lithium-ion battery packs based on improved cubature Kalman filter with LSTM" (pack level).
- "Accurate State of Charge Estimation for Real-World Battery Systems Using a Novel Grid Search and Cross Validated Optimised LSTM" (Energies 2022, 15(24), 9654): real-vehicle operating data.
- "Combined SOC Estimation Method Using LSTM and UKF" (Energies 2025, 18(5), 1106); "SOC Estimation: Online Method Combining DNN and Adaptive Kalman Filter" (Processes 2025, 13(11), 3559).
- Real driving cycle SOC prediction with deep learning (Applied Sciences, Nov 2021).

## Proposed scope for our car
Inputs we already log: Pack_Voltage, Pack_Current / Bus_Voltage x Bus_Current, SOC_Ah, speed. Pack_Current is logged as 0 in the cleaned files, so bus power and SOC_Ah are the usable signals.
Target: SOC (or remaining Wh) as the output, trained on the coulomb-counted SOC_Ah of the 8 race days + charging files.
Plan: (1) baseline = voltage-OCV lookup + coulomb counting; (2) small LSTM (1-2 layers, window of the last 5-10 min of V, I) predicting SOC; (3) optional LSTM+Kalman correction as in the papers above; (4) evaluate leave-one-day-out. Needs a ground-truth SOC reference (full charge / rest voltages), which is the open item.
