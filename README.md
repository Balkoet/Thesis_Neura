<div align="center">
  <h1>Diplomatic Thesis</h1>
  <p><em>Estimating Hourly Mean Active Power on High-Voltage Substations using Deep Learning</em></p>
</div>
<div align="center">
  <h2>Abstract</h1>
</div>
This thesis investigates artificial neural networks for estimating hourly mean active power on the University of Patras campus’s high voltage sub-stations. It uses 9 aggregated time series covering 24 electrical measurement points from October 2024 to September 2025. Electrical measurements were collected through the University monitoring system, under the written permission from the Rector, Professor Christos Bouras, and combined with calendar information and temperature, wind and rainfall observations from the University weather station, which were collected with personal initiative.
To account for differences in power levels, the series are trained jointly within three load groups, allowing each model to learn from installations of comparable scale. After data cleaning and temporal alignment, three architectures are examined: a fully connected feed-forward network (FFNN), a one-dimensional convolutional network (CNN) and a hybrid network (CNN-LSTM Hybrid) combining convolution with temporal memory. Each experiment is repeated ten times and the estimates are averaged to reduce sensitivity to an individual training run. A self-organizing map complements the analysis by visualizing hourly operating conditions.
Evaluation distinguishes estimation of a month excluded from feed-forward training from retrospective reconstruction of a month included in convolutional-model training. Under the shared reconstruction protocol, the hybrid network achieves pooled mean absolute errors of 8.52–8.59 kW, compared with 11.65–12.72 kW for the one-dimensional convolutional network across four validation fractions. These metrics are computed over valid hourly pairs throughout the twelve-month period. Analysis by series, period and training run highlights difficulties at low power levels and during peaks. The findings describe the recorded experiments and do not equate retrospective reconstruction with forecasting an unseen future month.
The experiments conducted demonstrated that the problem can be effectively approached using artificial neural networks, yielding a high degree of accuracy in the results. Optimal performance was achieved by employing a hybrid Convolutional Neural Network and Long Short-Term Memory (CNN-LSTM) architecture. The substations with the highest consumption exhibited the highest accuracy rates, as well as patterns resembling a sine wave.

**Keywords:** `electrical load forecasting` | `active power` | `artificial neural networks` | `deep learning` | `substations` | `self-organizing maps`

<p align="center">
 <img src="Excel%20Sheets%20&%20Graphs/sin_ys1.png" alt="Load Graph" width="600"/>
</p>
<p align="center">
 <img src="Excel%20Sheets%20&%20Graphs/sin_ys1_draw.png" alt="Load Graph" width="600"/>
</p>
