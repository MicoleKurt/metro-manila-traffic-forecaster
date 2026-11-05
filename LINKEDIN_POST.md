🚦 I mapped 17,000 real MMDA traffic incidents and trained a model to forecast them.

Using real MMDA incident alerts (2018–2020), hourly ERA5 rainfall and official PH holidays, I built a Metro Manila Traffic Incident Forecaster.

What the data showed:
📍 EDSA is the hotspot. The incident map traces EDSA, C5 and Commonwealth almost exactly.
🕗 Incidents peak during the morning and evening rush hours.
🦠 The 2020 ECQ cut reported incidents by about 95%, from ~28 a day to ~1.
🌧️ Light rain increases incidents per hour, but heavy rain doesn't. Fewer cars on the road during downpours likely offsets the risk.
🚗 About 70% of alerts are vehicular accidents.

📊 The model is a Poisson gradient-boosting model with a time-based train/test split. On 4 months of unseen data it beat the historical-average baseline by 8% on daily incident counts.

Every number comes from public data: MMDA, Open-Meteo/ERA5 and the official holiday proclamations.

🔗 https://github.com/MicoleKurt/metro-manila-traffic-forecaster

#MachineLearning #DataScience #Python #Philippines #MetroManila #EDSA #MMDA #Traffic #OpenData
