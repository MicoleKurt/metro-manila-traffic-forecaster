🚦 This week, Metro Manila traffic ran 41% worse than usual. I built a model that sees it coming.

My Metro Manila Traffic Forecaster runs on live, real data:
📡 TomTom's hourly congestion feed, collected automatically every 6 hours
📊 TomTom's 2025 speed profile for every hour of the week
🚨 17,000 real MMDA incident alerts
🌧️ Live and forecast rainfall (Open-Meteo) and official PH holidays

What the data shows:
🔴 Weekday evening peaks this week hit 175% congestion, so trips took 2.75× as long as on an empty road
🕯️ The Undas long weekend (Nov 1–2) was the only time traffic dropped below normal
🐢 In 2025, rush-hour speeds across the metro averaged about 16 km/h
📍 EDSA, C5 and Commonwealth are the incident hotspots

📈 The 48-hour congestion forecast missed by 15.9 points on average over the last two days. TomTom's own "usual for this hour" estimate missed by 39.5.

Built with Python, scikit-learn, Streamlit and GitHub Actions. The dataset updates itself.

🔗 https://github.com/MicoleKurt/metro-manila-traffic-forecaster

#MachineLearning #DataScience #Python #Philippines #MetroManila #EDSA #Traffic #OpenData #Streamlit
