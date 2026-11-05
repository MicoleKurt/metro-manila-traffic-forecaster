🚦 Metro Manila traffic this week ran 41% worse than usual. I built a model that forecasts it 48 hours ahead.

My Metro Manila Traffic Forecaster runs on real, live 2026 data:
📡 TomTom's hourly congestion data for Metro Manila and Davao
🌧️ Live and forecast rainfall (Open-Meteo)
📅 2026 PH holidays

What the data shows (Oct 30 – Nov 6, 2026):
🔴 6 PM on a weekday averaged 164% congestion, so trips took about 2.6× as long as on an empty road
🚗 On Nov 5 at 5 PM there were 348 jams at once, totalling 284 km
🕯️ The Undas long weekend was the only time traffic dropped below normal
🏙️ Metro Manila averaged 69% congestion vs 56% for Davao City

📈 The forecast missed by 11.9 points on average over the last 48 hours. TomTom's own "usual traffic" estimate missed by 39.5, so that's 70% less error.

Built with Python, scikit-learn and Streamlit. It's fully self-contained, with no API keys required.

🔗 https://github.com/MicoleKurt/metro-manila-traffic-forecaster

#MachineLearning #DataScience #Python #Philippines #MetroManila #EDSA #Traffic #Streamlit #OpenData
