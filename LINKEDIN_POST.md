🚦 Can machine learning predict EDSA traffic?

I built a Metro Manila Traffic Forecaster that predicts hourly travel speed on 7 major corridors: EDSA, C5, Commonwealth, España, Roxas Blvd, SLEX and NLEX.

What the model learned (no surprise to any Filipino commuter 😅):
🌧️ A typhoon day can cut EDSA rush-hour speed by more than half
🎄 The "ber months" make the evening rush slower every year, and December is the worst
💸 Payday Fridays are their own kind of chaos
🕕 6–7 PM on a Friday is the slowest hour of the week

📊 Results: a gradient-boosting model cut prediction error by 16% compared with a historical-average baseline (MAE 2.23 km/h, R² 0.97) on 6 months of unseen data.

The data is calibrated to real figures: EDSA averages ~21 km/h (MMDA), rush hour in Manila averages 15.2 km/h (TomTom Traffic Index 2025), and EDSA carries 407k+ vehicles a day.

The features are built for the Philippines: Holy Week, payday (the 15th and 30th), MMDA number coding, PAGASA rainy season and typhoons.

Note: the data is synthetic and calibrated to public MMDA, TomTom and PAGASA patterns. A loader for real data is included, and connecting live data is the next step.

🔗 Code + charts: [GitHub link]

#MachineLearning #DataScience #Python #Philippines #MetroManila #EDSA #Traffic #Portfolio
