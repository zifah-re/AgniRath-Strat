# Agnirath Strategy Dashboard 🏎️📊

Welcome to the **Agnirath Strategy Dashboard** repository. This interactive dashboard is designed for real-time race telemetry analysis, strategy simulation, and predictive analytics for the Agnirath solar/electric racing team. It helps strategists make data-driven decisions regarding energy management, speed profiles, and pit-stop optimization during competitive tracks.

---

## 🚀 Key Features

- **Real-Time Telemetry Visualization:** Live streaming of battery state-of-charge (SoC), cell temperatures, motor RPM, and solar array power generation.
- **Predictive Energy Modeling:** Advanced ML models to forecast energy consumption based on elevation profiles, weather conditions, and track friction.
- **Weather Integration:** Dynamic fetching of solar irradiance data mapped directly onto track coordinates.
- **Historical Analysis:** Post-race review modules to overlay actual run profiles against theoretical optima for future continuous improvement.

---

## 🛠️ Tech Stack

- **Frontend/Dashboard:** Built with static web assets running on a SvelteKit/Vite production build pipeline.
- **Data Visualization:** Interactive tracking and map interfaces (map.html, Google_Earth.py).
- **Backend/Analytics:** Python-based simulation engine (simulator.py, real_sim.py) and live data telemetry handling (downlink.py).

---

## 📦 Installation & Setup

### Prerequisites
Make sure you have Python 3.10+ and `pip` installed.

### 1. Clone the Repository
```bash
git clone https://github.com/zifah-re/AgniRath-Strat.git
cd AgniRath-Strat/Dashboard
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Dashboard Local Server
```bash
python main.py
```

---

## 🗂️ Project Structure

```text
agnirath-strategy-dashboard/
├── 2024 Sasol Solar Challenge Route (Publish).kml
├── Track_testing_chennai.kml
├── Track_testing_chennai_clockwise.kml
├── Google_Earth.py         # Code to get elevation data from google earth
├── downlink.py             # To be used during testing sessions along with the antenna
├── main.py                 # Run this to launch dashboard
├── real_sim.py             # File to load previous testing sessions under Logs/
├── simulator.py
├── requirements.txt        
├── README.md               # You are here
├── Logs/
│   ├── Bala-10__T8.jsonl
│   ├── abhinav12percent_T4.jsonl
│   ├── telemetry_log_2026-04-26_07-14-13_T4.jsonl
│   └── [30+ other .jsonl telemetry log files]
└── prodbuild/              # Production build assets (Frontend)
    ├── favicon.png
    ├── index.html
    ├── logo.png
    ├── map.html            # New section to display route map
    └── _app/               # Compiled SvelteKit/Vite static assets
        ├── env.js
        ├── version.json
        └── immutable/
            ├── assets/     # Compiled .css styles
            ├── chunks/     # Shared .js code splits
            ├── entry/      # App entry points
            └── nodes/      # Page/Layout routing components
```

---

## 📈 Usage Guide

1. **Live Telemetry and Downlink:** The downlink.py script actively manages and parses incoming telemetry payloads, saving them as sequential .jsonl tracks inside the Logs/ directory..
2. **Simulation Mode:** Run strategy optimization passes by executing simulator.py or real_sim.py to compare expected vehicle speed profiles against battery performance constraints..
3. **Map Verification:** Utilize the provided .kml route strings paired with Google_Earth.py to overlay geographical telemetry profiles directly onto competitive track geometry..Suffix

---

## 🤝 Contributing

We welcome contributions from engineers, data scientists, and strategists on the Agnirath team!
1. Create a new branch: `git checkout -b feature/awesome-strategy`
2. Commit your changes: `git commit -m 'Add some awesome strategy optimization'`
3. Push to the branch: `git push origin feature/awesome-strategy`
4. Open a Pull Request.

---

## 📜 License

This project is proprietary and confidential. Internal use only for the **Agnirath Racing Team**.