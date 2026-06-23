# Agnirath Strategy Dashboard 🏎️📊

Welcome to the **Agnirath Strategy Dashboard** repository[cite: 1]. This interactive dashboard is designed for real-time race telemetry analysis, strategy simulation, and predictive analytics for the Agnirath solar/electric racing team[cite: 1]. It helps strategists make data-driven decisions regarding energy management, speed profiles, and pit-stop optimization during competitive tracks[cite: 1].

---

## 🚀 Key Features

- **Real-Time Telemetry Visualization:** Live streaming of battery state-of-charge (SoC), cell temperatures, motor RPM, and solar array power generation[cite: 1].
- **Predictive Energy Modeling:** Advanced ML models to forecast energy consumption based on elevation profiles, weather conditions, and track friction[cite: 1].
- **Weather Integration:** Dynamic fetching of solar irradiance data mapped directly onto track coordinates[cite: 1].
- **Historical Analysis:** Post-race review modules to overlay actual run profiles against theoretical optima for future continuous improvement[cite: 1].

---

## 🛠️ Tech Stack

- **Frontend/Dashboard:** Built with static web assets running on a SvelteKit/Vite production build pipeline.[cite: 1]
- **Data Visualization:** Interactive tracking and map interfaces (map.html, Google_Earth.py).[cite: 1]
- **Backend/Analytics:** Python-based simulation engine (simulator.py, real_sim.py) and live data telemetry handling (downlink.py).[cite: 1]

---

## 📦 Installation & Setup

### Prerequisites
Make sure you have Python 3.10+ and `pip` installed[cite: 1].

### 1. Clone the Repository
```bash
git clone [https://github.com/your-team/agnirath-strategy-dashboard.git](https://github.com/your-team/agnirath-strategy-dashboard.git)
cd agnirath-strategy-dashboard
```[cite: 1]

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```[cite: 1]

### 5. Run the Dashboard Local Server
```bash
python main.py
```[cite: 1]

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
```[cite: 1]

---

## 📈 Usage Guide

1. **Live Telemetry and Downlink:** The downlink.py script actively manages and parses incoming telemetry payloads, saving them as sequential .jsonl tracks inside the Logs/ directory.[cite: 1].
2. **Simulation Mode:** Run strategy optimization passes by executing simulator.py or real_sim.py to compare expected vehicle speed profiles against battery performance constraints.[cite: 1].
3. **Map Verification:** Utilize the provided .kml route strings paired with Google_Earth.py to overlay geographical telemetry profiles directly onto competitive track geometry.[cite: 1].Suffix

---

## 🤝 Contributing

We welcome contributions from engineers, data scientists, and strategists on the Agnirath team![cite: 1]
1. Create a new branch: `git checkout -b feature/awesome-strategy`[cite: 1]
2. Commit your changes: `git commit -m 'Add some awesome strategy optimization'`[cite: 1]
3. Push to the branch: `git push origin feature/awesome-strategy`[cite: 1]
4. Open a Pull Request[cite: 1].

---

## 📜 License

This project is proprietary and confidential[cite: 1]. Internal use only for the **Agnirath Racing Team**[cite: 1].