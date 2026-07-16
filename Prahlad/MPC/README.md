# Solar Vehicle Tactical Model Predictive Control (MPC)

This script implements a **Tactical Model Predictive Control (MPC)** framework designed for real-time velocity optimization of a solar-powered racing vehicle. It acts as a localized optimization layer that modifies a high-level strategic target velocity based on immediate telemetry and short-term environmental forecasts.

---

## 1. System Architecture

The localized optimizer minimizes a multi-objective cost function over a finite horizon $N = 10$ steps (20 minutes look-ahead) using sequential least squares programming (`SLSQP`).




```
            ┌────────────────────────────────────────────────────────┐
            │             Strategic High-Level Target                │
            └───────────────────────────┬────────────────────────────┘
                                        │
                                        ▼
        ┌──────────────────────────────────────────────────────────────────┐
        │                   Tactical MPC Optimizer (SLSQP)                 │
        ├──────────────────────────────────────────────────────────────────┤
        │  Inputs:                                                         │
        │  - Current Velocity & Battery State-of-Charge (SoC)              │
        │  - Look-ahead Terrain Topography (Slope radians)                 │
        │  - Shifting Solar Irradiance Profile (W/m²)                      │
        └────────────────────────────────┬─────────────────────────────────┘
                                         │
                                         ▼
            ┌────────────────────────────────────────────────────────┐
            │             Optimal Velocity Control Command           │
            └────────────────────────────────────────────────────────┘

```

---

## 2. Mathematical Framework

### 2.1 Physics & Power Architecture
The underlying physics engine solves the total force balance on the vehicle at every discrete time step $\Delta t$:

$$F_{\text{total}} = F_{\text{drag}} + F_{\text{rolling}} + F_{\text{gravity}} + F_{\text{acceleration}}$$

Where:
* **Aerodynamic Drag:** $F_{\text{drag}} = \frac{1}{2} \rho C_d A \cdot v_k^2$
* **Rolling Resistance:** $F_{\text{rolling}} = m \cdot g \cdot C_{rr} \cdot \cos(\theta_k)$
* **Gravitational Gradient:** $F_{\text{gravity}} = m \cdot g \cdot \sin(\theta_k)$
* **Transient Inertial Force:** $F_{\text{acceleration}} = m \cdot \frac{v_{k+1} - v_k}{\Delta t}$

The structural conversion from mechanical power to net battery electrical flow is given by:

$$P_{\text{net}} = P_{\text{solar}} - P_{\text{electric}} - P_{\text{loss}}$$

$$\text{Where } P_{\text{electric}} = \begin{cases} 
      \frac{F_{\text{total}} \cdot v_k}{\eta_{\text{motor}}} & F_{\text{total}} \geq 0 \\
      F_{\text{total}} \cdot v_k \cdot \eta_{\text{regen}} & F_{\text{total}} < 0 
   \end{cases}$$

### 2.2 Discrete MPC Cost Function
The localized optimizer minimizes a multi-objective cost function over the horizon window:

$$J = \sum_{i=1}^{N} \left[ w_1 (v_i - v_{\text{target}, i})^2 + w_2 (v_i - v_{i-1})^2 + \text{Penalty}_{\text{SoC}}(SoC_i) \right]$$

1. **Velocity Tracking ($w_1 = 1.0$):** Penalizes deviations from the macro-level strategic target velocity.
2. **Drivetrain Smoothing ($w_2 = 0.5$):** Penalizes abrupt acceleration changes to preserve driver comfort and minimize motor thermal stress.
3. **Boundary Protection Penalty:** If the battery State of Charge ($SoC$) falls below the critical $20\%$ threshold, a severe non-linear penalty is injected:
   $$\text{Penalty}_{\text{SoC}} = 1000.0 \times (20.0 - SoC_i)^2$$

---

## 3. Core File Components

* `calculate_net_power()`: The core mechanical-electrical physics model that calculates exact battery energy drain or recharge for a given state change.
* `mpc_cost_function()`: Evaluates the fitness of a proposed velocity vector sequence across the upcoming look-ahead window.
* **Receding Horizon Loop**: Steps sequentially through the race. At each step, it samples current telemetry, runs the localized optimization window via `scipy.optimize.minimize`, applies the *first* optimal velocity command, and slides the look-ahead window forward.
