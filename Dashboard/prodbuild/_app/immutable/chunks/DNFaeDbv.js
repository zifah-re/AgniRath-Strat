import {h as _, aB as f} from "./B2hAlVi-.js";
function m(e) {
    var i, a, t = "";
    if (typeof e == "string" || typeof e == "number")
        t += e;
    else if (typeof e == "object")
        if (Array.isArray(e)) {
            var s = e.length;
            for (i = 0; i < s; i++)
                e[i] && (a = m(e[i])) && (t && (t += " "),
                t += a)
        } else
            for (a in e)
                e[a] && (t && (t += " "),
                t += a);
    return t
}
function b() {
    for (var e, i, a = 0, t = "", s = arguments.length; a < s; a++)
        (e = arguments[a]) && (i = m(e)) && (t && (t += " "),
        t += i);
    return t
}
function A(e) {
    return typeof e == "object" ? b(e) : e ?? ""
}
function v(e, i, a) {
    var t = e == null ? "" : "" + e;
    return i && (t = t ? t + " " + i : i),
    t === "" ? null : t
}
function O(e, i) {
    return e == null ? null : String(e)
}
function P(e, i, a, t, s, r) {
    var o = e.__className;
    if (_ || o !== a || o === void 0) {
        var n = v(a, t);
        (!_ || n !== e.getAttribute("class")) && (n == null ? e.removeAttribute("class") : e.className = n),
        e.__className = a
    }
    return r
}
const u = {
    metric: {
        Pack_Voltage: 0,
        SOC_Ah: 0,
        power_consumption: 0,
        solar_input: 0,
        distance_travelled: 0,
        Motor_Temp: 0,
        Speed: 0,
        predicted: 0,
        Pack_Current: 0,
        cmus: Array.from({
            length: 5
        }, () => ({
            temperature: 0,
            cell_temperature: 0,
            cell_voltages: Array.from({
                length: 8
            }, () => 0)
        })),
        battery_ranges: {
            min_temp: 0,
            max_temp: 0,
            min_volt: 0,
            max_volt: 0
        },
        precharge_state: 0,
        contactor_flags: {
            contactor1_error: !1,
            contactor2_error: !1,
            contactor3_error: !1,
            contactor1_output: !1,
            contactor2_output: !1,
            contactor3_output: !1,
            contactor_supply: !1
        },
        bmsFlags: {
            cell_over_voltage: !1,
            cell_under_voltage: !1,
            cell_over_temp: !1,
            measurement_untrusted: !1,
            cmu_comm_timeout: !1,
            vehicle_comm_timeout: !1,
            bms_setup_mode: !1,
            cmu_can_status: !1,
            isolation_test_fail: !1,
            soc_invalid: !1,
            can_supply_low: !1,
            contactor_not_engaged: !1,
            extra_cell_detected: !1
        },
        Motor_Velocity: 0,
        Speed2: 0,
        HeatSink_Temp: 0,
        PhaseA_Current: 0,
        PhaseB_Current: 0,
        PhaseC_Current: 0,
        Bus_Voltage: 0,
        Bus_Current: 0,
        Bus_Power: 0,
        DSP_Board_Temp: 0,
        MotorLimits: {
            ipm_temp_limit: !1,
            bus_voltage_lower_limit: !1,
            bus_voltage_upper_limit: !1,
            bus_current_limit: !1,
            velocity_limit: !1,
            motor_current_limit: !1,
            output_voltage_pwm_limit: !1
        },
        MotorErrors: {
            motor_over_speed: !1,
            desaturation_fault: !1,
            rail_15v_uvlo: !1,
            config_read_error: !1,
            watchdog_reset: !1,
            bad_motor_position: !1,
            dc_bus_over_voltage: !1,
            software_over_current: !1,
            hardware_over_current: !1
        },
        mppts: Array.from({
            length: 4
        }, () => ({
            Input_Voltage: 0,
            Input_Current: 0,
            Output_Voltage: 0,
            Output_Current: 0,
            Output_Power: 0,
            efficiency: 0,
            Mosfet_Temperature: 0,
            MPPT_Temperature: 0,
            flags: {
                hw_overvolt: !1,
                hw_overcurrent: !1,
                under12v: !1,
                low_array_power: !1,
                battery_full: !1,
                battery_low: !1,
                mosfet_overheat: !1
            }
        })),
        CabinSensors: {
            Cabin_CO_Content: 0,
            Cabin_CH4_Content: 0,
            Cabin_NH3_Content: 0,
            Cabin_NO2_Content: 0,
            Cabin_O2_Content: 0,
            Cabin_Temperature: 0,
            Cabin_Pressure: 0,
            Cabin_CO2_Content: 0
        },
        Latitude: 0.0,
        Longitude: 0.0,
        Altitude: 0.0,
        Gradient: 0.0,
        Bearing: 0.0
    },
    historic: {
        Timestamps: [],
        Speed: [],
        Battery: [],
        Power: [],
        Solar: [],
        PhaseA_Current: [],
        Bus_Power: [],
        Motor_Velocity: [],
        Speed2: [],
        solar_input_voltage: [],
        solar_output_power: [],
        solar_mppt_A_Output_Voltage: [],
        solar_mppt_B_Output_Voltage: [],
        solar_mppt_C_Output_Voltage: [],
        solar_mppt_D_Output_Voltage: [],
        solar_mppt_A_Output_Power: [],
        solar_mppt_B_Output_Power: [],
        solar_mppt_C_Output_Power: [],
        solar_mppt_D_Output_Power: [],
        solar_mppt_A_Output_Current: [],
        solar_mppt_B_Output_Current: [],
        solar_mppt_C_Output_Current: [],
        solar_mppt_D_Output_Current: [],
        Acceleration: [],
        Altitude: [],
        Latitudes: [],
        Longitudes: []
    },
    profile: {
        Altitude: [],
        Gradient: [],
        Coordinates: [],
        Distance: [],
        SpeedLimit: []
    }
}
  , c = f([]);
function g(e) {
    return e.split("_").map(i => i.charAt(0).toUpperCase() + i.slice(1)).join(" ")
}
function C(e) {
    const i = ["cell_over_voltage", "cell_under_voltage", "cell_over_temp", "isolation_test_fail", "extra_cell_detected"]
      , a = ["measurement_untrusted", "cmu_comm_timeout", "vehicle_comm_timeout", "soc_invalid", "can_supply_low", "contactor_not_engaged"];
    return i.includes(e) ? "error" : a.includes(e) ? "warning" : "info"
}
function w() {
    const {subscribe: e, set: i, update: a} = f(u);
    return {
        subscribe: e,
        notifications: c,
        updateValue: (t, s) => a(r => (r.metric[t] = s,
        r)),
        appendToArray: (t, s) => a(r => {
            const o = r.historic[t];
            return Array.isArray(o) ? (r.historic[t] = [...o, ...s],
            r) : (console.error(`Key ${String(t)} is not an array`),
            r)
        }
        ),
        reset: () => i(u),
        addNotification: t => {
            const s = {
                ...t,
                id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
                timestamp: new Date
            };
            c.update(r => [s, ...r].slice(0, 50))
        }
        ,
        removeNotification: t => {
            c.update(s => s.filter(r => r.id !== t))
        }
        ,
        clearNotifications: () => {
            c.set([])
        }
        ,
        handleWebSocketUpdate: t => {
            try {
                a(s => {
                    const r = {
                        ...s
                    };
                    return t.metric && (t.metric.bmsFlags && r.metric.bmsFlags && Object.keys(t.metric.bmsFlags).forEach(o => {
                        const n = r.metric.bmsFlags[o]
                          , p = t.metric.bmsFlags[o];
                        if (!n && p) {
                            const l = g(o)
                              , d = C(o);
                            setTimeout( () => {
                                S.addNotification({
                                    flagName: o,
                                    message: `BMS Alert: ${l} flag activated`,
                                    severity: d
                                })
                            }
                            , 0),
                            fetch("https://ntfy.sh/agnirath_telemtry", {
                                method: "POST",
                                body: `🚨 Agnirath BMS Alert: ${l} flag activated`
                            }).catch(h => {
                                console.warn("Failed to send push notification:", h)
                            }
                            )
                        }
                    }
                    ),
                    Object.keys(t.metric).forEach(o => {
                        t.metric && o in r.metric && (r.metric[o] = t.metric[o])
                    }
                    )),
                    t.historic && Object.keys(t.historic).forEach(o => {
                        if (t.historic && t.historic[o] != null) {
                            r.historic[o] || (r.historic[o] = []);
                            r.historic[o] = [...r.historic[o], t.historic[o]];
                            const n = 1e3;
                            r.historic[o].length > n && (r.historic[o] = r.historic[o].slice(-n))
                        }
                    }
                    ),
                    t.profile && Object.keys(t.profile).forEach(o => {
                        if (t.profile && t.profile[o] != null) {
                            r.profile[o] || (r.profile[o] = []);
                            r.profile[o] = [...r.profile[o], t.profile[o]];
                            const n = 1e3;
                            r.profile[o].length > n && (r.profile[o] = r.profile[o].slice(-n))
                        }
                    }
                    ),
                    r
                }
                )
            } catch (s) {
                console.error("Error processing WebSocket message:", s)
            }
        }
        ,
        handleWebSocketData: t => {
            try {
                a(s => {
                    const r = {
                        ...s
                    };
                    return t.metric && Object.keys(t.metric).forEach(n => {
                        n in r.metric && t.metric[n] !== void 0 && (r.metric[n] = t.metric[n])
                    }
                    ),
                    t.historic && Object.keys(t.historic).forEach(n => {
                        t.historic[n] && (r.historic[n] = [...t.historic[n]])
                    }
                    ),
                    t.profile && Object.keys(t.profile).forEach(n => {
                        t.profile[n] && (r.profile[n] = [...t.profile[n]])
                    }
                    ),
                    r
                }
                )
            } catch (s) {
                console.error("Error processing WebSocket message:", s)
            }
        }
    }
}
const S = w();
export {A as c, S as g, P as s, O as t};
