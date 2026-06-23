import "../chunks/Bzak7iHL.js";
import {i as Vt} from "../chunks/BaVNoogA.js";
import {Y as Wt, aa as S, ab as Gt, _ as O, $, a0 as C, a1 as qt, t as e, A as w, a2 as r, a3 as a, u as d, a5 as u, W as A, a6 as t, au as D, v as St, a4 as L, a9 as Qt, av as P} from "../chunks/B2hAlVi-.js";
import {e as h, i as M} from "../chunks/T28DYXdB.js";
import {s as f, g as zt} from "../chunks/DNFaeDbv.js";
import {a as Jt, s as Xt} from "../chunks/DkIwFic-.js";
var Zt = O('<div><div class="sensor-label svelte-tc5t5g"> </div> <div> </div></div>')
  , te = O('<div class="flag-indicator-row svelte-tc5t5g"><div><div class="status-ring svelte-tc5t5g"></div> <div class="status-core svelte-tc5t5g"></div></div> <div class="flag-info svelte-tc5t5g"><div class="flag-name svelte-tc5t5g"> </div> <div> </div></div></div>')
  , ee = O('<div class="flag-indicator-row svelte-tc5t5g"><div><div class="status-ring svelte-tc5t5g"></div> <div class="status-core svelte-tc5t5g"></div></div> <div class="flag-info svelte-tc5t5g"><div class="flag-name svelte-tc5t5g"> </div> <div> </div></div></div>')
  , ae = O('<div class="compact-sensor-card bg-gray-800/50 border-gray-700/50 backdrop-blur-sm svelte-tc5t5g"><div class="sensor-label svelte-tc5t5g"></div> <div class="sensor-value text-emerald-300 svelte-tc5t5g"> </div></div> <div class="compact-sensor-card bg-gray-800/50 border-gray-700/50 backdrop-blur-sm svelte-tc5t5g"><div class="sensor-label svelte-tc5t5g"></div> <div class="sensor-value text-emerald-300 svelte-tc5t5g"> </div></div>', 1)
  , se = O('<div class="compact-sensor-card bg-gray-800/50 border-gray-700/50 backdrop-blur-sm svelte-tc5t5g"><div class="sensor-label svelte-tc5t5g"> </div> <div class="sensor-value text-blue-300 svelte-tc5t5g"> </div></div>')
  , re = O('<div class="flag-indicator-row svelte-tc5t5g"><div><div class="status-ring svelte-tc5t5g"></div> <div class="status-core svelte-tc5t5g"></div></div> <div class="flag-info svelte-tc5t5g"><div class="flag-name svelte-tc5t5g"> </div> <div> </div></div></div>')
  , ve = O('<div class="flag-indicator-row svelte-tc5t5g"><div><div class="status-ring svelte-tc5t5g"></div> <div class="status-core svelte-tc5t5g"></div></div> <div class="flag-info svelte-tc5t5g"><div class="flag-name svelte-tc5t5g"> </div> <div> </div></div></div>')
  , ie = O('<div class="compact-flag-row svelte-tc5t5g"><div></div> <div class="compact-flag-name svelte-tc5t5g"> </div> <div> </div></div>')
  , ce = O('<div><div class="mppt-header svelte-tc5t5g"><span class="mppt-name svelte-tc5t5g"> </span> <div> </div></div> <div class="temp-readings svelte-tc5t5g"><div class="temp-item svelte-tc5t5g"><span class="temp-label svelte-tc5t5g">MOSFET</span> <span class="temp-value svelte-tc5t5g"> </span></div> <div class="temp-item svelte-tc5t5g"><span class="temp-label svelte-tc5t5g">MPPT</span> <span class="temp-value svelte-tc5t5g"> </span></div></div> <div class="mppt-flags svelte-tc5t5g"><div class="flags-header svelte-tc5t5g">System Status</div> <div class="modern-flags-compact svelte-tc5t5g"></div></div></div>')
  , le = O('<div class="dashboard-container svelte-tc5t5g"><div class="sensor-grid svelte-tc5t5g"><div class="category-panel bg-cyan-500/10 border-cyan-500/30 svelte-tc5t5g"><div class="category-header svelte-tc5t5g"><span class="category-icon svelte-tc5t5g">🌬️</span> <span class="category-title text-cyan-400 svelte-tc5t5g">CABIN AIR QUALITY</span></div> <div class="sensors-compact-grid svelte-tc5t5g"></div></div> <div class="category-panel bg-yellow-500/5 border-yellow-500/20 svelte-tc5t5g"><div class="category-header svelte-tc5t5g"><span class="category-icon svelte-tc5t5g">🔌</span> <span class="category-title text-yellow-400 svelte-tc5t5g">BATTERY CONTACTORS</span> <div> </div></div> <div class="modern-flags-container svelte-tc5t5g"></div></div> <div class="category-panel bg-green-500/5 border-green-500/20 svelte-tc5t5g"><div class="category-header svelte-tc5t5g"><span class="category-icon svelte-tc5t5g">🔋</span> <span class="category-title text-green-400 svelte-tc5t5g">BATTERY MANAGEMENT</span> <div> </div></div> <div class="modern-flags-container svelte-tc5t5g"></div></div> <div class="category-panel bg-emerald-500/5 border-emerald-500/20 svelte-tc5t5g"><div class="category-header svelte-tc5t5g"><span class="category-icon svelte-tc5t5g">🌡️</span> <span class="category-title text-emerald-400 svelte-tc5t5g">CMU TEMPERATURES</span></div> <div class="sensors-compact-grid svelte-tc5t5g"></div></div> <div class="category-panel bg-blue-500/5 border-blue-500/20 lg:col-span-2 svelte-tc5t5g"><div class="category-header svelte-tc5t5g"><span class="category-icon svelte-tc5t5g">⚡</span> <span class="category-title text-blue-400 svelte-tc5t5g">MOTOR SYSTEM & ERRORS</span> <div> </div></div> <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6 svelte-tc5t5g"><div class="svelte-tc5t5g"><h4 class="section-subtitle svelte-tc5t5g">Temperatures</h4> <div class="sensors-compact-grid svelte-tc5t5g"></div></div> <div class="svelte-tc5t5g"><h4 class="section-subtitle svelte-tc5t5g">System Limits</h4> <div class="modern-flags-container svelte-tc5t5g"></div></div></div> <div class="svelte-tc5t5g"><h4 class="section-subtitle svelte-tc5t5g">Motor Errors</h4> <div class="modern-flags-container svelte-tc5t5g"></div></div></div>  <div class="category-panel bg-orange-500/5 border-orange-500/20 lg:col-span-2 svelte-tc5t5g"><div class="category-header svelte-tc5t5g"><span class="category-icon svelte-tc5t5g">☀️</span> <span class="category-title text-orange-400 svelte-tc5t5g">SOLAR MPPT CONTROLLERS</span></div> <div class="mppt-grid svelte-tc5t5g"></div></div></div></div>');
function ue(wt, At) {
    Wt(At, !1);
    const [Et,kt] = Jt()
      , l = () => Xt(zt, "$globalStore", Et)
      , E = w()
      , k = w()
      , H = w()
      , Y = w()
      , N = w()
      , Rt = w()
      , Lt = w()
      , Pt = ["A", "B", "C", "D"]
      , Ut = ["HeatSink_Temp", "DSP_Board_Temp"];
    function R(v, s="", g=1) {
        return `${v.toFixed(g)}${s}`
    }
    function T(v) {
        const s = Object.values(v).filter(Boolean);
        return s.length > 2 ? "error" : s.length > 0 ? "warning" : "ok"
    }
    function jt(v) {
        switch (v) {
        case "ok":
            return "text-green-400";
        case "warning":
            return "text-orange-400";
        case "error":
            return "text-red-400";
        default:
            return "text-gray-400"
        }
    }
    function ot(v) {
        switch (v) {
        case "ok":
            return "bg-green-500/10 border-green-500/30";
        case "warning":
            return "bg-orange-500/10 border-orange-500/50";
        case "error":
            return "bg-red-500/10 border-red-500/70";
        default:
            return "bg-gray-500/10 border-gray-500/30"
        }
    }
    function Nt(v, s) {
        const g = {
            CO: {
                warning: 10,
                error: 25
            },
            CH4: {
                warning: 100,
                error: 500
            },
            NH3: {
                warning: 25,
                error: 50
            },
            NO2: {
                warning: 3,
                error: 10
            },
            O2: {
                warning: 16,
                error: 14
            },
            CO2: {
                warning: 1e3,
                error: 5e3
            }
        }
          , i = s.replace("Cabin_", "").replace("_Content", "")
          , c = g[i];
        return c ? i === "O2" ? v <= c.error ? "error" : v <= c.warning ? "warning" : "ok" : v >= c.error ? "error" : v >= c.warning ? "warning" : "ok" : "ok"
    }
    function U(v) {
        return v.replace(/_/g, " ").replace(/\b\w/g, s => s.toUpperCase()).replace(/Cmu/g, "CMU").replace(/Bms/g, "BMS").replace(/Soc/g, "SOC").replace(/Can/g, "CAN").replace(/Ipm/g, "IPM").replace(/Pwm/g, "PWM").replace(/Dsp/g, "DSP").replace(/Hw/g, "HW").replace(/Uvlo/g, "UVLO")
    }
    S( () => l(), () => {
        A(E, T(l().metric.contactor_flags))
    }
    ),
    S( () => l(), () => {
        A(k, T(l().metric.bmsFlags))
    }
    ),
    S( () => l(), () => {
        A(H, T(l().metric.MotorLimits))
    }
    ),
    S( () => l(), () => {
        A(Y, T(l().metric.MotorErrors))
    }
    ),
    S( () => l(), () => {
        A(N, T({
            ...l().metric.MotorLimits,
            ...l().metric.MotorErrors
        }))
    }
    ),
    S( () => (e(E),
    e(k),
    e(H),
    e(Y),
    l()), () => {
        A(Rt, ( () => {
            const v = [e(E), e(k), e(H), e(Y), ...l().metric.mppts.map(s => T(s.flags))];
            return v.includes("error") ? "error" : v.includes("warning") ? "warning" : "ok"
        }
        )())
    }
    ),
    S( () => l(), () => {
        A(Lt, [...Object.values(l().metric.contactor_flags), ...Object.values(l().metric.bmsFlags), ...Object.values(l().metric.MotorLimits), ...Object.values(l().metric.MotorErrors), ...l().metric.mppts.flatMap(v => Object.values(v.flags))].filter(Boolean).length)
    }
    ),
    Gt(),
    Vt();
    var K = le()
      , gt = a(K)
      , V = a(gt)
      , nt = r(a(V), 2);
    h(nt, 5, () => (l(),
    d( () => Object.entries(l().metric.CabinSensors))), M, (v, s) => {
        var g = L( () => P(e(s), 2));
        let i = () => e(g)[0]
          , c = () => e(g)[1];
        const o = St( () => (i(),
        c(),
        d( () => i().includes("_Content") ? Nt(c(), i()) : "ok")));
        var _ = Zt()
          , m = a(_)
          , n = a(m, !0);
        t(m);
        var b = r(m, 2)
          , p = a(b, !0);
        t(b),
        t(_),
        $( (y, x, j, B) => {
            f(_, 1, `compact-sensor-card ${y ?? ""}`, "svelte-tc5t5g"),
            u(n, x),
            f(b, 1, `sensor-value ${j ?? ""}`, "svelte-tc5t5g"),
            u(p, B)
        }
        , [ () => (D(e(o)),
        d( () => ot(e(o)))), () => (i(),
        d( () => i().replace("Cabin_", "").replace("_", " "))), () => (D(e(o)),
        d( () => jt(e(o)))), () => (c(),
        i(),
        d( () => R(c(), i().includes("Temperature") ? "°C" : i().includes("Pressure") ? "kPa" : i().includes("O2") ? "%" : "ppm")))]),
        C(v, _)
    }
    ),
    t(nt),
    t(V);
    var W = r(V, 2)
      , G = a(W)
      , q = r(a(G), 4)
      , Bt = a(q, !0);
    t(q),
    t(G);
    var _t = r(G, 2);
    h(_t, 5, () => (l(),
    d( () => Object.entries(l().metric.contactor_flags))), M, (v, s) => {
        var g = L( () => P(e(s), 2));
        let i = () => e(g)[0]
          , c = () => e(g)[1];
        var o = te()
          , _ = a(o)
          , m = r(_, 2)
          , n = a(m)
          , b = a(n, !0);
        t(n);
        var p = r(n, 2)
          , y = a(p, !0);
        t(p),
        t(m),
        t(o),
        $(x => {
            const name = i() || "";
            const isInverted = name.includes("output") || name.includes("supply");
            const isSupply = name.includes("supply");
            const isLightActive = isInverted ? !c() : c();
            const textClass = isInverted ? (c() ? "text-green-400" : "text-red-300") : (c() ? "text-red-300" : "text-green-400");
            const textLabel = isInverted ? (isSupply ? (c() ? "OK" : "FAULT") : (c() ? "ON" : "OFF")) : (c() ? "ERROR" : "OK");
            f(_, 1, `flag-status-light ${isLightActive ? "active" : "inactive"}`, "svelte-tc5t5g"),
            u(b, x),
            f(p, 1, `flag-status-text ${textClass}`, "svelte-tc5t5g"),
            u(y, textLabel)
        }
        , [ () => (i(),
        d( () => U(i())))]),
        C(v, o)
    }
    ),
    t(_t),
    t(W);
    var Q = r(W, 2)
      , z = a(Q)
      , J = r(a(z), 4)
      , Ft = a(J, !0);
    t(J),
    t(z);
    var pt = r(z, 2);
    h(pt, 5, () => (l(),
    d( () => Object.entries(l().metric.bmsFlags))), M, (v, s) => {
        var g = L( () => P(e(s), 2));
        let i = () => e(g)[0]
          , c = () => e(g)[1];
        var o = ee()
          , _ = a(o)
          , m = r(_, 2)
          , n = a(m)
          , b = a(n, !0);
        t(n);
        var p = r(n, 2)
          , y = a(p, !0);
        t(p),
        t(m),
        t(o),
        $(x => {
            f(_, 1, `flag-status-light ${c() ? "active" : "inactive"}`, "svelte-tc5t5g"),
            u(b, x),
            f(p, 1, `flag-status-text ${c() ? "text-red-300" : "text-green-400"}`, "svelte-tc5t5g"),
            u(y, c() ? "FAULT" : "OK")
        }
        , [ () => (i(),
        d( () => U(i())))]),
        C(v, o)
    }
    ),
    t(pt),
    t(Q);
    var X = r(Q, 2)
      , ut = r(a(X), 2);
    h(ut, 5, () => (l(),
    d( () => l().metric.cmus)), M, (v, s, g) => {
        var i = ae()
          , c = Qt(i)
          , o = a(c);
        o.textContent = `CMU${g + 1} Temp`;
        var _ = r(o, 2)
          , m = a(_, !0);
        t(_),
        t(c);
        var n = r(c, 2)
          , b = a(n);
        b.textContent = `Cell${g + 1} Temp`;
        var p = r(b, 2)
          , y = a(p, !0);
        t(p),
        t(n),
        $( (x, j) => {
            u(m, x),
            u(y, j)
        }
        , [ () => (e(s),
        d( () => R(e(s).temperature, "°C"))), () => (e(s),
        d( () => R(e(s).cell_temperature, "°C")))]),
        C(v, i)
    }
    ),
    t(ut),
    t(X);
    var Z = r(X, 2)
      , tt = a(Z)
      , et = r(a(tt), 4)
      , It = a(et, !0);
    t(et),
    t(tt);
    var at = r(tt, 2)
      , st = a(at)
      , mt = r(a(st), 2);
    h(mt, 5, () => Ut, M, (v, s) => {
        var g = se()
          , i = a(g)
          , c = a(i, !0);
        t(i);
        var o = r(i, 2)
          , _ = a(o, !0);
        t(o),
        t(g),
        $( (m, n) => {
            u(c, m),
            u(_, n)
        }
        , [ () => (e(s),
        d( () => e(s).replace("_", " "))), () => (l(),
        e(s),
        d( () => R(l().metric[e(s)], "°C")))]),
        C(v, g)
    }
    ),
    t(mt),
    t(st);
    var ft = r(st, 2)
      , bt = r(a(ft), 2);
    h(bt, 5, () => (l(),
    d( () => Object.entries(l().metric.MotorLimits))), M, (v, s) => {
        var g = L( () => P(e(s), 2));
        let i = () => e(g)[0]
          , c = () => e(g)[1];
        var o = re()
          , _ = a(o)
          , m = r(_, 2)
          , n = a(m)
          , b = a(n, !0);
        t(n);
        var p = r(n, 2)
          , y = a(p, !0);
        t(p),
        t(m),
        t(o),
        $(x => {
            f(_, 1, `flag-status-light ${c() ? "active" : "inactive"}`, "svelte-tc5t5g"),
            u(b, x),
            f(p, 1, `flag-status-text ${c() ? "text-orange-300" : "text-green-400"}`, "svelte-tc5t5g"),
            u(y, c() ? "LIMITED" : "NORMAL")
        }
        , [ () => (i(),
        d( () => U(i())))]),
        C(v, o)
    }
    ),
    t(bt),
    t(ft),
    t(at);
    var yt = r(at, 2)
      , xt = r(a(yt), 2);
    h(xt, 5, () => (l(),
    d( () => Object.entries(l().metric.MotorErrors))), M, (v, s) => {
        var g = L( () => P(e(s), 2));
        let i = () => e(g)[0]
          , c = () => e(g)[1];
        var o = ve()
          , _ = a(o)
          , m = r(_, 2)
          , n = a(m)
          , b = a(n, !0);
        t(n);
        var p = r(n, 2)
          , y = a(p, !0);
        t(p),
        t(m),
        t(o),
        $(x => {
            f(_, 1, `flag-status-light ${c() ? "active error" : "inactive"}`, "svelte-tc5t5g"),
            u(b, x),
            f(p, 1, `flag-status-text ${c() ? "text-red-300" : "text-green-400"}`, "svelte-tc5t5g"),
            u(y, c() ? "ERROR" : "OK")
        }
        , [ () => (i(),
        d( () => U(i())))]),
        C(v, o)
    }
    ),
    t(xt),
    t(yt),
    t(Z);
    var $t = r(Z, 2)
      , Ct = r(a($t), 2);
    h(Ct, 5, () => (l(),
    d( () => l().metric.mppts)), M, (v, s, g) => {
        const i = St( () => (e(s),
        d( () => T(e(s).flags))));
        var c = ce()
          , o = a(c)
          , _ = a(o)
          , m = a(_, !0);
        t(_);
        var n = r(_, 2)
          , b = a(n, !0);
        t(n),
        t(o);
        var p = r(o, 2)
          , y = a(p)
          , x = r(a(y), 2)
          , j = a(x, !0);
        t(x),
        t(y);
        var B = r(y, 2)
          , Ot = r(a(B), 2)
          , Dt = a(Ot, !0);
        t(Ot),
        t(B),
        t(p);
        var ht = r(p, 2)
          , Mt = r(a(ht), 2);
        h(Mt, 5, () => (e(s),
        d( () => Object.entries(e(s).flags))), M, (rt, vt) => {
            var F = L( () => P(e(vt), 2));
            let I = () => e(F)[0]
              , it = () => e(F)[1];
            var ct = ie()
              , Tt = a(ct)
              , lt = r(Tt, 2)
              , Ht = a(lt, !0);
            t(lt);
            var dt = r(lt, 2)
              , Yt = a(dt, !0);
            t(dt),
            t(ct),
            $(Kt => {
                f(Tt, 1, `flag-status-dot ${it() ? "active" : "inactive"}`, "svelte-tc5t5g"),
                u(Ht, Kt),
                f(dt, 1, `compact-flag-status ${it() ? "text-red-300" : "text-green-400"}`, "svelte-tc5t5g"),
                u(Yt, it() ? "⚠" : "✓")
            }
            , [ () => (I(),
            d( () => U(I())))]),
            C(rt, ct)
        }
        ),
        t(Mt),
        t(ht),
        t(c),
        $( (rt, vt, F, I) => {
            f(c, 1, `mppt-card ${rt ?? ""} backdrop-blur-sm`, "svelte-tc5t5g"),
            u(m, d( () => Pt[g])),
            f(n, 1, `status-pill ${e(i) ?? ""}`, "svelte-tc5t5g"),
            u(b, vt),
            u(j, F),
            u(Dt, I)
        }
        , [ () => (D(e(i)),
        d( () => ot(e(i)))), () => (D(e(i)),
        d( () => e(i).toUpperCase())), () => (e(s),
        d( () => R(e(s).Mosfet_Temperature, "°C"))), () => (e(s),
        d( () => R(e(s).MPPT_Temperature, "°C")))]),
        C(v, c)
    }
    ),
    t(Ct),
    t($t),
    t(gt),
    t(K),
    $( (v, s, g) => {
        f(q, 1, `status-pill ${e(E) ?? ""}`, "svelte-tc5t5g"),
        u(Bt, v),
        f(J, 1, `status-pill ${e(k) ?? ""}`, "svelte-tc5t5g"),
        u(Ft, s),
        f(et, 1, `status-pill ${e(N) ?? ""}`, "svelte-tc5t5g"),
        u(It, g)
    }
    , [ () => (e(E),
    d( () => e(E).toUpperCase())), () => (e(k),
    d( () => e(k).toUpperCase())), () => (e(N),
    d( () => e(N).toUpperCase()))]),
    C(wt, K),
    qt(),
    kt()
}
export {ue as component};
