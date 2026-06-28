import "../chunks/Bzak7iHL.js";
import {o as _e} from "../chunks/NmWs3qIF.js";
import {Y as ye, Z as xe, _ as Se, $ as Ce, a0 as je, a1 as we, a2 as i, a3 as t, t as g, a4 as X, a5 as h, a6 as a, a7 as k} from "../chunks/B2hAlVi-.js";
import {g as ke, s as V, c as Ae} from "../chunks/DNFaeDbv.js";
import {b as f} from "../chunks/CqHrAp-I.js";
import {s as Te, a as $e} from "../chunks/DkIwFic-.js";
import {C as n, a as Le, L as Pe, P as Be, b as We, c as Me, p as Ee, d as Ie, e as Re} from "../chunks/DHTsbXwT.js";
var Ze = Se('<div class="space-y-6 p-6"><div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8"><div><div class="flex items-center justify-between mb-2"><h3 class="text-lg font-semibold">Speed Status</h3> <div></div></div> <div class="grid grid-cols-2 gap-4"><div><div class="metric-value text-blue-400 svelte-14ry3bj"> </div> <div class="metric-label svelte-14ry3bj">Current Speed</div></div> <div><div class="metric-value text-gray-400 svelte-14ry3bj"> </div> <div class="metric-label svelte-14ry3bj">Predicted Speed</div></div></div> <div class="mt-2 text-sm"><span class="text-gray-400">Margin:</span> <span> </span></div></div> <div class="metric-card svelte-14ry3bj"><div class="metric-value text-purple-400 svelte-14ry3bj"> </div> <div class="metric-label svelte-14ry3bj">Current Acceleration</div></div> <div class="metric-card svelte-14ry3bj"><div class="metric-value text-green-400 svelte-14ry3bj"> </div> <div class="metric-label svelte-14ry3bj">Battery Level</div></div></div> <div class="grid grid-cols-1 xl:grid-cols-2 gap-6"><div class="plot-container xl:col-span-2 svelte-14ry3bj"><canvas class="w-full h-80"></canvas></div> <div class="plot-container svelte-14ry3bj"><canvas class="w-full h-80"></canvas></div> <div class="plot-container svelte-14ry3bj"><canvas class="w-full h-80"></canvas></div> <div style="display:none;" class="plot-container svelte-14ry3bj"><canvas class="w-full h-80"></canvas></div> <div style="display:none;" class="plot-container svelte-14ry3bj"><canvas class="w-full h-80"></canvas></div> <div style="display:none;" class="plot-container xl:col-span-2 svelte-14ry3bj"><canvas class="w-full h-80"></canvas></div><div class="plot-container svelte-14ry3bj"><canvas id="grad-dist" class="w-full h-80"></canvas></div><div class="plot-container svelte-14ry3bj"><canvas id="alt-dist" class="w-full h-80"></canvas></div></div></div>');
function De(ee, te) {
    ye(te, !0);
    const [ae,se] = $e()
      , e = () => Te(ke, "$globalStore", ae);
    n.register(Le, Pe, Be, We, Me, Ee, Ie, Re);
    let _, y, x, S, C, j, r, l, o, d, c, v;
    function m(s, p="", m=1) {
        return typeof s != "number" ? (Array.isArray(s) ? s.map(x => Math.round(x * 10**m) / 10**m) : "N/A") : `${s.toFixed(m)} ${p}`
    }
    function u(s, b, p, N, O=!1, he) {
        return {
            type: "line",
            data: {
                labels: e().historic.Timestamps,
                datasets: [{
                    label: s,
                    data: b,
                    borderColor: p,
                    backgroundColor: p + "20",
                    borderWidth: 2,
                    fill: !1,
                    tension: .1
                }]
            },
            options: {
                responsive: !0,
                maintainAspectRatio: !1,
                scales: {
                    y: {
                        beginAtZero: O,
                        max: he,
                        title: {
                            display: !0,
                            text: N,
                            color: "#fff"
                        },
                        ticks: {
                            color: "#fff"
                        },
                        grid: {
                            color: "#374151"
                        }
                    },
                    x: {
                        title: {
                            display: !0,
                            text: "Time",
                            color: "#fff"
                        },
                        ticks: {
                            color: "#fff"
                        },
                        grid: {
                            color: "#374151"
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: "#fff"
                        }
                    },
                    title: {
                        display: !0,
                        text: `${s} vs Time`,
                        color: "#fff"
                    }
                }
            }
        }
    }
    function ie() {
        return {
            type: "line",
            data: {
                labels: e().historic.Timestamps,
                datasets: [{
                    label: "Speed",
                    data: e().historic.Speed,
                    borderColor: "#3b82f6",
                    backgroundColor: "#3b82f620",
                    borderWidth: 2,
                    fill: !1,
                    tension: .1
                }, {
                    label: "Speed 2",
                    data: e().historic.Speed2 || [],
                    borderColor: "#ef4444",
                    backgroundColor: "#ef444420",
                    borderWidth: 2,
                    fill: !1,
                    tension: .1
                }]
            },
            options: {
                responsive: !0,
                maintainAspectRatio: !1,
                scales: {
                    y: {
                        beginAtZero: !0,
                        title: {
                            display: !0,
                            text: "Speed (km/h)",
                            color: "#fff"
                        },
                        ticks: {
                            color: "#fff"
                        },
                        grid: {
                            color: "#374151"
                        }
                    },
                    x: {
                        title: {
                            display: !0,
                            text: "Time",
                            color: "#fff"
                        },
                        ticks: {
                            color: "#fff"
                        },
                        grid: {
                            color: "#374151"
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: "#fff"
                        }
                    },
                    title: {
                        display: !0,
                        text: "Speed & Speed 2 vs Time",
                        color: "#fff"
                    }
                }
            }
        }
    }
    function re() {
        // 1. Existing live telemetry charts update logic (r, l, o updates here...)
        if (e().historic?.Timestamps?.length > 0) {
            r && (r.data.labels = e().historic.Timestamps, r.data.datasets[0].data = e().historic.Speed, r.data.datasets[1].data= e().historic.Speed2, r.update("none"));
            l && (l.data.labels = e().historic.Timestamps, l.data.datasets[0].data = e().historic.Acceleration || [], l.update("none"));
            o && (o.data.labels = e().historic.Timestamps, o.data.datasets[0].data = e().historic.Altitude || [], o.update("none"));
        }

        // 2. 🔥 FIX: Accurate Chart.js property paths for Distance profiles
        const profileData = e().profile;
        if (profileData && profileData.Distance && profileData.Distance.length > 0) {
            
            if (window.chartAD) {
                if (window.chartAD.data.datasets[0].data.length === 0){
                    window.chartAD.data.labels = m(e().profile.Distance,"",2);
                    window.chartAD.data.datasets[0].data = e().profile.Altitude || []; 
                    window.chartAD.update("none");
                }
            }
            if (window.chartGD) {
                if (window.chartGD.data.datasets[0].data.length === 0){
                window.chartGD.data.labels = m(e().profile.Distance);
                window.chartGD.data.datasets[0].data = e().profile.Gradient || []; 
                window.chartGD.update("none");
                }
            }
        }
    }
    xe( () => {
        const hasHistory = e().historic?.Timestamps?.length > 0;
        const hasProfile = e().profile?.Distance?.length > 0;
        if (hasHistory || hasProfile) {
            re();
        }
    }
    ),
   _e( () => (_ && (r = new n(_,ie())),
    y && (l = new n(y,u("Acceleration", e().historic.Acceleration || [], "#8b5cf6", "Acceleration (m/s²)"))),
    x && (o = new n(x,u("Altitude", e().historic.Altitude || [], "#06b6d4", "Altitude (m)"))),
    
    // 🚀 REMOVED SETTIMEOUT: Initialize immediately since Svelte elements are ready
    (() => {
        let cGD = document.getElementById("grad-dist");
        if (cGD) {
            window.chartGD = new n(cGD,{
                type: "line",
                data: {
                    labels: m(e().profile?.Distance,"",2) || [],
                    datasets: [{
                        label: "Gradient",
                        data: e().profile?.Gradient || [],
                        borderColor: "#3b82f6",
                        backgroundColor: "#3b82f620",
                        borderWidth: 2,
                        fill: !1,
                        tension: .1
                    }]
                },
                options: {
                    responsive: !0,
                    maintainAspectRatio: !1,
                    scales: {
                        y: { title: { display: !0, text: "Gradient (%)", color: "#fff" }, ticks: { color: "#fff" }, grid: { color: "#374151" } },
                        x: { title: { display: !0, text: "Distance (km)", color: "#fff" }, ticks: { color: "#fff" }, grid: { color: "#374151" } }
                    },
                    plugins: { legend: { labels: { color: "#fff" } }, title: { display: !0, text: "Gradient vs Distance", color: "#fff" } }
                }
            });
            window.chartGD.update();
        }

        let cAD = document.getElementById("alt-dist");
        if (cAD) {
            window.chartAD = new n(cAD,{
                type: "line",
                data: {
                    labels: m(e().profile?.Distance,"",2) || [],
                    datasets: [{
                        label: "Altitude",
                        data: e().profile?.Altitude || [],
                        borderColor: "#10b981",
                        backgroundColor: "#10b98120",
                        borderWidth: 2,
                        fill: !1,
                        tension: .1
                    }]
                },
                options: {
                    responsive: !0,
                    maintainAspectRatio: !1,
                    scales: {
                        y: { title: { display: !0, text: "Altitude (m)", color: "#fff" }, ticks: { color: "#fff" }, grid: { color: "#374151" } },
                        x: { title: { display: !0, text: "Distance (km)", color: "#fff" }, ticks: { color: "#fff" }, grid: { color: "#374151" } }
                    },
                    plugins: { legend: { labels: { color: "#fff" } }, title: { display: !0, text: "Altitude vs Distance", color: "#fff" } }
                }
            });
            window.chartAD.update();
        }
    })(), // Executed instantly
    () => {
        r == null || r.destroy(),
        l == null || l.destroy(),
        o == null || o.destroy(),
        window.chartGD && window.chartGD.destroy(),
        window.chartAD && window.chartAD.destroy()
    }
    ));
    const Y = X( () => Math.abs(e().metric.Speed - e().metric.predicted))
      , A = X( () => g(Y) > 3 ? "error" : "ok");
    var T = Ze()
      , $ = t(T)
      , w = t($)
      , L = t(w)
      , le = i(t(L), 2);
    a(L);
    var P = i(L, 2)
      , B = t(P)
      , q = t(B)
      , oe = t(q, !0);
    a(q),
    k(2),
    a(B);
    var z = i(B, 2)
      , D = t(z)
      , de = t(D, !0);
    a(D),
    k(2),
    a(z),
    a(P);
    var G = i(P, 2)
      , W = i(t(G), 2)
      , ce = t(W, !0);
    a(W),
    a(G),
    a(w);
    var M = i(w, 2)
      , H = t(M)
      , ve = t(H, !0);
    a(H),
    k(2),
    a(M);
    var J = i(M, 2)
      , K = t(J)
      , ne = t(K, !0);
    a(K),
    k(2),
    a(J),
    a($);
    var Q = i($, 2)
      , E = t(Q)
      , pe = t(E);
    f(pe, s => _ = s, () => _),
    a(E);
    var I = i(E, 2)
      , fe = t(I);
    f(fe, s => y = s, () => y),
    a(I);
    var R = i(I, 2)
      , me = t(R);
    f(me, s => x = s, () => x),
    a(R);
    var Z = i(R, 2)
      , ue = t(Z);
    f(ue, s => S = s, () => S),
    a(Z);
    var F = i(Z, 2)
      , be = t(F);
    f(be, s => C = s, () => C),
    a(F);
    var U = i(F, 2)
      , ge = t(U);
    f(ge, s => j = s, () => j),
    a(U),
    a(Q),
    a(T),
    Ce( (s, b, p, N, O) => {
        V(w, 1, `metric-card col-span-1 md:col-span-2 ${g(A) === "ok" ? "status-ok" : "status-error"} border-2`, "svelte-14ry3bj"),
        V(le, 1, `w-4 h-4 rounded-full ${g(A) === "ok" ? "bg-green-500" : "bg-red-500"}`),
        h(oe, s),
        h(de, b),
        V(W, 1, Ae(g(A) === "ok" ? "text-green-400" : "text-red-400")),
        h(ce, p),
        h(ve, N),
        h(ne, O)
    }
    , [ () => m(e().metric.Speed, "km/h"), () => m(e().metric.predicted, "km/h"), () => m(g(Y), "km/h"), () => m(e().historic.Acceleration[-1] || 0, "m/s²"), () => m(e().metric.SOC_Ah, "%", 0)]),
    je(ee, T),
    we(),
    se()
}
export {De as component};
