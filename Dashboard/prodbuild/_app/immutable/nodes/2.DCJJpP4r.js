import "../chunks/Bzak7iHL.js";
import {o as $e} from "../chunks/NmWs3qIF.js";
import {Y as Pe, Z as Te, _ as Le, $ as Be, a0 as We, a1 as Ae, a2 as i, a3 as e, t as g, a4 as te, a5 as v, a6 as t, a7 as n} from "../chunks/B2hAlVi-.js";
import {s as j, c as Me, g as Ve} from "../chunks/DNFaeDbv.js";
import {b as y} from "../chunks/CqHrAp-I.js";
import {s as Ie, a as Ee} from "../chunks/DkIwFic-.js";
import {C as u, a as Re, L as Ze, P as je, b as De, c as Fe, p as Ne, d as Oe, e as Ye} from "../chunks/DHTsbXwT.js";
var qe = Le('<div class="space-y-6 p-6"><div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4"><div><div class="flex items-center justify-between mb-2"><h3 class="text-lg font-semibold">Speed Status</h3> <div></div></div> <div class="grid grid-cols-2 gap-4"><div><div class="metric-value text-blue-400 svelte-g8vw4"> </div> <div class="metric-label svelte-g8vw4">Current Speed</div></div> <div><div class="metric-value text-gray-400 svelte-g8vw4"> </div> <div class="metric-label svelte-g8vw4">Predicted Speed</div></div></div> <div class="mt-2 text-sm"><span class="text-gray-400">Margin:</span> <span> </span></div></div> <div class="metric-card svelte-g8vw4"><div class="metric-value text-yellow-400 svelte-g8vw4"> </div> <div class="metric-label svelte-g8vw4">Pack Voltage</div></div> <div class="metric-card svelte-g8vw4"><div class="metric-value text-green-400 svelte-g8vw4"> </div> <div class="metric-label svelte-g8vw4">Battery Level</div></div></div> <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4"><div class="metric-card svelte-g8vw4"><div class="metric-value text-red-400 svelte-g8vw4"> </div> <div class="metric-label svelte-g8vw4">Power Consumption</div></div> <div class="metric-card svelte-g8vw4"><div class="metric-value text-yellow-400 svelte-g8vw4"> </div> <div class="metric-label svelte-g8vw4">Solar Input</div></div> <div class="metric-card svelte-g8vw4"><div class="metric-value text-purple-400 svelte-g8vw4"> </div> <div class="metric-label svelte-g8vw4">Distance Travelled</div></div> <div class="metric-card svelte-g8vw4"><div class="metric-value text-orange-400 svelte-g8vw4"> </div> <div class="metric-label svelte-g8vw4">Road Slope</div></div></div> <div class="grid grid-cols-1 xl:grid-cols-2 gap-6"><div class="plot-container svelte-g8vw4"><canvas class="w-full h-80"></canvas></div> <div class="plot-container svelte-g8vw4"><canvas class="w-full h-80"></canvas></div> <div class="plot-container svelte-g8vw4"><canvas class="w-full h-80"></canvas></div> <div class="plot-container svelte-g8vw4"><canvas class="w-full h-80"></canvas></div><div class="plot-container svelte-g8vw4"><canvas id="dist-time" class="w-full h-80"></canvas></div><div class="plot-container svelte-g8vw4"><canvas id="vel-dist" class="w-full h-80"></canvas></div></div></div>');
function Xe(ae, se) {
    Pe(se, !0);
    const [ie,re] = Ee()
      , a = () => Ie(Ve, "$globalStore", ie);
    u.register(Re, Ze, je, De, Fe, Ne, Oe, Ye);
    const D = te( () => Math.abs(a().metric.Speed2 - a().metric.predicted))
      , S = te( () => g(D) > 3 ? "error" : "ok");
    let f, _, w, b, l, d, c, o;
    function r(s, p="", m=1) {
        return typeof s != "number" ? (Array.isArray(s) ? s.map(x => Math.round(x * 10**m) / 10**m) : "N/A") : `${s.toFixed(m)} ${p}`
    }
    function h(s, p, m, Z) {
        return {
            type: "line",
            data: {
                labels: a().historic.Timestamps,
                datasets: [{
                    label: s,
                    data: p,
                    borderColor: m,
                    backgroundColor: m + "20",
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
                        beginAtZero: s === "Battery Level",
                        max: s === "Battery Level" ? 100 : void 0,
                        title: {
                            display: !0,
                            text: Z,
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
    function ve() {
        if (a().historic.Timestamps.length !== 0) {
            if (l) {
                l.data.labels = a().historic.Timestamps;
                l.data.datasets[0].data = a().historic.Speed2;
                l.update("none")
            }
            if (d) {
                d.data.labels = a().historic.Timestamps;
                d.data.datasets[0].data = a().historic.Battery;
                d.update("none")
            }
            if (c) {
                c.data.labels = a().historic.Timestamps;
                c.data.datasets[0].data = a().historic.Power;
                c.update("none")
            }
            if (o) {
                o.data.labels = a().historic.Timestamps;
                o.data.datasets[0].data = a().historic.Solar;
                o.update("none")
            }
            if (typeof cDT !== "undefined" && cDT) {
                cDT.data.labels = a().historic.Timestamps;
                cDT.data.datasets[0].data = a().historic.Distance;
                cDT.update("none")
            }
            if (typeof cVD !== "undefined" && cVD) {
                cVD.data.labels = r(a().historic.Distance,"",2);
                cVD.data.datasets[0].data = a().historic.Speed2;
                cVD.update("none")
            }
        }
    }
    Te( () => {
        a().historic.Timestamps.length > 0 && ve()
    }
    );
    let cDT, cVD;
    $e( () => {
        if (f)
            l = new u(f,h("Speed", a().historic.Speed2, "#3b82f6", "Speed (km/h)"));
        if (_)
            d = new u(_,h("Battery Level", a().historic.Battery, "#10b981", "Battery Level (%)"));
        if (w)
            c = new u(w,h("Power Consumption", a().historic.Power, "#f59e0b", "Power (W)"));
        if (b)
            o = new u(b,h("Solar Input", a().historic.Solar, "#eab308", "Solar Input (W)"));
        setTimeout( () => {
            let eDT = document.getElementById("dist-time");
            if (eDT)
                cDT = new u(eDT,h("Distance Travelled", a().historic.Distance, "#a855f7", "Distance (km)"));
            let eVD = document.getElementById("vel-dist");
            if (eVD)
                cVD = new u(eVD,{
                    type: "line",
                    data: {
                        labels: r(a().historic.Distance,"",2),
                        datasets: [{
                            label: "Velocity",
                            data: a().historic.Speed2,
                            borderColor: "#ec4899",
                            backgroundColor: "#ec489920",
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
                                title: {
                                    display: !0,
                                    text: "Velocity (km/h)",
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
                                    text: "Distance (km)",
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
                                text: `Velocity vs Distance`,
                                color: "#fff"
                            }
                        }
                    }
                });
        }
        , 50);
        return () => {
            if (l)
                l.destroy();
            if (d)
                d.destroy();
            if (c)
                c.destroy();
            if (o)
                o.destroy();
            if (cDT)
                cDT.destroy();
            if (cVD)
                cVD.destroy()
        }
    }
    );
    var k = qe()
      , C = e(k)
      , x = e(C)
      , $ = e(x)
      , le = i(e($), 2);
    t($);
    var P = i($, 2)
      , T = e(P)
      , F = e(T)
      , de = e(F, !0);
    t(F),
    n(2),
    t(T);
    var N = i(T, 2)
      , O = e(N)
      , ce = e(O, !0);
    t(O),
    n(2),
    t(N),
    t(P);
    var Y = i(P, 2)
      , L = i(e(Y), 2)
      , oe = e(L, !0);
    t(L),
    t(Y),
    t(x);
    var B = i(x, 2)
      , q = e(B)
      , ne = e(q, !0);
    t(q),
    n(2),
    t(B);
    var z = i(B, 2)
      , G = e(z)
      , me = e(G, !0);
    t(G),
    n(2),
    t(z),
    t(C);
    var W = i(C, 2)
      , A = e(W)
      , H = e(A)
      , pe = e(H, !0);
    t(H),
    n(2),
    t(A);
    var M = i(A, 2)
      , J = e(M)
      , ge = e(J, !0);
    t(J),
    n(2),
    t(M);
    var V = i(M, 2)
      , K = e(V)
      , ue = e(K, !0);
    t(K),
    n(2),
    t(V);
    var Q = i(V, 2)
      , U = e(Q)
      , fe = e(U, !0);
    t(U),
    n(2),
    t(Q),
    t(W);
    var X = i(W, 2)
      , I = e(X)
      , _e = e(I);
    y(_e, s => f = s, () => f),
    t(I);
    var E = i(I, 2)
      , we = e(E);
    y(we, s => _ = s, () => _),
    t(E);
    var R = i(E, 2)
      , be = e(R);
    y(be, s => w = s, () => w),
    t(R);
    var ee = i(R, 2)
      , he = e(ee);
    y(he, s => b = s, () => b),
    t(ee),
    t(X),
    t(k),
    Be( (s, p, m, Z, xe, ye, Se, ke, Ce) => {
        j(x, 1, `metric-card col-span-1 md:col-span-2 ${g(S) === "ok" ? "status-ok" : "status-error"} border-2`, "svelte-g8vw4"),
        j(le, 1, `w-4 h-4 rounded-full ${g(S) === "ok" ? "bg-green-500" : "bg-red-500"}`),
        v(de, s),
        v(ce, p),
        j(L, 1, Me(g(S) === "ok" ? "text-green-400" : "text-red-400")),
        v(oe, m),
        v(ne, Z),
        v(me, xe),
        v(pe, ye),
        v(ge, Se),
        v(ue, ke),
        v(fe, Ce)
    }
    , [ () => r(a().metric.Speed2, "km/h"), () => r(a().metric.predicted, "km/h"), () => r(g(D), "km/h"), () => r(a().metric.Pack_Voltage, "V"), () => r(a().metric.SOC_Ah, "%", 0), () => r(a().metric.Bus_Power, "W", 0), () => r(a().metric.solar_input, "W", 0), () => r(a().metric.distance_travelled, "km"), () => r(a().metric.Gradient, "%", 3)]),
    We(ae, k),
    Ae(),
    re()
}
export {Xe as component};
