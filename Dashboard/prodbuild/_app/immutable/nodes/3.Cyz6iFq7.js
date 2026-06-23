import "../chunks/Bzak7iHL.js";
import {i as et} from "../chunks/BaVNoogA.js";
import {Y as tt, _ as p, $ as P, a0 as g, a1 as at, a2 as a, a3 as t, a5 as l, a6 as e, a7 as A, t as v, v as te, a8 as rt, a9 as st} from "../chunks/B2hAlVi-.js";
import {i as ae} from "../chunks/3d2OysfF.js";
import {e as N, i as R} from "../chunks/T28DYXdB.js";
import {s as m, g as it} from "../chunks/DNFaeDbv.js";
import {s as re} from "../chunks/zvPHBe4h.js";
import{a as vt,s as lt}from"../chunks/DkIwFic-.js";import {C as Chart, a as C_a, L as C_L, P as C_P, b as C_b, c as C_c, p as C_p, d as C_d, e as C_e} from "../chunks/DHTsbXwT.js";Chart.register(C_a, C_L, C_P, C_b, C_c, C_p, C_d, C_e);
var dt = p('<div class="w-px h-4 bg-gray-600"></div>')
  , ct = p('<div class="flex items-center space-x-3"><span class="text-sm text-gray-300"> </span> <div class="flex items-center space-x-1"><span class="text-sm text-gray-400">O:</span> <div></div></div> <div class="flex items-center space-x-1"><span class="text-sm text-gray-400">E:</span> <div></div></div> <!></div>')
  , nt = p('<div class="flex items-center justify-center space-x-1"><div></div> <span class="text-xs text-gray-300"> </span></div>')
  , ot = p('<div class="bg-gray-600 rounded p-3 text-center"><div class="text-xs text-gray-400 mb-1"></div> <div class="text-sm font-medium mb-2"> </div> <div class="w-full h-2 rounded-full bg-gray-500"><div></div></div></div>')
  , _t = p('<div class="bg-gray-600 rounded p-3 text-center"><div class="text-xs text-gray-400 mb-1"></div> <div class="text-sm font-medium mb-2"></div> <div class="w-full h-2 rounded-full bg-gray-500"><div class="h-2 rounded-full bg-gray-500"></div></div></div>')
  , gt = p('<div style="display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 0.75rem;"></div>')
  , mt = p('<div class="text-center text-gray-400 py-4"></div>')
  , pt = p('<div class="bg-gray-700 rounded-lg p-4 border border-gray-600"><div class="flex items-center justify-between mb-3"><div class="flex items-center space-x-4"><h4 class="text-lg font-medium text-white"></h4> <div class="flex items-center space-x-4"><div class="flex items-center space-x-2"><div></div> <span class="text-sm text-gray-300"> </span></div> <div class="flex items-center space-x-2"><div></div> <span class="text-sm text-gray-300"> </span></div></div></div></div> <!></div>')
  , ut = p('<div class="space-y-6 p-6"><div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4"><div class="metric-card col-span-1 md:col-span-2 svelte-23yu5p"><div class="metric-value text-green-400 text-3xl svelte-23yu5p"> </div> <div class="metric-label svelte-23yu5p">State of Charge</div> <div class="mt-2 bg-gray-700 rounded-full h-2"><div class="bg-green-500 h-2 rounded-full transition-all duration-300"></div></div></div> <div class="metric-card svelte-23yu5p"><div class="metric-value text-yellow-400 svelte-23yu5p"> </div> <div class="metric-label svelte-23yu5p">Pack Voltage</div></div> <div class="metric-card svelte-23yu5p"><div class="metric-value text-blue-400 svelte-23yu5p"> </div> <div class="metric-label svelte-23yu5p">Pack Current</div></div> <div class="hidden"><div class="metric-value text-red-400 svelte-23yu5p"> </div> <div class="metric-label svelte-23yu5p">Max Temperature</div></div> <div class="hidden"><div class="metric-value text-cyan-400 svelte-23yu5p"> </div> <div class="metric-label svelte-23yu5p">Min Temperature</div></div></div> <div class="grid grid-cols-1 md:grid-cols-3 gap-4"><div class="metric-card svelte-23yu5p"><div class="flex items-center justify-between"><span class="metric-label svelte-23yu5p">Voltage Range</span> <span class="text-sm text-gray-300"> </span></div></div> <div class="metric-card svelte-23yu5p"><div class="flex items-center justify-between"><span class="metric-label svelte-23yu5p">Temperature Range</span> <span class="text-sm text-gray-300"> </span></div></div> <div class="metric-card svelte-23yu5p"><div class="flex items-center justify-between"><span class="metric-label svelte-23yu5p">Pre-charge State</span> <div class="flex items-center space-x-2"><div></div> <span> </span></div></div></div></div> <div class="bg-gray-800 rounded-lg p-4 border border-gray-700"><div class="flex items-center justify-between"><div class="flex items-center justify-start flex-1 space-x-6"><span class="text-sm text-gray-400">Contactors:</span> <!></div> <div class="flex items-center space-x-2"><span class="text-sm text-gray-400">Supply:</span> <div></div></div></div></div> <div class="bg-gray-800 rounded-lg p-4 border border-gray-700"><div class="flex items-center justify-between mb-3"><span class="text-sm text-gray-400">BMS Status:</span></div> <div class="grid grid-cols-4 md:grid-cols-5 lg:grid-cols-7 gap-4"></div></div> <div class="bg-gray-800 rounded-lg p-4 border border-gray-700"><div class="mb-4"><h3 class="text-lg font-semibold">Battery Pack Details</h3></div> <div class="space-y-4"></div> <div class="mt-4 flex justify-center space-x-6 text-xs"><div class="flex items-center space-x-2"><div class="w-3 h-3 bg-green-500 rounded"></div> <span>Normal</span></div> <div class="flex items-center space-x-2"><div class="w-3 h-3 bg-yellow-500 rounded"></div> <span>Warning</span></div> <div class="flex items-center space-x-2"><div class="w-3 h-3 bg-red-500 rounded"></div> <span>Critical</span></div></div></div><div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-4 mt-6"><div class="metric-card svelte-23yu5p"><div id="cmu1-max" class="metric-value text-red-400 svelte-23yu5p">-</div><div class="metric-label svelte-23yu5p text-xs">CMU 1 Max</div></div><div class="metric-card svelte-23yu5p"><div id="cmu1-min" class="metric-value text-cyan-400 svelte-23yu5p">-</div><div class="metric-label svelte-23yu5p text-xs">CMU 1 Min</div></div><div class="metric-card svelte-23yu5p"><div id="cmu2-max" class="metric-value text-red-400 svelte-23yu5p">-</div><div class="metric-label svelte-23yu5p text-xs">CMU 2 Max</div></div><div class="metric-card svelte-23yu5p"><div id="cmu2-min" class="metric-value text-cyan-400 svelte-23yu5p">-</div><div class="metric-label svelte-23yu5p text-xs">CMU 2 Min</div></div><div class="metric-card svelte-23yu5p"><div id="cmu3-max" class="metric-value text-red-400 svelte-23yu5p">-</div><div class="metric-label svelte-23yu5p text-xs">CMU 3 Max</div></div><div class="metric-card svelte-23yu5p"><div id="cmu3-min" class="metric-value text-cyan-400 svelte-23yu5p">-</div><div class="metric-label svelte-23yu5p text-xs">CMU 3 Min</div></div><div class="metric-card svelte-23yu5p"><div id="cmu4-max" class="metric-value text-red-400 svelte-23yu5p">-</div><div class="metric-label svelte-23yu5p text-xs">CMU 4 Max</div></div><div class="metric-card svelte-23yu5p"><div id="cmu4-min" class="metric-value text-cyan-400 svelte-23yu5p">-</div><div class="metric-label svelte-23yu5p text-xs">CMU 4 Min</div></div></div><div class="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6"><div class="plot-container svelte-23yu5p"><canvas id="cmu1-chart" class="w-full h-80"></canvas></div><div class="plot-container svelte-23yu5p"><canvas id="cmu2-chart" class="w-full h-80"></canvas></div><div class="plot-container svelte-23yu5p"><canvas id="cmu3-chart" class="w-full h-80"></canvas></div><div class="plot-container svelte-23yu5p"><canvas id="cmu4-chart" class="w-full h-80"></canvas></div></div></div>');
function kt(Ve, Pe) {
    tt(Pe, !1);
    const [Me,je] = vt()
      , i = () => lt(it, "$globalStore", Me);
    function d(r, s="", _=2) {
        return typeof r != "number" ? "N/A" : `${r.toFixed(_)} ${s}`
    }
    function Ae(r) {
        return typeof r != "number" ? "bg-gray-600" : r > 4.2 || r < 2.5 ? "bg-red-500" : r > 2.5 && r < 2.8 ? "bg-yellow-500" : "bg-green-500"
    }
    function se(r) {
        return typeof r != "number" ? "bg-gray-600" : r > 45 ? "bg-red-500" : r > 35 ? "bg-yellow-500" : "bg-green-500"
    }
    function ie(r) {
        return typeof r != "number" ? 0 : Math.min(100, Math.max(0, (r - 2.5) / (4.2 - 2.5) * 100))
    }
    function B(r) {
        const s = [{
            name: "Error",
            color: "bg-red-500",
            textColor: "text-red-400"
        }, {
            name: "Idle",
            color: "bg-gray-500",
            textColor: "text-gray-400"
        }, {
            name: "Measure",
            color: "bg-blue-500",
            textColor: "text-blue-400"
        }, {
            name: "Pre-charge",
            color: "bg-yellow-500",
            textColor: "text-yellow-400"
        }, {
            name: "Run",
            color: "bg-green-500",
            textColor: "text-green-400"
        }, {
            name: "Enable Pack",
            color: "bg-cyan-500",
            textColor: "text-cyan-400"
        }];
        return s[r] || s[0]
    }
    const Ee = [{
        key: "cell_over_voltage",
        label: "Cell Over Volt"
    }, {
        key: "cell_under_voltage",
        label: "Cell Under Volt"
    }, {
        key: "cell_over_temp",
        label: "Cell Over Temp"
    }, {
        key: "measurement_untrusted",
        label: "Measurement untrusted"
    },{
        key: "cmu_comm_timeout",
        label: "CMU COMM Timeout"
    },
    {
        key: "vehicle_comm_timeout",
        label: "Vehicle COMM Timeout"
    }, {
        key: "bms_setup_mode",
        label: "BMS Setup Mode"
    }, {
        key: "cmu_can_status",
        label: "CMU CAN Status",
    }, {
        key: "isolation_test_fail",
        label: "Isolation Test Failed"
    }, {
        key: "soc_invalid",
        label: "SoC Invalid"
    }, {
        key: "can_supply_low",
        label: "CAN Supply Low"
    }, {
        key: "contactor_not_engaged",
        label: "Contactor Not Engaged"
    }, {
        key: "extra_cell_detected",
        label: "Extra Cell Detected"
    }];
    et();
    var I = ut()
      , L = t(I)
      , W = t(L)
      , q = t(W)
      , Oe = t(q, !0);
    e(q);
    var ve = a(q, 4)
      , Te = t(ve);
    re(Te, "width: 80%"),
    e(ve),
    e(W);
    var D = a(W, 2)
      , le = t(D)
      , Ue = t(le, !0);
    e(le),
    A(2),
    e(D);
    var Y = a(D, 2)
      , de = t(Y)
      , Fe = t(de, !0);
    e(de),
    A(2),
    e(Y);
    var z = a(Y, 2)
      , ce = t(z)
      , Ne = t(ce, !0);
    e(ce),
    A(2),
    e(z);
    var ne = a(z, 2)
      , oe = t(ne)
      , Re = t(oe, !0);
    e(oe),
    A(2),
    e(ne),
    e(L);
    var G = a(L, 2)
      , H = t(G)
      , _e = t(H)
      , ge = a(t(_e), 2)
      , Be = t(ge);
    e(ge),
    e(_e),
    e(H);
    var J = a(H, 2)
      , me = t(J)
      , pe = a(t(me), 2)
      , Ie = t(pe);
    e(pe),
    e(me),
    e(J);
    var ue = a(J, 2)
      , xe = t(ue)
      , ye = a(t(xe), 2)
      , be = t(ye)
      , K = a(be, 2)
      , Le = t(K, !0);
    e(K),
    e(ye),
    e(xe),
    e(ue),
    e(G);
    var Q = a(G, 2)
      , fe = t(Q)
      , X = t(fe)
      , We = a(t(X), 2);
    N(We, 0, () => [1, 2, 3], R, (r, s) => {
        const _ = te( () => i().metric.contactor_flags[`contactor${s}_error`])
          , n = te( () => i().metric.contactor_flags[`contactor${s}_output`]);
        var c = ct()
          , o = t(c)
          , u = t(o);
        e(o);
        var x = a(o, 2)
          , h = a(t(x), 2);
        e(x);
        var y = a(x, 2)
          , w = a(t(y), 2);
        e(y);
        var M = a(y, 2);
        {
            var E = $ => {
                var O = dt();
                g($, O)
            }
            ;
            ae(M, $ => {
                s < 3 && $(E)
            }
            )
        }
        e(c),
        P( () => {
            l(u, `C${s ?? ""}`),
            m(h, 1, `w-3 h-3 rounded-full ${v(n) ? "bg-green-500" : "bg-gray-500"}`),
            m(w, 1, `w-3 h-3 rounded-full ${v(_) ? "bg-red-500" : "bg-green-500"}`)
        }
        ),
        g(r, c)
    }
    ),
    e(X);
    var he = a(X, 2)
      , qe = a(t(he), 2);
    e(he),
    e(fe),
    e(Q);
    var Z = a(Q, 2)
      , Ce = a(t(Z), 2);
    N(Ce, 5, () => Ee, R, (r, s) => {
        const _ = te( () => i().metric.bmsFlags[v(s).key]);
        var n = nt()
          , c = t(n)
          , o = a(c, 2)
          , u = t(o, !0);
        e(o),
        e(n),
        P( () => {
            m(c, 1, `w-3 h-3 rounded-full ${v(_) ? `${v(s).color || "bg-red-500"}` : "bg-gray-500"} flex-shrink-0`, "svelte-23yu5p"),
            l(u, v(s).label)
        }
        ),
        g(r, n)
    }
    ),
    e(Ce),
    e(Z);
    var we = a(Z, 2)
      , $e = a(t(we), 2);
    N($e, 5, () => i().metric.cmus, R, (r, s, _) => {
        var n = pt()
          , c = t(n)
          , o = t(c)
          , u = t(o);
        u.textContent = `CMU ${_ + 1}`;
        var x = a(u, 2)
          , h = t(x)
          , y = t(h)
          , w = a(y, 2)
          , M = t(w);
        e(w),
        e(h);
        var E = a(h, 2)
          , $ = t(E)
          , O = a($, 2)
          , De = t(O);
        e(O),
        e(E),
        e(x),
        e(o),
        e(c);
        var Ye = a(c, 2);
        {
            var ze = b => {
                var f = gt();
                N(f, 5, () => (v(s).cell_voltages||[]).slice(0, 6), R, (ee, k, ke) => {
                    var Se = rt()
                      , He = st(Se);
                    {
                        var Je = S => {
                            var C = ot()
                              , j = t(C);
                            j.textContent = `Cell ${ke}`;
                            var V = a(j, 2)
                              , T = t(V, !0);
                            e(V);
                            var U = a(V, 2)
                              , F = t(U);
                            e(U),
                            e(C),
                            P( (Qe, Xe, Ze) => {
                                l(T, Qe),
                                m(F, 1, `h-2 rounded-full ${Xe ?? ""}`, "svelte-23yu5p"),
                                re(F, `width: ${Ze ?? ""}%`)
                            }
                            , [ () => d(v(k), "V", 3), () => Ae(v(k)), () => ie(v(k))]),
                            g(S, C)
                        }
                          , Ke = S => {
                            var C = _t()
                              , j = t(C);
                            j.textContent = `Cell ${ke}`;
                            var V = a(j, 2);
                            V.textContent = "-";
                            var T = a(V, 2)
                              , U = t(T);
                            e(T),
                            e(C),
                            P(F => re(U, `width: ${F ?? ""}%`), [ () => ie(4.2)]),
                            g(S, C)
                        }
                        ;
                        ae(He, S => {
                            v(k) < 10 ? S(Je) : S(Ke, !1)
                        }
                        )
                    }
                    g(ee, Se)
                }
                ),
                e(f),
                g(b, f)
            }
              , Ge = b => {
                var f = mt();
                f.textContent = `No cell data available for Pack ${_ + 1}`,
                g(b, f)
            }
            ;
            ae(Ye, b => {
                v(s).cell_voltages && v(s).cell_voltages.length > 0 ? b(ze) : b(Ge, !1)
            }
            )
        }
        e(n),
        P((b,f,ee,k)=>{m(y,1,`w-3 h-3 rounded-full ${b??""}`,"svelte-23yu5p"),l(M,`CMU: ${f??""}`),m($,1,"hidden"),l(De,"")}
        , [ () => se(v(s).temperature), () => d(v(s).temperature, "°C", 1), () => se(v(s).cell_temperature), () => d(v(s).cell_temperature, "°C", 1)]),
        g(r, n)
    }
    ),
    e($e),
    A(2),
    e(we),
    e(I);
    let chartsInit=false;let ch1,ch2,ch3,ch4;function getChartConfig(s,v1,v2,m1,m2,_){return{type:"line",data:{labels:i().historic.Timestamps,datasets:[{label:"PCB Temp",data:v1,borderColor:m1,backgroundColor:m1+"20",borderWidth:2,fill:!1,tension:.1},{label:"Cell Temp",data:v2,borderColor:m2,backgroundColor:m2+"20",borderWidth:2,fill:!1,tension:.1}]},options:{responsive:!0,maintainAspectRatio:!1,scales:{y:{title:{display:!0,text:_,color:"#fff"},ticks:{color:"#fff"},grid:{color:"#374151"}},x:{title:{display:!0,text:"Time",color:"#fff"},ticks:{color:"#fff"},grid:{color:"#374151"}}},plugins:{legend:{labels:{color:"#fff"}},title:{display:!0,text:`${s} vs Time`,color:"#fff"}}}}}
    P( (r, s, _, n, c, o, u, x, h, y, w, M) => {
        if(!chartsInit&&document.getElementById("cmu1-chart")){chartsInit=true;ch1=new Chart(document.getElementById("cmu1-chart"),getChartConfig("CMU 1",i().historic.cmu1_temp,i().historic.cmu1_cell_temp,"#f59e0b","#ef4444","Temperature (°C)"));ch2=new Chart(document.getElementById("cmu2-chart"),getChartConfig("CMU 2",i().historic.cmu2_temp,i().historic.cmu2_cell_temp,"#f59e0b","#ef4444","Temperature (°C)"));ch3=new Chart(document.getElementById("cmu3-chart"),getChartConfig("CMU 3",i().historic.cmu3_temp,i().historic.cmu3_cell_temp,"#f59e0b","#ef4444","Temperature (°C)"));ch4=new Chart(document.getElementById("cmu4-chart"),getChartConfig("CMU 4",i().historic.cmu4_temp,i().historic.cmu4_cell_temp,"#f59e0b","#ef4444","Temperature (°C)"));}else if(chartsInit&&i().historic.Timestamps&&i().historic.Timestamps.length>0){if(ch1){ch1.data.labels=i().historic.Timestamps;ch1.data.datasets[0].data=i().historic.cmu1_temp;ch1.data.datasets[1].data=i().historic.cmu1_cell_temp;ch1.update("none")}if(ch2){ch2.data.labels=i().historic.Timestamps;ch2.data.datasets[0].data=i().historic.cmu2_temp;ch2.data.datasets[1].data=i().historic.cmu2_cell_temp;ch2.update("none")}if(ch3){ch3.data.labels=i().historic.Timestamps;ch3.data.datasets[0].data=i().historic.cmu3_temp;ch3.data.datasets[1].data=i().historic.cmu3_cell_temp;ch3.update("none")}if(ch4){ch4.data.labels=i().historic.Timestamps;ch4.data.datasets[0].data=i().historic.cmu4_temp;ch4.data.datasets[1].data=i().historic.cmu4_cell_temp;ch4.update("none")}}for(let k=1;k<=4;k++){let m=i().metric.cmus[k-1];let el1=document.getElementById(`cmu${k}-max`),el2=document.getElementById(`cmu${k}-min`);if(m&&el1&&el2){el1.innerText=Math.max(m.temperature,m.cell_temperature).toFixed(1)+" °C";el2.innerText=Math.min(m.temperature,m.cell_temperature).toFixed(1)+" °C";}}
        l(Oe, r),
        re(Te,`width: ${Math.max(0,Math.min(100,Number(i().metric.SOC_Ah)||0))}%`),
        l(Ue, s),
        l(Fe, _),
        l(Ne, n),
        l(Re, c),
        l(Be, `${o ?? ""} - ${u ?? ""}`),
        l(Ie, `${x ?? ""} - ${h ?? ""}`),
        m(be, 1, `w-3 h-3 rounded-full ${y ?? ""}`, "svelte-23yu5p"),
        m(K, 1, `text-sm ${w ?? ""}`, "svelte-23yu5p"),
        l(Le, M),
        m(qe, 1, `w-3 h-3 rounded-full ${i().metric.contactor_flags.contactor_supply ? "bg-green-500" : "bg-red-500"}`)
    }
    , [ () => d(i().metric.SOC_Ah, "%", 0), () => d(i().metric.Pack_Voltage, "V"), () => d(i().metric.Pack_Current, "A"), () => d(i().metric.battery_ranges.max_temp, "°C"), () => d(i().metric.battery_ranges.min_temp, "°C"), () => d(i().metric.battery_ranges.min_volt, "V"), () => d(i().metric.battery_ranges.max_volt, "V"), () => d(i().metric.battery_ranges.min_temp, "°C"), () => d(i().metric.battery_ranges.max_temp, "°C"), () => B(i().metric.precharge_state).color, () => B(i().metric.precharge_state).textColor, () => B(i().metric.precharge_state).name]),
    g(Ve, I),
    at(),
    je()
}
export {kt as component};
