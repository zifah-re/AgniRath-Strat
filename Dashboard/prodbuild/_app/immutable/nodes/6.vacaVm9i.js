import "../chunks/Bzak7iHL.js";
import {o as _e} from "../chunks/NmWs3qIF.js";
import {Y as ye, Z as xe, _ as Se, $ as Ce, a0 as je, a1 as we, a2 as i, a3 as t, t as g, a4 as X, a5 as h, a6 as a, a7 as k} from "../chunks/B2hAlVi-.js";
import {g as ke, s as V, c as Ae} from "../chunks/DNFaeDbv.js";
import {b as f} from "../chunks/CqHrAp-I.js";
import {s as Te, a as $e} from "../chunks/DkIwFic-.js";
import {C as n, a as Le, L as Pe, P as Be, b as We, c as Me, p as Ee, d as Ie, e as Re} from "../chunks/DHTsbXwT.js";
var Ze = Se('<div class="space-y-6 p-6"><div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-8"><div><div class="flex items-center justify-between mb-2"><h3 class="text-lg font-semibold">Speed Status</h3> <div></div></div> <div class="grid grid-cols-2 gap-4"><div><div class="metric-value text-blue-400 svelte-14ry3bj"> </div> <div class="metric-label svelte-14ry3bj">Current Speed</div></div> <div><div class="metric-value text-gray-400 svelte-14ry3bj"> </div> <div class="metric-label svelte-14ry3bj">Predicted Speed</div></div></div> <div class="mt-2 text-sm"><span class="text-gray-400">Margin:</span> <span> </span></div></div> <div class="metric-card svelte-14ry3bj"><div class="metric-value text-purple-400 svelte-14ry3bj"> </div> <div class="metric-label svelte-14ry3bj">Current Acceleration</div></div> <div class="metric-card svelte-14ry3bj"><div class="metric-value text-green-400 svelte-14ry3bj"> </div> <div class="metric-label svelte-14ry3bj">Battery Level</div></div> <div class="metric-card svelte-14ry3bj"><div class="metric-value text-red-400 svelte-14ry3bj"> </div> <div class="metric-label svelte-14ry3bj">ETA</div></div></div> <div class="grid grid-cols-1 xl:grid-cols-2 gap-6"><div class="plot-container xl:col-span-2 svelte-14ry3bj"><canvas class="w-full h-80"></canvas></div> <div class="plot-container svelte-14ry3bj"><canvas class="w-full h-80"></canvas></div> <div class="plot-container svelte-14ry3bj"><canvas class="w-full h-80"></canvas></div> <div style="display:none;" class="plot-container svelte-14ry3bj"><canvas class="w-full h-80"></canvas></div> <div style="display:none;" class="plot-container svelte-14ry3bj"><canvas class="w-full h-80"></canvas></div> <div style="display:none;" class="plot-container xl:col-span-2 svelte-14ry3bj"><canvas class="w-full h-80"></canvas></div><div class="plot-container svelte-14ry3bj"><canvas id="grad-dist" class="w-full h-80"></canvas></div><div class="plot-container svelte-14ry3bj"><canvas id="alt-dist" class="w-full h-80"></canvas></div></div> <div class="metric-card svelte-14ry3bj p-4 mt-4"><div class="flex items-center justify-between mb-3"><h3 class="text-lg font-semibold">Offline Model Strategy</h3> <span class="text-sm text-gray-400">Upload &rarr; select day &rarr; apply</span></div> <div class="grid grid-cols-1 md:grid-cols-4 gap-3 items-end"><div><label class="text-sm text-gray-400 block mb-1">Offline model output (.json)</label> <input id="strategy-file-input" type="file" accept=".json,application/json" class="w-full text-sm text-gray-300 bg-gray-800 rounded border border-gray-600 p-1.5" /></div> <div><label class="text-sm text-gray-400 block mb-1">Strategy</label> <select id="strategy-variant-select" class="w-full bg-gray-800 text-white rounded px-2 py-1.5 border border-gray-600"><option value="">No strategies</option></select></div> <div><label class="text-sm text-gray-400 block mb-1">Day</label> <select id="strategy-day-select" class="w-full bg-gray-800 text-white rounded px-2 py-1.5 border border-gray-600"></select></div> <button id="strategy-apply-btn" disabled class="bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded px-4 py-1.5 font-medium">Apply to Dashboard</button></div> <p id="strategy-status" class="mt-2 text-sm hidden"></p></div></div>');
function De(ee, te) {
    ye(te, !0);
    const [ae,se] = $e()
      , e = () => Te(ke, "$globalStore", ae);
    n.register(Le, Pe, Be, We, Me, Ee, Ie, Re);
    let _, y, x, S, C, j, r, l, o, d, c, v, cleanupOfflineStrategyWidget;
    function m(s, p="", m=1) {
        return typeof s != "number" ? (Array.isArray(s) ? s.map(x => Math.round(x * 10**m) / 10**m) : "N/A") : `${s.toFixed(m)} ${p}`
    }
    const zipToCoords = (timestamps, speeds) => {
        if (!timestamps || !speeds || !Array.isArray(timestamps)) return [];
        return timestamps.map((dateTimeStr, index) => ({
            x: dateTimeStr, // Converts to absolute seconds number
            y: speeds[index] || 0
        }));
    };
    const zipToCoordsMPC = (dataPairs) => {
        // Safety check: ensure we received a valid array
        if (!dataPairs || !Array.isArray(dataPairs)) return [];
        
        // Each 'pair' is an array [timeStr, speedValue], e.g., ["11:41:35", 45]
        return dataPairs.map((pair) => ({
            x: pair[0], // Convert your time string token to numeric seconds
            y: pair[1] || 0         // Grab the speed value
        }));
    };
    function time_format(s) {
        var hrs=Math.floor(s/3600);
        var mins=Math.floor((s%3600)/60);
        if (hrs === 0) {
            return `${mins} min`;
        } else {
            var hrLabel = hrs > 1 ? "hrs" : "hr";
            return `${hrs} ${hrLabel} ${mins} min`;
        }
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
        const historicalData = e().historic;
        return {
            type: "line",
            data: {
                datasets: [{
                    label: "Speed",
                    data: zipToCoords(historicalData.Time_seconds, historicalData.Speed),
                    borderColor: "#3b82f6",
                    backgroundColor: "#3b82f620",
                    borderWidth: 2,
                    fill: !1,
                    tension: .1,
                    order:2
                }, {
                    label: "Offline model speed",
                    data: zipToCoordsMPC(e().profile.TargetProfile),
                    borderColor: "#ef4444",
                    backgroundColor: "#ef444420",
                    borderWidth: 2,
                    fill: !1,
                    tension: .1,
                    order:3
                }, {
                    label: "Real time model speed",
                    data: zipToCoordsMPC(e().profile.MPCProfile),
                    borderColor: "#d3d017",
                    backgroundColor: "#ef444420",
                    borderWidth: 2,
                    fill: !1,
                    tension: .1,
                    order:1
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
                        type: "linear",
                        min:e().historic.Time_seconds.at(-1)-120,
                        max:e().historic.Time_seconds.at(-1)+900,
                        title: {
                            display: !0,
                            text: "Time",
                            color: "#fff"
                        },
                        ticks: { 
                            color: "#fff",
                            // This turns the numeric seconds back into clean "HH:MM:SS" text for display
                            callback: function(value) {
                                // Multiply by 1000 to convert back to milliseconds for JS Date
                                const dateObj = new Date(value * 1000);
                                const hrs = dateObj.getHours().toString().padStart(2, '0');
                                const mins = dateObj.getMinutes().toString().padStart(2, '0');
                                const secs = dateObj.getSeconds().toString().padStart(2, '0');
                                return `${hrs}:${mins}:${secs}`;
                            }
                        },
                        grid: {
                            color: "#374151"
                        }
                    }
                },
                plugins: {
                    zoom: {
                        pan: {
                            enabled: true,
                            mode: 'x', // Allow scrolling left and right along the x-axis
                        },
                        zoom: {
                            wheel: { enabled: true }, // Scroll to zoom in/out
                            mode: 'x'
                        }
                    },
                    legend: {
                        labels: {
                            color: "#fff"
                        }
                    },
                    title: {
                        display: !0,
                        text: "Speed & Speed 2 vs Time",
                        color: "#fff"
                    },
                    tooltip: {
                        callbacks: {
                            title: function(value) {
                                // Multiply by 1000 to convert back to milliseconds for JS Date
                                const val =value[0].parsed.x;
                                const dateObj = new Date(val * 1000);
                                const hrs = dateObj.getHours().toString().padStart(2, '0');
                                const mins = dateObj.getMinutes().toString().padStart(2, '0');
                                const secs = dateObj.getSeconds().toString().padStart(2, '0');
                                return `${hrs}:${mins}:${secs}`;
                            }
                        }
                    }
                }
            }
        }
    }
    function re() {
        // 1. Existing live telemetry charts update logic (r, l, o updates here...)
        if (e().historic?.Timestamps?.length > 0) {
            r && (r.data.datasets[0].data = zipToCoords(e().historic.Time_seconds, e().historic.Speed), r.data.datasets[1].data= zipToCoordsMPC(e().profile.TargetProfile),r.data.datasets[2].data= zipToCoordsMPC(e().profile.MPCProfile), r.options.scales.x.min=e().historic.Time_seconds.at(-1)-120,r.options.scales.x.max=e().historic.Time_seconds.at(-1)+900, r.update("none"));
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
    // 🔌 Offline Model Strategy widget — wired imperatively (like grad-dist/alt-dist
    // charts below) since this markup lives outside Svelte's tracked template output.
    // Mirrors StrategyPush.svelte's loadOptions/uploadFile/applyStrategy logic 1:1.
    function initOfflineStrategyWidget() {
        const fileInput = document.getElementById("strategy-file-input");
        const variantSelect = document.getElementById("strategy-variant-select");
        const daySelect = document.getElementById("strategy-day-select");
        const applyBtn = document.getElementById("strategy-apply-btn");
        const statusEl = document.getElementById("strategy-status");
        if (!fileInput || !variantSelect || !daySelect || !applyBtn || !statusEl) return () => {};

        let variants = [];
        let busy = false;

        function setStatus(msg, isError) {
            statusEl.textContent = msg || "";
            statusEl.classList.toggle("hidden", !msg);
            statusEl.classList.toggle("text-red-400", !!isError);
            statusEl.classList.toggle("text-green-400", !isError && !!msg);
        }

        function refreshApplyDisabled() {
            applyBtn.disabled = busy || !variantSelect.value || daySelect.value === "";
        }

        function setBusy(b) {
            busy = b;
            applyBtn.textContent = b ? "Working..." : "Apply to Dashboard";
            refreshApplyDisabled();
        }

        function renderDays(variant) {
            const v = variants.find(x => x.variant === variant);
            const days = v?.days ?? [];
            daySelect.innerHTML = "";
            days.forEach(d => {
                const opt = document.createElement("option");
                opt.value = String(d.day);
                opt.textContent = `Day ${d.day} — ${d.route ?? "?"}`;
                daySelect.appendChild(opt);
            });
            refreshApplyDisabled();
        }

        function renderVariants(preferredVariant) {
            variantSelect.innerHTML = "";
            if (variants.length === 0) {
                const opt = document.createElement("option");
                opt.value = "";
                opt.textContent = "No strategies";
                variantSelect.appendChild(opt);
            } else {
                variants.forEach(v => {
                    const opt = document.createElement("option");
                    opt.value = v.variant;
                    opt.textContent = v.variant;
                    variantSelect.appendChild(opt);
                });
            }
            const wanted = preferredVariant && variants.some(v => v.variant === preferredVariant)
                ? preferredVariant
                : variantSelect.value && variants.some(v => v.variant === variantSelect.value)
                    ? variantSelect.value
                    : (variants[0]?.variant ?? "");
            variantSelect.value = wanted;
            renderDays(wanted);
        }

        async function loadOptions(preferredVariant) {
            try {
                const res = await fetch("/api/strategy/options", { cache: "no-store" });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || "Could not load strategy options");
                variants = data.variants ?? [];
                renderVariants(preferredVariant || null);
            } catch (e) {
                setStatus(e.message || "Could not load strategy options", true);
            }
        }

        async function uploadFile() {
            const file = fileInput.files?.[0];
            if (!file) return;
            setBusy(true); setStatus(`Uploading ${file.name}...`, false);
            try {
                const body = new FormData();
                body.append("file", file);
                const res = await fetch("/api/strategy/upload", { method: "POST", body });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || "Upload failed");
                await loadOptions(data.variant);
                setStatus(`Uploaded ${data.filename}. Select a day and click Apply.`, false);
            } catch (e) {
                setStatus(e.message || "Upload failed", true);
            } finally {
                setBusy(false);
                fileInput.value = "";
            }
        }

        async function applyStrategy() {
            const variant = variantSelect.value;
            const day = daySelect.value;
            if (!variant || day === "") return;
            setBusy(true); setStatus("Applying offline model...", false);
            try {
                const res = await fetch("/api/strategy/push", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ variant, day: Number(day) })
                });
                const data = await res.json();
                if (!res.ok) throw new Error(data.detail || "Push failed");
                setStatus(`Applied ${variant}, Day ${day} — ${data.points} points.`, false);
            } catch (e) {
                setStatus(e.message || "Push failed", true);
            } finally {
                setBusy(false);
            }
        }

        const onVariantChange = () => renderDays(variantSelect.value);
        const onDayChange = () => refreshApplyDisabled();

        fileInput.addEventListener("change", uploadFile);
        variantSelect.addEventListener("change", onVariantChange);
        daySelect.addEventListener("change", onDayChange);
        applyBtn.addEventListener("click", applyStrategy);

        loadOptions();

        return () => {
            fileInput.removeEventListener("change", uploadFile);
            variantSelect.removeEventListener("change", onVariantChange);
            daySelect.removeEventListener("change", onDayChange);
            applyBtn.removeEventListener("click", applyStrategy);
        };
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

        cleanupOfflineStrategyWidget = initOfflineStrategyWidget();

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
        window.chartAD && window.chartAD.destroy(),
        cleanupOfflineStrategyWidget && cleanupOfflineStrategyWidget()
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
    Ce( (s, b, p, N, O,etaVal) => {
        V(w, 1, `metric-card col-span-1 md:col-span-2 ${g(A) === "ok" ? "status-ok" : "status-error"} border-2`, "svelte-14ry3bj"),
        V(le, 1, `w-4 h-4 rounded-full ${g(A) === "ok" ? "bg-green-500" : "bg-red-500"}`),
        h(oe, s),
        h(de, b),
        V(W, 1, Ae(g(A) === "ok" ? "text-green-400" : "text-red-400")),
        h(ce, p),
        h(ve, N),
        h(ne, O)
        if (ne && ne.parentNode) {
            // Find the 4th metric card's value slot relative to the 3rd one (ne)
            const etaNode = ne.parentNode.parentNode.parentNode.querySelector('.metric-card:nth-child(4) .metric-value');
            if (etaNode) etaNode.textContent = etaVal;
        }
    }
    , [ () => m(e().metric.Speed, "km/h"), () => m(e().metric.predicted, "km/h"), () => m(g(Y), "km/h"), () => m(e().historic.Acceleration[-1] || 0, "m/s²"), () => m(e().metric.SOC_Ah, "%", 0), () => time_format(e().metric.ETA)]),
    je(ee, T),
    we(),
    se()
}
export {De as component};