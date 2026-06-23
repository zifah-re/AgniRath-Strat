import "../chunks/Bzak7iHL.js";
import {o as ee, s as me} from "../chunks/NmWs3qIF.js";
import {h as fe, aC as be, aD as xe, aE as _e, aF as te, Y as re, aG as q, aH as he, Z as ye, t as s, _ as $, a9 as we, a0 as k, a1 as ae, a2 as l, W as S, a6 as t, a3 as r, $ as z, a5 as b, aI as ke, aJ as Se} from "../chunks/B2hAlVi-.js";
import {i as B} from "../chunks/3d2OysfF.js";
import {e as se, i as Ee} from "../chunks/T28DYXdB.js";
import {g as E, s as L} from "../chunks/DNFaeDbv.js";
import {b as $e} from "../chunks/CqHrAp-I.js";
import {a as Ne, s as Ae} from "../chunks/DkIwFic-.js";
import {s as Le} from "../chunks/jT1E5TWv.js";
const We = Symbol("is custom element")
  , Ie = Symbol("is html");
function Te(e, a, o, f) {
    var p = Ce(e);
    fe && (p[a] = e.getAttribute(a),
    e.nodeName === "LINK") || p[a] !== (p[a] = o) && (o == null ? e.removeAttribute(a) : typeof o != "string" && Fe(e).includes(a) ? e[a] = o : e.setAttribute(a, o))
}
function Ce(e) {
    return e.__attributes ?? (e.__attributes = {
        [We]: e.nodeName.includes("-"),
        [Ie]: e.namespaceURI === be
    })
}
var X = new Map;
function Fe(e) {
    var a = X.get(e.nodeName);
    if (a)
        return a;
    X.set(e.nodeName, a = []);
    for (var o, f = e, p = Element.prototype; p !== f; ) {
        o = _e(f);
        for (var v in o)
            o[v].set && a.push(v);
        f = xe(f)
    }
    return a
}
var Me = (e, a) => E.removeNotification(s(a).id)
  , De = $('<div><div class="flex items-center justify-between"><div class="flex items-center space-x-3"><span class="text-xl"> </span> <div><div class="text-white font-medium text-sm"> </div> <div class="text-gray-400 text-xs mt-1"> </div></div></div> <button class="text-gray-400 hover:text-white ml-4 p-1" title="Dismiss">✕</button></div></div>')
  , je = () => E.clearNotifications()
  , Oe = $('<div class="fixed top-4 right-4 z-40 max-w-sm"><div class="bg-gray-700 rounded-lg p-3 text-white text-sm"><div class="flex items-center justify-between"><span> </span> <button class="text-gray-300 hover:text-white text-xs px-2 py-1 bg-gray-600 rounded">Clear All</button></div></div></div>')
  , Ue = $('<div class="fixed top-4 right-4 z-50 max-w-sm space-y-2"></div> <!>', 1);
function ze(e, a) {
    re(a, !0);
    let o = q(he([]));
    ee( () => E.notifications.subscribe(d => {
        S(o, d, !0)
    }
    ));
    function f(i) {
        setTimeout( () => {
            E.removeNotification(i.id)
        }
        , 1e4)
    }
    function p(i) {
        return i.toLocaleTimeString()
    }
    function v(i) {
        switch (i) {
        case "error":
            return "🚨";
        case "warning":
            return "⚠️";
        case "info":
            return "ℹ️";
        default:
            return "🔔"
        }
    }
    ye( () => {
        s(o).forEach(i => {
            f(i)
        }
        )
    }
    );
    var g = Ue()
      , y = we(g);
    se(y, 21, () => s(o), i => i.id, (i, d) => {
        var c = De()
          , x = r(c)
          , m = r(x)
          , _ = r(m)
          , h = r(_, !0);
        t(_);
        var w = l(_, 2)
          , A = r(w)
          , W = r(A, !0);
        t(A);
        var I = l(A, 2)
          , M = r(I, !0);
        t(I),
        t(w),
        t(m);
        var D = l(m, 2);
        D.__click = [Me, d],
        t(x),
        t(c),
        z( (H, j) => {
            L(c, 1, `bg-gray-800 border-l-4 rounded-lg shadow-lg p-4 animate-slide-in
                   ${s(d).severity === "error" ? "border-red-500" : ""}
                   ${s(d).severity === "warning" ? "border-yellow-500" : ""}
                   ${s(d).severity === "info" ? "border-blue-500" : ""}`, "svelte-1laav8v"),
            b(h, H),
            b(W, s(d).message),
            b(M, j)
        }
        , [ () => v(s(d).severity), () => p(s(d).timestamp)]),
        k(i, c)
    }
    ),
    t(y);
    var N = l(y, 2);
    {
        var F = i => {
            var d = Oe()
              , c = r(d)
              , x = r(c)
              , m = r(x)
              , _ = r(m);
            t(m);
            var h = l(m, 2);
            h.__click = [je],
            t(x),
            t(c),
            t(d),
            z( () => b(_, `${s(o).length ?? ""} active alerts`)),
            k(i, d)
        }
        ;
        B(N, i => {
            s(o).length > 3 && i(F)
        }
        )
    }
    k(e, g),
    ae()
}
te(["click"]);
const He = () => {
    const e = Le;
    return {
        page: {
            subscribe: e.page.subscribe
        },
        navigating: {
            subscribe: e.navigating.subscribe
        },
        updated: e.updated
    }
}
  , Pe = {
    subscribe(e) {
        return He().page.subscribe(e)
    }
};
async function Re(e, a) {
    try {
        document.fullscreenElement ? (await document.exitFullscreen(),
        S(a, !1)) : (await document.documentElement.requestFullscreen(),
        S(a, !0))
    } catch (o) {
        console.error("Error toggling fullscreen:", o)
    }
}
var Je = () => E.clearNotifications()
  , Ke = $('<div class="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-400"></div>')
  , qe = $('<a><span class="text-lg"> </span> <span> </span> <!></a>')
  , Be = $('<div class="absolute top-4 right-4 z-50"><div class="bg-black bg-opacity-50 text-white px-3 py-2 rounded-lg text-sm backdrop-blur-sm">Press <kbd class="bg-gray-700 px-2 py-1 rounded text-xs">ESC</kbd> to exit fullscreen</div></div>')
  , Ge = $('<div class="min-h-screen bg-gray-900 text-gray-200"><header><div class="flex items-center justify-between"><div class="flex items-center space-x-4"><img src="/logo.png" class="h-12" alt="Agnirath Logo"/> <h1 class="text-2xl font-bold text-white">Agnirath Telemetry Dashboard</h1> <div class="flex items-center space-x-2"><div></div> <span class="text-sm text-gray-300"> </span></div></div> <div class="flex items-center space-x-4"><div class="text-sm text-gray-400">Live Telemetry Dashboard</div> <button class="flex items-center space-x-2 px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors duration-200 text-sm" title="Clear All Notifications"><span class="text-lg">🔔</span> <span>Clear Alerts</span></button> <button class="flex items-center space-x-2 px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors duration-200 text-sm" title="Toggle Fullscreen (ESC to exit)"><span class="text-lg"> </span> <span> </span></button></div></div></header> <nav><div class="px-4"><div class="flex justify-between"></div></div></nav> <main><!> <!></main> <!></div>');
function st(e, a) {
    re(a, !0);
    const [o,f] = Ne()
      , p = () => Ae(Pe, "$page", o);
    let v = null, g = null, y, N = q(!1);
    function F() {
        !y || v && v.readyState === WebSocket.OPEN || (v = new WebSocket(y),
        v.onopen = () => {
            S(N, !0),
            console.log("WebSocket connected"),
            g && (clearInterval(g),
            g = null)
        }
        ,
        v.onmessage = n => {
            try {
                const u = JSON.parse(n.data);
                if (u.type == "reload") {
                    console.log("Forced reload requested by server.");
                    window.location.reload();
                    return;
                }
                u.type == "update" ? E.handleWebSocketUpdate(u) : u.type == "data" && E.handleWebSocketData(u)
            } catch (u) {
                console.error("Error parsing WebSocket message:", u)
            }
        }
        ,
        v.onclose = n => {
            S(N, !1),
            console.log("WebSocket disconnected:", n.code, n.reason),
            g || (g = setInterval( () => {
                console.log("Attempting to reconnect..."),
                F()
            }
            , 3e3))
        }
        ,
        v.onerror = n => {
            console.error("WebSocket error:", n)
        }
        )
    }
    const i = [{
        name: "Overview",
        path: "/",
        icon: "🏠"
    }, {
        name: "Battery",
        path: "/battery",
        icon: "🔋"
    }, {
        name: "Motor",
        path: "/motor",
        icon: "⚙️"
    }, {
        name: "Solar",
        path: "/solar",
        icon: "☀️"
    }, {
        name: "Strategy",
        path: "/strategy",
        icon: "📋"
    }, {
        name: "Map",
        path: "/map.html",
        icon: "🗺️"
    }, {
        name: "System Status",
        path: "/system_status",
        icon: "🚨"
    }];
    function d(n) {
        return n === "/" ? p().url.pathname === "/" : p().url.pathname.startsWith(n)
    }
    let c = q(!1), x;
    function m(n) {
        n.key === "Escape" && document.fullscreenElement && (document.exitFullscreen(),
        S(c, !1))
    }
    function _() {
        S(c, !!document.fullscreenElement)
    }
    ee( () => (y = `ws://${window.location.host}/ws/updates`,
    F(),
    document.addEventListener("keydown", m),
    document.addEventListener("fullscreenchange", _),
    () => {
        document.removeEventListener("keydown", m),
        document.removeEventListener("fullscreenchange", _),
        v && v.close(),
        g && clearInterval(g)
    }
    ));
    var h = Ge();
    ke(n => {
        Se.title = "Agnirath"
    }
    );
    var w = r(h)
      , A = r(w)
      , W = r(A)
      , I = l(r(W), 4)
      , M = r(I)
      , D = l(M, 2)
      , H = r(D, !0);
    t(D),
    t(I),
    t(W);
    var j = l(W, 2)
      , G = l(r(j), 2);
    G.__click = [Je];
    var P = l(G, 2);
    P.__click = [Re, c];
    var R = r(P)
      , ne = r(R, !0);
    t(R);
    var Y = l(R, 2)
      , oe = r(Y, !0);
    t(Y),
    t(P),
    t(j),
    t(A),
    t(w);
    var O = l(w, 2)
      , Z = r(O)
      , Q = r(Z);
    se(Q, 21, () => i, Ee, (n, u) => {
        var T = qe()
          , J = r(T)
          , de = r(J, !0);
        t(J);
        var K = l(J, 2)
          , ve = r(K, !0);
        t(K);
        var ue = l(K, 2);
        {
            var pe = C => {
                var ge = Ke();
                k(C, ge)
            }
            ;
            B(ue, C => {
                d(s(u).path) && C(pe)
            }
            )
        }
        t(T),
        z(C => {
            Te(T, "href", s(u).path),
            L(T, 1, `relative flex items-center justify-center space-x-2 px-6 py-4 text-sm font-medium transition-colors duration-200 hover:bg-gray-700 flex-1 ${C ?? ""}`),
            b(de, s(u).icon),
            b(ve, s(u).name)
        }
        , [ () => d(s(u).path) ? "text-blue-400 bg-gray-700" : "text-gray-300 hover:text-white"]),
        k(n, T)
    }
    ),
    t(Q),
    t(Z),
    t(O);
    var U = l(O, 2)
      , V = r(U);
    {
        var ie = n => {
            var u = Be();
            k(n, u)
        }
        ;
        B(V, n => {
            s(c) && n(ie)
        }
        )
    }
    var ce = l(V, 2);
    me(ce, () => a.children),
    t(U);
    var le = l(U, 2);
    ze(le, {}),
    t(h),
    $e(h, n => x = n, () => x),
    z( () => {
        L(w, 1, `bg-gray-800 border-b border-gray-700 p-4 ${s(c) ? "hidden" : "block"}`),
        L(M, 1, `w-3 h-3 rounded-full ${s(N) ? "bg-green-500" : "bg-red-500"}`),
        b(H, s(N) ? "Connected" : "Disconnected"),
        b(ne, s(c) ? "🪟" : "⛶"),
        b(oe, s(c) ? "Exit" : "Fullscreen"),
        L(O, 1, `bg-gray-800 border-b border-gray-700 ${s(c) ? "hidden" : "block"}`),
        L(U, 1, `flex-1 ${s(c) ? "min-h-screen" : ""}`)
    }
    ),
    k(e, h),
    ae(),
    f()
}
te(["click"]);
export {st as component};
