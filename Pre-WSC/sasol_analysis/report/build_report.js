// Final report (ID4100) in the layout of the midterm progress update. Run: node build_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, AlignmentType,
  LevelFormat, WidthType, ShadingType, BorderStyle, Footer, PageNumber, HeadingLevel,
} = require("docx");

const FIG = path.join(__dirname, "fig");
const BODY = "Aptos", HEAD = "Aptos Display";
const W = 9792; // content width in DXA (Letter, 0.85 in side margins as in the midterm)

// ---------- helpers ----------
function runs(text, extra = {}) {
  // **bold** markup; x_{sub} for subscripts; 1-10 -> 1–10, -37 -> −37
  text = text.replace(/(\d)-(\d)/g, "$1\u2013$2").replace(/(^|[\s(])-(\d)/g, "$1\u2212$2");
  const out = [];
  text.split(/(\*\*[^*]+\*\*)/).forEach((part) => {
    if (!part) return;
    const bold = part.startsWith("**");
    const t = bold ? part.slice(2, -2) : part;
    t.split(/(_\{[^}]+\})/).forEach((p) => {
      if (!p) return;
      if (p.startsWith("_{")) out.push(new TextRun({ text: p.slice(2, -1), subScript: true, bold, ...extra }));
      else out.push(new TextRun({ text: p, bold, ...extra }));
    });
  });
  return out;
}
const P = (t, o = {}) => new Paragraph({ spacing: { after: 100 }, children: runs(t), ...o });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)] });
const B = (t) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 50 }, children: runs(t) });
const N = (t) => new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { after: 50 }, children: runs(t) });
const EQ = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 120 }, children: runs(t.replace(/ - /g, " \u2212 "), { italics: true }) });

function figure(file, caption, widthIn) {
  const buf = fs.readFileSync(path.join(FIG, file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const wpx = Math.round(widthIn * 96);
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 40 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: wpx, height: Math.round(wpx * h / w) },
        altText: { title: caption, description: caption, name: file } })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
      children: [new TextRun({ text: caption, italics: true, color: "404040", size: 18 })] }),
  ];
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(widths, rows, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: rows.map((r, i) => new TableRow({
      tableHeader: i === 0, cantSplit: true,
      children: r.map((c, j) => new TableCell({
        borders, width: { size: widths[j], type: WidthType.DXA },
        shading: i === 0 ? { fill: "DCE6F1", type: ShadingType.CLEAR, color: "auto" } : undefined,
        margins: { top: 50, bottom: 50, left: 100, right: 100 },
        children: [new Paragraph({ spacing: { after: 0 }, children: runs(i === 0 ? `**${c}**` : c, { size: 19 }) })],
      })),
    })),
  });
}
const gap = () => new Paragraph({ spacing: { after: 60 }, children: [] });

// ---------- content ----------
const cover = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 80 },
    children: [new TextRun({ text: "Physics-Based Energy and Race Strategy Optimisation", bold: true, size: 30 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: [new TextRun({ text: "for the Sasol Solar Challenge 2026", bold: true, size: 30 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
    children: [new TextRun({ text: "Final Project Report", size: 24 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: [new TextRun({ text: "Name: Hafiz Rahman Elikkottil", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: [new TextRun({ text: "Roll No.: EE24B104", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: [new TextRun({ text: "B. Tech Electrical Engineering, IIT Madras", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 160 },
    children: [new TextRun({ text: "Faculty Mentor: Prof. Mayank Mittal", size: 22 })] }),
];

const s1 = [
  H1("1. Problem Re-Cap & Opportunity"),
  P("Sasol 2026 ran over 8 days across South Africa. Each day the car drives Stage 1 to a control stop, then any number of optional loops, then Stage 2 to the finish. The team with the most km wins. The car carries about 3.5 kWh of battery and gets at most about 1.4 kW from the sun, so it is energy-limited. The strategy has to decide how many loops to drive each day and how fast to drive on every part of the route, without running the battery flat or missing the cut-off."),
  P("Two gaps existed. The team had no tool that plans all 8 days together, so energy could not be traded between days. The car values used for planning (drag, rolling resistance, drivetrain and array efficiency) had also never been checked against what the car did in a real race."),
  P("The opportunity was to build a physics-based multi-day optimiser for the race, use it at Sasol 2026, and then use the car's own race telemetry to test the model and correct it for WSC 2027."),
];

const s2 = [
  H1("2. Objective & Result"),
  H2("The Objective"),
  P("From the proposal: develop, validate and refine a physics-based optimisation framework that generates race strategies for the AgniRath car at Sasol 2026. In measurable terms:"),
  B("plan all 8 days under the race regulations, fast enough to re-run during the race;"),
  B("compare the plan with the race telemetry and find the main sources of model error;"),
  B("refine the model and show the improvement against the Sasol data;"),
  B("test how sensitive the strategy is to each car parameter."),
  H2("The Result"),
  P("All four were met. The table gives the evidence."),
  table([2600, 5592, 1600], [
    ["Objective", "Result", "Met?"],
    ["8-day plan under the rules", "3202 km, 43 loops, no trailering, every day inside the cut-off; about 9 min to solve. Deployed at Sasol 2026. The same solver re-plans the rest of a day from the current state in about 45 s on a laptop.", "Yes"],
    ["Plan vs actual", "3202 km planned, 1192.6 km driven, split by day and cause. The changed Day 2 and Day 3 routes were worth +56 km and -37 km.", "Yes"],
    ["Find the model error", "The model under-predicted motor energy by a third. Drivetrain efficiency was 0.76, not the assumed 0.95. The array gave 17 % while driving, not 24.5 %, because the canopy shades the front strings.", "Yes"],
    ["Refine, with before/after", "Battery energy used on the healthy days: 7.44 kWh measured. The original model predicted a small net gain (0.07 kWh); the refined model predicts 6.09 kWh used. Per-stage error fell from 626 Wh to 188 Wh.", "Yes"],
    ["Sensitivity", "A 10 % change in drivetrain efficiency changes the energy used by 9-11 %, more than any other parameter.", "Yes"],
    ["LSTM for SOC (mentor's remark)", "Trained and tested on the race data. 4.6 % SOC error, no drift and no start SOC needed, but not yet better than a voltage lookup (3.7 %).", "Partly"],
  ]),
  gap(),
  P("One point from the midterm needs correcting. The midterm put the whole shortfall down to failures on the car. The validation shows the plan itself was also optimistic. On Day 2, about half of the gap between plan and actual comes from the model and half from the failures."),
];

const s3 = [
  H1("3. Project Description"),
  H2("The Strategy Model"),
  P("The model takes four inputs: the car config, the race rules config, the route data and the Solcast solar and wind forecast (Figure 1)."),
  ...figure("architecture.png", "Figure 1. Model architecture.", 5.3),
  P("A force balance turns speed, gradient, wind and acceleration into battery power:"),
  EQ("P_{net} = A η_{array} G - (F_{drag} + F_{roll} + F_{grade} + F_{acc}) v / η_{motor} - P_{idle}"),
  P("A simulator steps through each day on a 200 m grid and tracks time, SOC, stops and solar charging. The optimiser has three tiers (Figure 2):"),
  B("**Tier 1:** a coarse plan with one speed per stage gives a first guess of the start SOC for each day."),
  B("**Tier 2:** for every day and loop count, the single-day solver (genetic algorithm, then SLSQP) is run at a few start SOCs. This gives a per-day relation from start SOC to end SOC and finish time."),
  B("**Tier 3:** dynamic programming over the 8 days on those relations picks the loops per day and the SOC carried between days. Tier 2 is re-run around the new SOC values until they stop moving."),
  ...figure("solver_flow.png", "Figure 2. Multi-day solver flow.", 5.5),
  P("During the race, the same single-day solver re-plans the rest of a day from the current SOC, clock time, distance done and loops left, and gives new speed targets, loops, finish time and end SOC."),
  H2("The Post-Race Validation"),
  P("After the race a separate set of tools uses the car's telemetry to test the model. They match each log packet to the route, find the car's real parameters, check the solar array string by string and compare the battery energy the model predicts with what was measured. The logs had no GPS, altitude, accelerometer or working SOC, so all of this is built from speed, bus power, cell voltages, throttle, brake and MPPT readings."),
  H2("What Is Different"),
  B("**Three-tier decomposition.** Exact single-day solves sit inside an 8-day search, which would be far too slow to run directly. Splitting it into tiers makes a speed plan for every 10 km of an 8-day race solvable in about 9 minutes."),
  B("**Live re-planning.** The same single-day solver re-plans the rest of a day from the current state when the race does not go to plan."),
  B("**The model is tested against the race.** The car's real drivetrain efficiency, road load and array output were found from the race logs, with no GPS. Climbs were used to separate tyre losses from drivetrain losses."),
  B("**Shading found from sun geometry.** The loss on each solar string was matched to the sun's direction relative to the car, which showed which front string sits on which side."),
  B("**Reusable.** Car, rules and route are separate inputs, so the model carries over to WSC 2027."),
];

const s4 = [
  H1("4. Methodology & Execution"),
  H2("Weeks 1-10: Building and Deploying the Model"),
  P("The physics model, the route, solar and wind pipeline and the single-day and multi-day optimisers were built and tuned. The race regulations were written as constraints. Car parameters came from track testing in India. The pre-race strategy (3202 km) was generated and used at Sasol 2026. This was covered in the midterm."),
  H2("Weeks 11-16: Testing the Model Against the Race"),
  P("**Plan vs actual.** The Sasol logs were cleaned and split by day and stage. Distance comes from integrating speed. Days 2 and 3 were planned on blind routes, so both days were re-solved on the planned and on the issued routes with the same code and the same solar settings as the deployed plan. Figure 3 splits the 2010 km shortfall into route changes and day-by-day losses."),
  ...figure("waterfall.png", "Figure 3. From the deployed plan to the distance driven.", 6.4),
  P("**Validation.** Running the model on the speeds the car actually drove, it predicted only 0.67 of the measured motor energy (13.3 kWh against 19.8 kWh over nine stages)."),
  P("**Diagnosis.** Without GPS, the position from integrated speed drifted away from the route, so the gradient read from the route was often wrong. Each kilometre of log was shifted along the route until the bus power best matched the grade pattern. This raised the correlation between power and grade from 0.03-0.36 to 0.40-0.84 on 13 of 14 stages."),
  P("Packet-by-packet fits still gave unstable values, so the parameters were found from energy over 2 km stretches of steady driving (162 stretches on Days 1, 2, 3, 6 and 7, the days the car had no mechanical fault). Over each stretch:"),
  EQ("E_{bus} η = C_{rr} m g D + C_{d}A Σ ½ρ v_{air}² v Δt + m g Δh + ΔKE"),
  P("Air density comes from altitude and temperature, wind from Solcast along the car's heading and the climb from the route altitude at the two ends. Climbs separate the causes: tyres cannot make a climb cost more, but a drivetrain loss does."),
  table([3000, 2200, 2400, 2192], [
    ["Parameter", "Track testing", "Sasol data", "Range (5-95 %)"],
    ["Rolling resistance, Crr", "0.007", "0.0082", "0.0075-0.0107"],
    ["Drag area, CdA", "0.16 m²", "0.177 m²", "0.13-0.19 m²"],
    ["Drivetrain efficiency", "0.95 (assumed)", "0.76", "0.74-0.80"],
  ]),
  gap(),
  P("The road load from track testing holds up. The drivetrain efficiency does not: it was an assumed value, and the race data puts it at 0.76 whichever day is left out (Figure 4). On the 101 flat stretches (average 61 km/h) the bus drew 1100 W, against the 880 W estimate for 60 km/h. Thinner air on the highveld and the measured wind would have lowered that figure, not raised it. The logs also showed that regen only happens when cruise control holds speed downhill. Coasting and braking recover nothing."),
  ...figure("climb_eta.png", "Figure 4. Drivetrain efficiency needed to match the measured energy, by road grade.", 6.4),
  P("**Solar array.** Array power was compared with Solcast irradiance at the car's position. On clear days the whole array gave 15-17 % while driving. The best single string gave about 23 % and the array tilted at the sun at stops gave about 26 %. The loss is shading. When the sun is on the car's right, string D drops to 12-31 % of the best string; when it is on the left, string C drops to 29-57 % (Figure 5). So D is the front-left string and C the front-right, both shaded by the canopy."),
  ...figure("shading.png", "Figure 5. Output of each solar string against the sun's direction relative to the car.", 6.4),
  P("**Refinement.** The model core was left unchanged and was run with the refined values (drivetrain efficiency 0.77, Crr 0.0103 and CdA 0.126 as fitted under the model's own air assumptions, array 17 %, idle 9 W). On the healthy days the car used 7.44 kWh of battery energy. The original model predicted a small net gain of 0.07 kWh; the refined model predicts 6.09 kWh (Figure 6). On the days mechanical C broke (5 and 8) the car still used 12-14 % more than the refined model."),
  ...figure("energy_balance.png", "Figure 6. Battery energy used per stage on the healthy days: measured, original and refined model.", 6.4),
  P("Re-planning Day 2 on the issued route from 90 % SOC:"),
  table([3800, 2400, 1800, 1792], [
    ["Parameters", "Plan", "Speed", "vs 302 km driven"],
    ["Original (track values)", "513 km, 6 loops", "about 67 km/h", "+211 km"],
    ["Refined", "407 km, 3 loops", "55-57 km/h", "+105 km"],
    ["Refined, no canopy shading", "478 km, 5 loops", "about 67 km/h", "+176 km"],
  ]),
  gap(),
  P("Two optimiser fixes were made after the race: a day plan is now judged on its final simulated profile rather than on SLSQP\u2019s convergence flag, which had discarded up to half of the legal day plans, and the multi-day search re-centres instead of stopping after an infeasible first pass. With the same pre-race inputs the fixed optimiser converges and plans 3247 km, against the 3202 km deployed. Re-solved from Day 2 at the measured start SOC of 87 % on the routes actually issued, it plans 3130 km."),
  P("**Sensitivity.** Each parameter was moved by 10 % on the actual Day 1-3 driving. Drivetrain efficiency changed the energy used by 9-11 %, array output by 7 %, mass by 6 %, Crr by 4.6 % and CdA by 4.0 %. In the planner, 10 % lower drivetrain efficiency made Day 4 finish 29 minutes later. Regen efficiency hardly mattered."),
  P("**LSTM for SOC.** The mentor asked for an LSTM to estimate SOC in place of coulomb counting. The logger's SOC was stuck and pack current read zero, so a reference SOC was built from the BMS cell voltages and the measured currents (capacity 29.3 Ah, 41 mV fit). An LSTM (one layer, 32 units, three networks averaged) reads the last 10 minutes of voltage, current and temperature and outputs SOC. It needs no start SOC and does not drift:"),
  table([4600, 2000, 3192], [
    ["Estimator (held-out days)", "SOC error (RMSE)", "Needs start SOC / drifts"],
    ["Voltage lookup", "3.7 %", "no / no"],
    ["Coulomb counting", "4.3 %", "yes / yes, about 18 % over a 9 h day"],
    ["LSTM", "4.6 %", "no / no"],
  ]),
  gap(),
  P("The LSTM does not yet beat the voltage lookup, because the reference itself is built from the same voltage curve and there are only six days to train on. A working pack current sensor and logged full charges would give a true reference."),
  P("**Software.** The model ran slowly on the team laptop. The cause was a rolling power window in the simulator that re-summed its whole history at every step. Fixing it made solves 1.9 times faster with identical results."),
  H2("Changes from the Weekly Plan"),
  B("The model's parameters were not changed in code. The refined values are applied when the model is run, so the core stays the same."),
  B("The SOC comparison was done as a battery energy balance, because SOC was not logged."),
  B("The LSTM, listed for WSC 2027 at the midterm, was done this semester. Coulomb counting in the BMS and the weather station remain future work."),
  B("Diagnosis took longer than planned. The first packet-by-packet fits gave unphysical values and were replaced by route alignment and the energy method."),
];

const s5 = [
  H1("5. Deliverables & Outputs"),
  H2("What Was Produced"),
  B("**Strategy model:** the three-tier optimiser, the deployed Sasol 2026 plan and dashboard, the live re-plan, an as-issued route mode for re-solves and a demo script for running it live."),
  B("**Post-race analysis tools:** telemetry cleaning, plan vs actual, route alignment, parameter identification, solar and shading checks, battery energy balance and sensitivity, each a script with its own outputs."),
  B("**LSTM SOC estimator:** dataset builder, reference fit, training and leave-one-day-out testing."),
  B("**Findings for WSC 2027:**"),
  table([4300, 5492], [
    ["Finding", "Action"],
    ["Drivetrain efficiency is 0.76, not 0.95", "Measure motor and controller efficiency on a bench; plan with the measured value"],
    ["Canopy shades the front strings (about 70 km on Day 2)", "Rework canopy or string layout; plan with 17 % while driving"],
    ["Position, altitude and battery state had to be rebuilt from the logs", "Log GPS position and altitude, fix the pack-current sensor (it read zero) and the BMS SOC (it stayed stuck); add a gradient profiler, as the mentor suggested"],
    ["One array efficiency hides string-by-string losses", "Build a per-MPPT solar model from each string's measured efficiency, in place of one array value and full car and sun-angle geometry"],
    ["Wind and irradiance are modelled, not measured", "Build the weather station"],
  ]),
  gap(),
  H2("Skills Gained"),
  B("Breaking a large optimisation problem into tiers that can be solved in practical time."),
  B("Combining global search (GA) with local gradient methods (SLSQP) and dynamic programming."),
  B("Building a physics model of a real car and finding where it stops matching reality."),
  B("System identification from messy field data: aligning data without GPS, robust regression and testing on held-out days."),
  B("Solar geometry and array analysis from telemetry."),
  B("Training and testing an LSTM on time-series data."),
  B("Profiling and fixing performance problems in a large Python code base."),
];

const s6 = [
  H1("6. Resources Utilized"),
  H2("Software and Data"),
  B("Python: NumPy, SciPy, pandas, pvlib, matplotlib, JAX; Git and GitHub."),
  B("Sasol Solar Challenge 2026 regulations and official route files."),
  B("Solcast race-week forecasts and estimated actuals."),
  B("AgniRath Sasol 2026 telemetry, track-testing data from India and the team debrief."),
  H2("Literature"),
  N("D. Kraft, A software package for sequential quadratic programming, DFVLR-FB 88-28, 1988."),
  N("R. Bellman, Dynamic Programming, Princeton University Press, 1957."),
  N("P. J. Huber, Robust estimation of a location parameter, Annals of Mathematical Statistics 35(1), 1964."),
  N("W. F. Holmgren, C. W. Hansen and M. A. Mikofski, pvlib python: a python package for modeling solar energy systems, Journal of Open Source Software 3(29), 2018."),
  N("E. Chemali, P. J. Kollmeyer, M. Preindl, R. Ahmed and A. Emadi, Long short-term memory networks for accurate state-of-charge estimation of Li-ion batteries, IEEE Transactions on Industrial Electronics 65(8), 2018."),
  N("F. Yang, S. Zhang, W. Li and Q. Miao, State-of-charge estimation of lithium-ion batteries using LSTM and UKF, Energy, 2020."),
  N("K. L. Wong et al., Li-ion batteries state-of-charge estimation using deep LSTM at various battery specifications and discharge cycles, GoodIT 2021."),
  N("Accurate state of charge estimation for real-world battery systems using a novel grid search and cross validated optimised LSTM neural network, Energies 15(24), 9654, 2022."),
];

const doc = new Document({
  creator: "Hafiz Rahman Elikkottil",
  title: "Physics-Based Energy and Race Strategy Optimisation for the Sasol Solar Challenge 2026: Final Project Report",
  styles: {
    default: { document: { run: { font: BODY, size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: HEAD, size: 30, bold: true, color: "365F91" },
        paragraph: { spacing: { before: 280, after: 120 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: HEAD, size: 23, bold: true, color: "4F81BD" },
        paragraph: { spacing: { before: 180, after: 70 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1080, right: 1224, bottom: 1080, left: 1224, header: 708, footer: 708 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [
      new TextRun({ text: "ID4100 Final Report  |  Hafiz Rahman Elikkottil  |  ", size: 18 }),
      new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] }) },
    children: [...cover, ...s1, ...s2, ...s3, ...s4, ...s5, ...s6],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "ID4100_Final_Report_Hafiz_EE24B104.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out);
});
