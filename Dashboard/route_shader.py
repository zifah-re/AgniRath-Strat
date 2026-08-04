import json
import math
from pathlib import Path
from constants import MASS, RHO, CDA, G, MOTOR_EFF, REGEN_EFF, CRR
import re
import argparse
import glob

def main(file_name):
    SCRIPT_DIR = Path(__file__).resolve().parent
    if file_name is None:
        file_name = input("Enter file name: ")
    FILE_PATH = SCRIPT_DIR / "Saves" / file_name
    SAVE_FILE_PATH = SCRIPT_DIR / "Shaded" / file_name[0:len(file_name)-5]

    with open(FILE_PATH, 'r') as f:
        data = json.loads(f.read())

    distance_profile = data['profile']['Distance']      
    gradient_profile = data['profile']['Gradient']       
    speed_limit = data['profile']['SpeedLimit']         
    coords = data['profile']['Coordinates']

    # =====================================================================
    # Patch speed_limit: fill 0s by interpolating from nearest valid neighbours
    # TomTom sometimes returns 0 where it has no data.
    # =====================================================================
    def patch_speed_limits(sl):
        """Replace 0s with linearly interpolated values from nearest non-zero neighbours."""
        patched = sl[:]
        n = len(patched)
        i = 0
        total_patched = 0
        while i < n:
            if patched[i] <= 0:
                j = i
                while j < n and patched[j] <= 0:
                    j += 1
                left = patched[i - 1] if i > 0 else None
                right = patched[j] if j < n else None

                if left is not None and right is not None:
                    span = j - i + 2
                    for k in range(i, j):
                        t = (k - i + 1) / span
                        patched[k] = left + t * (right - left)
                elif left is not None:
                    for k in range(i, j):
                        patched[k] = left
                elif right is not None:
                    for k in range(i, j):
                        patched[k] = right
                else:
                    for k in range(i, j):
                        patched[k] = 120

                total_patched += (j - i)
                i = j
            else:
                i += 1

        if total_patched > 0:
            print(f"WARNING: Patched {total_patched} points with speed_limit=0 (TomTom data gap)")
        return patched

    speed_limit = patch_speed_limits(speed_limit)

    # --- Constants ---
    P_MOTOR_MAX = 4000.0    # W (peak motor electrical power we allow)
    CRUISE_SPEED = 90 / 3.6         # m/s — target cruising speed
    LOWEST_SPEED_HIGHWAY = 60 / 3.6 # m/s — failure floor on highways (speed_limit >= 100)
    LOWEST_SPEED_OTHER   = 40 / 3.6 # m/s — failure floor elsewhere
    TRAILER_EXIT_SPEED   = 90 / 3.6 # m/s — assumed speed after being trailered
    V_FLOOR = 5 / 3.6               # m/s — absolute minimum to prevent division by zero

    # NEW: realistic braking model
    # Comfortable, sustainable braking deceleration for a lightweight (300 kg)
    # solar car on open road — not panic braking. 2.5 m/s^2 ~ 0.25g.
    # This is a judgment call, not measured — tune it if you have real brake
    # test data. Lower = more conservative (car must "see" limit changes from
    # further away) = more stretches will look infeasible.
    BRAKE_DECEL_MS2 = 2.5

    n = len(gradient_profile)


    def get_v_limit(i):
        """Legal speed limit at point i (already patched, but belt-and-suspenders)."""
        sl = speed_limit[i]
        if sl <= 0:
            return CRUISE_SPEED
        return min(CRUISE_SPEED, sl / 3.6)


    def get_speed_floor(i):
        """Minimum acceptable speed at point i."""
        return LOWEST_SPEED_HIGHWAY if speed_limit[i] >= 100 else LOWEST_SPEED_OTHER


    # =====================================================================
    # Backward pass: compute the fastest speed the car could LEGALLY and
    # PHYSICALLY be going at each point, accounting for the fact that
    # braking to a lower limit ahead takes real distance.
    #
    # v_cap[i] = min(
    #     v_limit[i],                                   # the actual legal limit here
    #     sqrt(v_cap[i+1]^2 + 2 * BRAKE_DECEL * ds)      # what braking physics allows
    # )
    #
    # This is the standard backward speed-profile pass used in racing-line /
    # ADAS planning: walk from the end of the route to the start, and at each
    # step, cap the entry speed so there's enough room to brake for whatever
    # is coming next.
    # =====================================================================
    v_limit_arr = [get_v_limit(i) for i in range(n)]
    v_cap = v_limit_arr[:]  # start equal to the legal limit, then tighten backward

    for i in range(n - 2, -1, -1):
        ds = (distance_profile[i + 1] - distance_profile[i]) * 1000  # metres
        if ds <= 0:
            # Duplicate/zero-length point in the KML — no distance to brake over,
            # so this point inherits the next point's cap directly.
            v_cap[i] = min(v_cap[i], v_cap[i + 1])
            continue
        achievable = math.sqrt(v_cap[i + 1] ** 2 + 2 * BRAKE_DECEL_MS2 * ds)
        v_cap[i] = min(v_cap[i], achievable)


    def simulate(safe_mask):
        """
        Run a single forward simulation through the entire route.
        Segments where safe_mask[i] is False are skipped (trailered).
        Returns (speeds, motor_power, failed_indices).
        """
        spd = [0.0] * n
        mpow = [0.0] * n
        failed = []

        v = CRUISE_SPEED

        for i in range(n):
            # Skip trailered segments
            if not safe_mask[i]:
                spd[i] = 0.0
                mpow[i] = 0.0
                if i + 1 < n and safe_mask[i + 1]:
                    v = TRAILER_EXIT_SPEED
                continue

            # --- Entry clamp: this is the fix. ---
            # v_cap already encodes "the fastest you could legally/physically be
            # going here, given you must be able to brake for anything ahead."
            # Clamping HERE (before any force calc, speed report, or failure
            # check at THIS index) closes the loophole where the car could use
            # illegal, un-braked-for kinetic energy to power through a hill.
            v = min(v, v_cap[i])
            v = max(v, V_FLOOR)

            grad = gradient_profile[i] / 100.0
            v_limit = v_limit_arr[i]

            # --- Forces ---
            f_drag = 0.5 * RHO * CDA * (v ** 2)
            f_roll = MASS * G * CRR
            f_grav = MASS * G * grad

            # --- Motor strategy ---
            if grad > 0:
                f_motor = (P_MOTOR_MAX * MOTOR_EFF) / v
                p_elec = P_MOTOR_MAX
            else:
                f_resist = f_drag + f_roll + f_grav
                if f_resist > 0 and v <= v_limit:
                    p_mech_needed = f_resist * v
                    p_mech = min(p_mech_needed, P_MOTOR_MAX * MOTOR_EFF)
                    f_motor = p_mech / v
                    p_elec = p_mech / MOTOR_EFF
                else:
                    f_motor = 0.0
                    p_elec = 0.0

            f_net = f_motor - f_drag - f_roll - f_grav
            a = f_net / MASS

            spd[i] = v * 3.6
            mpow[i] = p_elec

            # Speed floor check — now operating on a speed that was legal
            # AND physically reachable to begin with, so this is no longer
            # contaminated by borrowed illegal kinetic energy.
            #
            # NOTE: deliberately NOT gated on "grad > 0". Real elevation data
            # is noisy — a sustained climb can have individual samples read as
            # flat or slightly negative due to GPS/elevation jitter, even while
            # the car is still stalled from the climb. Gating on grad>0 let a
            # genuinely-stalled car slip through undetected whenever a sample
            # happened to read non-positive (this is what produced "Springbok
            # Loop: min speed 20 km/h -> All clear", which is a contradiction).
            # f_grav/f_roll below already reflect the ACTUAL current gradient
            # (positive, flat, or negative), so this check still correctly
            # leaves real downhills alone — gravity assist there makes
            # f_net_at_floor positive on its own, without needing a grad>0 gate.
            floor = get_speed_floor(i)
            if v < floor:
                f_drag_at_floor = 0.5 * RHO * CDA * (floor ** 2)
                f_motor_at_floor = (P_MOTOR_MAX * MOTOR_EFF) / floor
                f_net_at_floor = f_motor_at_floor - f_drag_at_floor - f_roll - f_grav
                if f_net_at_floor < 0:
                    failed.append(i)

            # --- Advance ---
            # No more end-of-loop v_limit clamp here — the NEXT iteration's
            # entry clamp (v = min(v, v_cap[i+1])) is what enforces legality,
            # and it does so using the braking-aware cap, not an instant snap.
            if i + 1 < len(distance_profile):
                ds = (distance_profile[i + 1] - distance_profile[i]) * 1000
                if ds > 0:
                    dt = ds / v
                    v_new = v + a * dt
                    v = max(v_new, V_FLOOR)

        return spd, mpow, failed

    def generate_kml(flags: list[bool], coords: list[tuple[float, float]], output_filename: str = "path.kml"):
        """
        Generates a KML file with connected line segments colored Black (True) and Red (False).
        
        :param flags: List of boolean values (True / False) of length N.
        :param coords: List of (latitude, longitude) tuples of length N.
        :param output_filename: Target filename for the generated KML file.
        """
        if len(flags) != len(coords):
            raise ValueError(f"flags and coords arrays must have the same length. ({len(flags)}!={len(coords)})")
        
        if len(flags) < 2:
            raise ValueError("At least 2 points are required to draw a line segment.")

        # Colors in KML format (aabbggrr)
        COLOR_BLACK = "ff000000"
        COLOR_RED = "ff0000ff"

        # Group points into contiguous segments
        segments = []
        current_segment = [coords[0]]
        current_flag = flags[0]

        for i in range(1, len(flags)):
            current_segment.append(coords[i])
            
            # If the state changes or we hit the end, finalize the current segment
            if flags[i] != current_flag:
                segments.append((current_flag, current_segment))
                # Start the next segment sharing the overlap point for seamless connection
                current_segment = [coords[i]]
                current_flag = flags[i]

        # Append the final segment
        segments.append((current_flag, current_segment))

        # Construct KML XML content
        Day=re.search(r"(Day [1-8]{1})",file_name).group(1)
        kml_header = f"""<?xml version="1.0" encoding="UTF-8"?>
    <kml xmlns="http://www.opengis.net/kml/2.2">
    <Document>
        <name>Colored Path</name>
        <Folder>
            <name>{Day}</name>
            <open>1</open>
            <Style id="blackLine">
            <LineStyle>
                <color>ff000000</color>
                <width>4</width>
            </LineStyle>
            </Style>
            <Style id="redLine">
            <LineStyle>
                <color>ff0000ff</color>
                <width>4</width>
            </LineStyle>
            </Style>
    """

        kml_footer = """  </Folder>
        </Document>
    </kml>
    """

        placemarks = []
        for idx, (flag, seg_coords) in enumerate(segments):
            style_id = "blackLine" if flag else "redLine"
            
            # KML requires longitude, latitude (lon, lat)
            coord_string = " ".join(f"{lon},{lat},0" for lat, lon in seg_coords)
            
            placemark = f"""    <Placemark>
        <name>Segment {idx + 1} ({'True' if flag else 'False'})</name>
        <styleUrl>#{style_id}</styleUrl>
        <LineString>
            <tessellate>1</tessellate>
            <coordinates>
            {coord_string}
            </coordinates>
        </LineString>
        </Placemark>"""
            placemarks.append(placemark)

        # Write to file
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(kml_header + "\n".join(placemarks) + "\n" + kml_footer)

        print(f"KML file successfully written ")

    # =====================================================================
    # Find contiguous uphill stretches
    # =====================================================================
    stretches = []
    i = 0
    while i < n:
        if gradient_profile[i] > 0:
            start = i
            while i < n and gradient_profile[i] > 0:
                i += 1
            stretches.append((start, i - 1))
        else:
            i += 1

    # =====================================================================
    # Iterative simulation: first pass gets actual speeds, subsequent passes
    # resolve trailer cascades.
    # =====================================================================
    safe = [True] * n
    MAX_ITERATIONS = 10

    def mark_unsafe(new_failures, safe, stretches):
        """
        Mark unsafe points. If a failure falls inside a recognised uphill
        stretch, trailer the whole stretch (that's the intended strategy —
        don't attempt a hill you can't clear). If a failure falls outside
        any stretch (now possible since the floor check isn't gated on
        grad>0), mark just that point — the car stalled there specifically,
        not as part of a climb.
        """
        handled = set()
        for (s, e) in stretches:
            if any(j in new_failures for j in range(s, e + 1)):
                for j in range(s, e + 1):
                    safe[j] = False
                handled.update(range(s, e + 1))
        for j in new_failures:
            if j not in handled:
                safe[j] = False


    first_pass_speeds, first_pass_power, failed = simulate(safe)

    if failed:
        mark_unsafe(set(failed), safe, stretches)

        for iteration in range(1, MAX_ITERATIONS):
            speeds, motor_power, failed = simulate(safe)
            if not failed:
                break
            mark_unsafe(set(failed), safe, stretches)
        iteration_count = iteration + 1
    else:
        speeds = first_pass_speeds
        motor_power = first_pass_power
        iteration_count = 1

    # =====================================================================
    # Report
    # =====================================================================
    unsafe_count = safe.count(False)
    if unsafe_count > 0:
        print(f"\n{unsafe_count} unsafe points (need trailer)\n")

        stretch_index_set = set()
        for (s, e) in stretches:
            stretch_index_set.update(range(s, e + 1))

        any_stretch_reported = False
        print("Failed uphill stretches:")
        for (s, e) in stretches:
            if not safe[s]:
                any_stretch_reported = True
                peak_grad = max(gradient_profile[s:e+1])
                dist_start = distance_profile[s]
                dist_end = distance_profile[e] if e < len(distance_profile) else distance_profile[-1]
                length_m = (dist_end - dist_start) * 1000
                min_speed = min(first_pass_speeds[s:e+1])
                entry_speed = first_pass_speeds[s]
                print(f"  idx {s:>5d}-{e:<5d}  |  dist {dist_start:>7.1f}-{dist_end:<7.1f} km  "
                    f"|  length {length_m:>6.0f} m  |  peak grad {peak_grad:>5.2f}%  "
                    f"|  entry {entry_speed:>5.1f} -> min {min_speed:>5.1f} km/h")
        if not any_stretch_reported:
            print("  (none)")

        # Points marked unsafe that AREN'T part of any recognized uphill stretch —
        # e.g. a genuine stall carried over onto a flat/noisy-gradient sample.
        # Without this, unsafe_count could be >0 with nothing explaining why.
        isolated = [i for i in range(n) if not safe[i] and i not in stretch_index_set]
        if isolated:
            print("\nIsolated unsafe points (not part of a recognized uphill stretch):")
            for i in isolated:
                print(f"  idx {i:>5d}  |  dist {distance_profile[i]:>7.1f} km  "
                    f"|  grad {gradient_profile[i]:>5.2f}%  |  speed {first_pass_speeds[i]:>5.1f} km/h")

        print(f"\nConverged in {iteration_count} iteration(s)")

        # Report where the braking cap actually bit (informational, so you can
        # sanity check it against the KML instead of trusting it blindly)
        tightened = [i for i in range(n) if v_cap[i] < v_limit_arr[i] - 0.1]
        if tightened:
            print(f"\n{len(tightened)} points where anticipatory braking reduced "
                f"entry speed below the raw posted limit (BRAKE_DECEL_MS2={BRAKE_DECEL_MS2}).")
    else:
        print("\nAll clear — car can handle every hill on the route!")
        print(f"  Max gradient:    {max(gradient_profile):.2f}%")
        print(f"  Min speed seen:  {min(first_pass_speeds):.1f} km/h")
        print(f"  Max motor power: {max(first_pass_power):.0f} W")
        print(f"  Uphill stretches checked: {len(stretches)}")

    generate_kml(safe, coords, SAVE_FILE_PATH)

parser = argparse.ArgumentParser(description="Plot solar JSONL files vs IST time of day")
parser.add_argument("files", nargs="*", help="JSONL paths (default: auto-discover)")
parser.add_argument("--output-dir", default=".", help="Directory to save plots (default: current dir)")
args = parser.parse_args()

paths = args.files or sorted(glob.glob("Dashboard\\Saves\\*.kml.save"))
if not paths:
    print("No JSONL files found. Pass file paths or run from the folder containing them.")

for p in paths:
    main(Path(p).name)