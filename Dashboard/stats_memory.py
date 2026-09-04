from __future__ import annotations

import json
import math
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / ".stats_uploads"

# A gap larger than this is not integrated inside a single metric stream.
# This prevents one stale packet from creating fake distance/energy.
MAX_GAP_SECONDS = 120.0


def finite(value: Any) -> Optional[float]:
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def parse_time(packet: dict) -> Optional[datetime]:
    raw = packet.get("_rx_time")

    if raw is None:
        return None

    try:
        return datetime.fromisoformat(
            str(raw).replace("Z", "+00:00")
        )
    except (TypeError, ValueError):
        return None


def solar_power(packet: dict) -> Optional[float]:
    """
    Exact same solar-power definition used in main.py:

        Output_Voltage_A × Output_Current_A
      + Output_Voltage_B × Output_Current_B
      + Output_Voltage_C × Output_Current_C
      + Output_Voltage_D × Output_Current_D
    """

    total = 0.0
    found = False

    for channel in "ABCD":
        voltage = finite(
            packet.get(f"Output_Voltage_{channel}")
        )
        current = finite(
            packet.get(f"Output_Current_{channel}")
        )

        if voltage is not None and current is not None:
            total += voltage * current
            found = True

    return total if found else None


def power_lost(packet: dict) -> Optional[float]:
    """
    Prefer an explicit Motor_Power field when the telemetry contains one.

    Otherwise use exactly the electrical power definition currently used
    by main.py:

        Bus_Power = Bus_Voltage × Bus_Current
    """

    motor_power = finite(packet.get("Motor_Power"))

    if motor_power is not None:
        return abs(motor_power)

    bus_voltage = finite(packet.get("Bus_Voltage"))
    bus_current = finite(packet.get("Bus_Current"))

    if bus_voltage is None or bus_current is None:
        return None

    return abs(bus_voltage * bus_current)


def vehicle_velocity(packet: dict) -> Optional[float]:
    """
    Vehicle_Velocity in the existing main.py is converted from m/s to km/h
    by multiplying by 3.6.

    Therefore the raw log value is treated as m/s here.
    """

    value = finite(packet.get("Vehicle_Velocity"))

    if value is None:
        return None

    return max(0.0, value)


class StatsMemory:

    def __init__(self):
        self.runs: list[dict] = []
        self.loaded_files: list[str] = []

        self.live_packets: list[dict] = []
        self.live_capture = False

        self.uploaded: dict[str, dict] = {}

        UPLOAD_DIR.mkdir(exist_ok=True)

    # ============================================================
    # FILE READING
    # ============================================================

    def _read_file(self, path: Path) -> tuple[list[dict], Optional[datetime]]:
        """
        Returns:

            sorted_packets
            first_valid_row_time

        IMPORTANT:
        The first_valid_row_time is taken from the first valid JSONL row
        in the ACTUAL FILE ORDER.

        This is different from the earliest timestamp after sorting.
        The user specifically requested that run/day ordering be determined
        from the first valid row of every log file.
        """

        packets: list[dict] = []
        first_valid_time: Optional[datetime] = None

        try:
            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        packet = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if not isinstance(packet, dict):
                        continue

                    timestamp = parse_time(packet)

                    if timestamp is None:
                        continue

                    if first_valid_time is None:
                        first_valid_time = timestamp

                    packets.append(packet)

        except OSError:
            return [], None

        packets.sort(
            key=lambda packet: parse_time(packet)
        )

        return packets, first_valid_time

    # ============================================================
    # UPLOAD MANAGEMENT
    # ============================================================

    def _describe_uploaded(
        self,
        log_id: str
    ) -> Optional[dict]:

        entry = self.uploaded.get(log_id)

        if entry is None:
            return None

        packets = entry["packets"]

        if not packets:
            return None

        first_row_time = entry["first_row_time"]
        earliest_time = parse_time(packets[0])
        latest_time = parse_time(packets[-1])

        return {
            "id": log_id,
            "filename": entry["filename"],
            "packet_count": len(packets),

            # This is the first valid row in actual file order.
            "start_time": first_row_time.isoformat(),

            # This is the final timestamp in the run.
            "end_time": latest_time.isoformat(),

            # Same-day validation is based on the first valid row.
            "date": first_row_time.date().isoformat(),

            "duration_seconds": max(
                0.0,
                (latest_time - earliest_time).total_seconds()
            )
        }

    def available_logs(self) -> list[dict]:

        result = []

        for log_id in self.uploaded:
            description = self._describe_uploaded(log_id)

            if description is not None:
                result.append(description)

        # Runs are displayed in first-valid-row order.
        result.sort(
            key=lambda item: item["start_time"]
        )

        return result

    def upload_file(
        self,
        original_name: str,
        content: bytes
    ) -> dict:

        filename = Path(
            original_name or "run.jsonl"
        ).name

        log_id = uuid.uuid4().hex

        path = UPLOAD_DIR / (
            f"{log_id}_{filename}"
        )

        path.write_bytes(content)

        packets, first_row_time = self._read_file(path)

        if not packets or first_row_time is None:
            path.unlink(missing_ok=True)

            raise ValueError(
                "The selected file contains no valid JSONL "
                "packets with _rx_time."
            )

        self.uploaded[log_id] = {
            "path": path,
            "filename": filename,
            "packets": packets,
            "first_row_time": first_row_time
        }

        return self._describe_uploaded(log_id)

    def remove_uploaded(
        self,
        log_id: str
    ) -> bool:

        entry = self.uploaded.pop(
            log_id,
            None
        )

        if entry is None:
            return False

        entry["path"].unlink(
            missing_ok=True
        )

        self.runs = [
            run
            for run in self.runs
            if run["id"] != log_id
        ]

        self.loaded_files = [
            run["filename"]
            for run in self.runs
        ]

        return True

    # ============================================================
    # RUN LOADING
    # ============================================================

    def load_runs(
        self,
        ids: list[str]
    ) -> dict:

        chosen = []

        for log_id in map(str, ids):

            entry = self.uploaded.get(log_id)

            if entry is None:
                continue

            first_row_time = entry[
                "first_row_time"
            ]

            chosen.append({
                "id": log_id,
                "filename": entry["filename"],
                "packets": entry["packets"],

                # Ordering is based on first valid row.
                "first_row_time": first_row_time,

                # Same-day restriction is also based on first valid row.
                "date": first_row_time.date()
            })

        if not chosen:
            return {
                "success": False,
                "error": (
                    "No valid uploaded runs were selected."
                )
            }

        # --------------------------------------------------------
        # HARD SAME-DAY RULE
        # --------------------------------------------------------

        days = {
            run["date"].isoformat()
            for run in chosen
        }

        if len(days) != 1:
            return {
                "success": False,
                "error": (
                    "All selected log files must belong to "
                    "the same calendar day. The date is determined "
                    "from _rx_time in the first valid row of every file."
                ),
                "days_found": sorted(days)
            }

        # --------------------------------------------------------
        # RUN ORDERING
        # --------------------------------------------------------

        # The user's requested ordering:
        # compare the first valid row of every run.
        self.runs = sorted(
            chosen,
            key=lambda run: run["first_row_time"]
        )

        self.loaded_files = [
            run["filename"]
            for run in self.runs
        ]

        return {
            "success": True,
            "loaded_files": self.loaded_files,
            "run_count": len(self.runs),
            "date": next(iter(days))
        }

    # ============================================================
    # LIVE MEMORY
    # ============================================================

    def start_live(self):

        self.live_packets = []
        self.live_capture = True

    def stop_live(self):

        self.live_capture = False

    def clear_live(self):

        self.live_packets = []

    def add_live_packet(
        self,
        packet: dict
    ):

        if not self.live_capture:
            return

        if not isinstance(packet, dict):
            return

        timestamp = parse_time(packet)

        if timestamp is None:
            return

        # If historical logs are loaded, live packets must remain
        # on that same calendar day.
        if self.runs:
            historical_day = self.runs[0][
                "date"
            ]

            if timestamp.date() != historical_day:
                return

        self.live_packets.append(
            dict(packet)
        )

    # ============================================================
    # NUMERICAL INTEGRATION
    # ============================================================

    @staticmethod
    def integrate_sample(
        previous_sample,
        current_time: datetime,
        current_value: Optional[float],
        accumulator: float
    ):

        if current_value is None:
            return accumulator, previous_sample

        if previous_sample is not None:

            previous_time, previous_value = (
                previous_sample
            )

            dt = (
                current_time
                - previous_time
            ).total_seconds()

            if (
                0 < dt <= MAX_GAP_SECONDS
            ):
                accumulator += (
                    (
                        previous_value
                        + current_value
                    )
                    * 0.5
                    * dt
                )

        return (
            accumulator,
            (
                current_time,
                current_value
            )
        )

    # ============================================================
    # PROCESS ONE RUN
    # ============================================================

    def process_run(
        self,
        filename: str,
        packets: list[dict]
    ) -> dict:

        if not packets:
            return {
                "filename": filename,
                "timeline": [],
                "stats": {}
            }

        distance_m = 0.0
        solar_ws = 0.0
        loss_ws = 0.0
        active_time_seconds = 0.0

        maximum_velocity_mps = 0.0
        maximum_motor_velocity = 0.0

        velocity_samples: list[float] = []
        soc_values: list[float] = []

        last_velocity = None
        last_solar = None
        last_loss = None

        previous_packet_time = None

        timeline = []

        for packet in packets:

            timestamp = parse_time(packet)

            if timestamp is None:
                continue

            # ----------------------------------------------------
            # ACTIVE TIME
            # ----------------------------------------------------

            if previous_packet_time is not None:

                dt = (
                    timestamp
                    - previous_packet_time
                ).total_seconds()

                if (
                    0 < dt <= MAX_GAP_SECONDS
                ):
                    active_time_seconds += dt

            previous_packet_time = timestamp

            # ----------------------------------------------------
            # VALUES
            # ----------------------------------------------------

            velocity = vehicle_velocity(packet)
            current_solar_power = solar_power(packet)
            current_loss_power = power_lost(packet)

            # ----------------------------------------------------
            # DISTANCE
            # ----------------------------------------------------

            if velocity is not None:

                maximum_velocity_mps = max(
                    maximum_velocity_mps,
                    velocity
                )

                velocity_samples.append(
                    velocity
                )

                (
                    distance_m,
                    last_velocity
                ) = self.integrate_sample(
                    last_velocity,
                    timestamp,
                    velocity,
                    distance_m
                )

            # ----------------------------------------------------
            # SOLAR ENERGY
            # ----------------------------------------------------

            if current_solar_power is not None:

                (
                    solar_ws,
                    last_solar
                ) = self.integrate_sample(
                    last_solar,
                    timestamp,
                    current_solar_power,
                    solar_ws
                )

            # ----------------------------------------------------
            # ENERGY LOST
            # ----------------------------------------------------

            if current_loss_power is not None:

                (
                    loss_ws,
                    last_loss
                ) = self.integrate_sample(
                    last_loss,
                    timestamp,
                    current_loss_power,
                    loss_ws
                )

            # ----------------------------------------------------
            # MOTOR VELOCITY
            # ----------------------------------------------------

            motor_velocity = finite(
                packet.get(
                    "Motor_Velocity"
                )
            )

            if motor_velocity is not None:

                maximum_motor_velocity = max(
                    maximum_motor_velocity,
                    abs(motor_velocity)
                )

            # ----------------------------------------------------
            # SOC
            # ----------------------------------------------------

            soc = finite(
                packet.get("SOC_Ah")
            )

            if soc is not None:
                soc_values.append(soc)

            # ----------------------------------------------------
            # GRAPH POINT
            # ----------------------------------------------------

            timeline.append({
                "timestamp": timestamp.isoformat(),

                "epoch_ms": int(
                    timestamp.timestamp()
                    * 1000
                ),

                # Null deliberately means "no data point".
                "velocity_kmh": (
                    velocity * 3.6
                    if velocity is not None
                    else None
                ),

                "solar_power_w":
                    current_solar_power,

                "power_loss_w":
                    current_loss_power,

                "cumulative_distance_km":
                    distance_m / 1000.0,

                "cumulative_energy_received_wh":
                    solar_ws / 3600.0,

                "cumulative_energy_lost_wh":
                    loss_ws / 3600.0
            })

        start_time = (
            parse_time(packets[0])
        )

        end_time = (
            parse_time(packets[-1])
        )

        stats = {
            "distance_km":
                distance_m / 1000.0,

            "energy_received_wh":
                solar_ws / 3600.0,

            "energy_lost_wh":
                loss_ws / 3600.0,

            "net_energy_wh":
                (
                    solar_ws
                    - loss_ws
                )
                / 3600.0,

            "active_time_seconds":
                active_time_seconds,

            "max_velocity_kmh":
                maximum_velocity_mps * 3.6,

            "average_velocity_kmh": (
                (
                    sum(velocity_samples)
                    / len(velocity_samples)
                )
                * 3.6
                if velocity_samples
                else 0.0
            ),

            "max_motor_velocity":
                maximum_motor_velocity,

            "initial_soc_ah": (
                soc_values[0]
                if soc_values
                else None
            ),

            "final_soc_ah": (
                soc_values[-1]
                if soc_values
                else None
            ),

            "min_soc_ah": (
                min(soc_values)
                if soc_values
                else None
            ),

            "max_soc_ah": (
                max(soc_values)
                if soc_values
                else None
            )
        }

        return {
            "filename": filename,

            "start_time": (
                start_time.isoformat()
                if start_time
                else None
            ),

            "end_time": (
                end_time.isoformat()
                if end_time
                else None
            ),

            "stats": stats,

            "timeline": timeline
        }

    # ============================================================
    # BUILD RESPONSE
    # ============================================================

    def build(self) -> dict:

        processed_runs = []

        for run in self.runs:

            processed_runs.append(
                self.process_run(
                    run["filename"],
                    run["packets"]
                )
            )

        live_run = None

        if self.live_packets:

            live_packets = sorted(
                self.live_packets,
                key=lambda packet: parse_time(packet)
            )

            live_run = self.process_run(
                "LIVE_SESSION",
                live_packets
            )

            if live_run["timeline"]:
                processed_runs.append(
                    live_run
                )

        totals = {
            "cumulative_distance_km": 0.0,

            "cumulative_energy_received_wh": 0.0,

            "cumulative_energy_lost_wh": 0.0,

            "total_active_time_seconds": 0.0,

            "maximum_velocity_kmh": 0.0
        }

        boundaries = []

        # --------------------------------------------------------
        # THIS IS THE IMPORTANT PART
        #
        # Every run remains its OWN graph series.
        #
        # Therefore Chart.js will NEVER draw a fake line through
        # the real-world gap between two log files.
        #
        # But every new run receives the previous cumulative value
        # as its starting offset.
        # --------------------------------------------------------

        series = []

        for run_index, run in enumerate(
            processed_runs
        ):

            stats = run["stats"]

            distance_base = totals[
                "cumulative_distance_km"
            ]

            energy_in_base = totals[
                "cumulative_energy_received_wh"
            ]

            energy_out_base = totals[
                "cumulative_energy_lost_wh"
            ]

            run_points = []

            for point in run["timeline"]:

                run_points.append({
                    **point,

                    "cumulative_distance_km":
                        distance_base
                        + point[
                            "cumulative_distance_km"
                        ],

                    "cumulative_energy_received_wh":
                        energy_in_base
                        + point[
                            "cumulative_energy_received_wh"
                        ],

                    "cumulative_energy_lost_wh":
                        energy_out_base
                        + point[
                            "cumulative_energy_lost_wh"
                        ]
                })

            series.append({
                "name": run["filename"],

                "points": run_points
            })

            if run_points:

                boundaries.append({
                    "run": run["filename"],

                    "start_epoch_ms":
                        run_points[0]["epoch_ms"],

                    "end_epoch_ms":
                        run_points[-1]["epoch_ms"],

                    "start_time":
                        run["start_time"],

                    "end_time":
                        run["end_time"]
                })

            totals[
                "cumulative_distance_km"
            ] += stats.get(
                "distance_km",
                0.0
            )

            totals[
                "cumulative_energy_received_wh"
            ] += stats.get(
                "energy_received_wh",
                0.0
            )

            totals[
                "cumulative_energy_lost_wh"
            ] += stats.get(
                "energy_lost_wh",
                0.0
            )

            totals[
                "total_active_time_seconds"
            ] += stats.get(
                "active_time_seconds",
                0.0
            )

            totals[
                "maximum_velocity_kmh"
            ] = max(
                totals[
                    "maximum_velocity_kmh"
                ],

                stats.get(
                    "max_velocity_kmh",
                    0.0
                )
            )

        totals["net_energy_wh"] = (
            totals[
                "cumulative_energy_received_wh"
            ]
            -
            totals[
                "cumulative_energy_lost_wh"
            ]
        )

        # Global timeline bounds.
        global_start_ms = None
        global_end_ms = None

        if boundaries:

            global_start_ms = min(
                boundary["start_epoch_ms"]
                for boundary in boundaries
            )

            global_end_ms = max(
                boundary["end_epoch_ms"]
                for boundary in boundaries
            )

        loaded_date = None

        if self.runs:

            loaded_date = self.runs[0][
                "date"
            ].isoformat()

        return {
            "loaded_files":
                self.loaded_files,

            "loaded_date":
                loaded_date,

            "historical_run_count":
                len(self.runs),

            "live_capture":
                self.live_capture,

            "live_packet_count":
                len(self.live_packets),

            "overall":
                totals,

            "runs":
                processed_runs,

            "live_run":
                live_run,

            # Separate datasets per run.
            "series":
                series,

            "boundaries":
                boundaries,

            "global_start_ms":
                global_start_ms,

            "global_end_ms":
                global_end_ms,

            "calculation_notes": {

                "energy_received":
                    "Sum of all four MPPT output powers "
                    "(Output Voltage × Output Current), "
                    "integrated over packet time.",

                "energy_lost":
                    "Motor_Power when available; otherwise "
                    "Bus_Voltage × Bus_Current, matching "
                    "the existing main.py dashboard calculation.",

                "distance":
                    "Vehicle_Velocity integrated between "
                    "actual velocity samples.",

                "time_axis":
                    "Real global clock time for the loaded day. "
                    "Separate runs are separate chart datasets, "
                    "so real gaps remain visible and no fake line "
                    "is drawn between runs."
            }
        }


stats_memory = StatsMemory()