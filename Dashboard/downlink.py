import serial
import serial.tools.list_ports
import json
import sys
import asyncio # <-- Add this import
from datetime import datetime, timezone

PORT = "COM7"
BAUD = 115200

def find_port():
    # ... (Keep existing find_port code)
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found.")
        return
    print("Available ports:")
    for p in ports:
        print(f"  {p.device}  —  {p.description}")

# Add queue and loop parameters with defaults so it can still run standalone
def main(queue=None, loop=None): 
    if len(sys.argv) > 1:
        port = sys.argv[1]
    else:
        port = PORT

    OUTPUT_FILE = datetime.now().strftime("Logs\\telemetry_log_%Y-%m-%d_%H-%M-%S.jsonl")

    find_port()
    print(f"\nConnecting to {port} at {BAUD} baud ...")

    try:
        ser = serial.Serial(port, BAUD, timeout=1)
    except serial.SerialException as e:
        print(f"Error opening port: {e}")
        sys.exit(1)

    print(f"Logging to {OUTPUT_FILE}. Press Ctrl+C to stop.\n")

    with open(OUTPUT_FILE, "a") as f:
        while True:
            try:
                raw = ser.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="ignore").strip()
                if not line.startswith("{"):
                    print(f"[esp32] {line}")
                    continue

                data = json.loads(line)
                # Record local receive time with timezone offset so timestamps are correct for the host machine.
                data["_rx_time"] = datetime.now().astimezone().isoformat()

                out = json.dumps(data, separators=(",", ":"))
                f.write(out + "\n")
                f.flush()
                print(out)

                # Send the data to FastAPI if the thread was started with a queue
                if queue and loop:
                    # main.py expects a tuple of (packet_type, data)
                    # Adjust the "type" key below based on how your ESP32 formats the JSON
                    ptype = data.get("type") or ("A" if "SOC_Ah" in data else "B")
                    
                    # Thread-safe way to put data into the asyncio Queue
                    loop.call_soon_threadsafe(queue.put_nowait, (ptype, data))

            except json.JSONDecodeError:
                print(f"[bad json] {line!r}")
            except KeyboardInterrupt:
                print("\nStopped.")
                ser.close()
                break

if __name__ == "__main__":
    main()