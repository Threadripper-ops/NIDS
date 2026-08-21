import json
import time
import os


def tail_zeek_log(log_path):
    """
    A generator function that continuously watches the Zeek connection log.
    It yields new flow records as Zeek writes them to the disk.
    """
    # Open the file and seek to the absolute end.
    # We do not want to re-process old traffic every time we start the script.
    with open(log_path, 'r') as file:
        file.seek(0, os.SEEK_END)

        while True:
            line = file.readline()
            if not line:
                # If no new line is written, pause for a fraction of a second
                # to prevent the CPU from spinning at 100% usage.
                time.sleep(0.1)
                continue

            try:
                # Parse the JSON string into a Python dictionary
                flow_record = json.loads(line)
                yield flow_record
            except json.JSONDecodeError:
                # If Zeek is halfway through writing a line, skip and wait.
                pass


def map_zeek_to_unsw(zeek_json):
    """
    Translates the Zeek JSON keys into the exact column names expected
    by the UNSW-NB15 trained machine learning models.
    """
    # The 'conn.log' contains most of what we need, but the naming convention differs slightly.
    # We must explicitly map them to match the offline training phase.
    mapped_features = {
        'dur': zeek_json.get('duration', 0.0),
        'proto': zeek_json.get('proto', 'unknown'),
        'service': zeek_json.get('service', '-'),
        'state': zeek_json.get('conn_state', '-'),
        'spkts': zeek_json.get('orig_pkts', 0),
        'dpkts': zeek_json.get('resp_pkts', 0),
        'sbytes': zeek_json.get('orig_ip_bytes', 0),
        'dbytes': zeek_json.get('resp_ip_bytes', 0),
        # In a full deployment, you would map all relevant features here.
    }
    return mapped_features


if __name__ == "__main__":
    # Ensure Zeek is configured to output JSON and is running before starting this script.
    zeek_log_path = "/path/to/zeek/logs/current/conn.log"

    print("Initializing Zeek Log Watchdog...")

    try:
        for raw_zeek_record in tail_zeek_log(zeek_log_path):
            ml_ready_record = map_zeek_to_unsw(raw_zeek_record)
            print(f"Captured and Mapped Flow: {ml_ready_record}")
            # In Phase 3, this is where we will inject the ml_ready_record into the AI.
    except KeyboardInterrupt:
        print("\nTerminating Zeek Watchdog.")