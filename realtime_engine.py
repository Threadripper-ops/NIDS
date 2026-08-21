import json
import time
import os
import csv
import joblib
import pandas as pd
from datetime import datetime

# --- System Configuration ---
ZEEK_LOG_PATH = "conn.log"  # Path to your live Zeek JSON log
ENABLE_IOT_SEGMENTATION = False  # Feature flag for Phase 5
IOT_SUBNET = "192.168.2."  # The dedicated VLAN/subnet for smart devices
QUARANTINE_FILE = "quarantine_buffer.csv"


def load_segmented_brains():
    """Conditionally loads the required machine learning artifacts into memory."""
    print("1. Loading Primary IT Baselines (UNSW-NB15)...")
    try:
        it_prep = joblib.load('core_transformer.pkl')
        it_xgb = joblib.load('core_xgboost.pkl')
        it_iso = joblib.load('core_iso_forest.pkl')
    except FileNotFoundError as e:
        print(f"[CRITICAL] Missing core IT model artifacts: {e}")
        exit(1)

    iot_prep, iot_iso = None, None
    if ENABLE_IOT_SEGMENTATION:
        print("2. Loading Secondary IoT Baselines (ToN_IoT)...")
        try:
            iot_prep = joblib.load('iot_feature_transformer.pkl')
            iot_iso = joblib.load('iot_iso_forest.pkl')
        except FileNotFoundError:
            print("[WARNING] IoT models not found. Proceeding with IT models only.")
            global ENABLE_IOT_SEGMENTATION
            ENABLE_IOT_SEGMENTATION = False

    print(" -> AI Engine fully initialized.\n")
    return (it_prep, it_xgb, it_iso), (iot_prep, iot_iso)


def map_zeek_to_unsw(zeek_json):
    """Translates Zeek's output into the standard IT schema."""
    return {
        'dur': float(zeek_json.get('duration', 0.0)),
        'proto': str(zeek_json.get('proto', 'unknown')),
        'service': str(zeek_json.get('service', '-')),
        'state': str(zeek_json.get('conn_state', '-')),
        'spkts': int(zeek_json.get('orig_pkts', 0)),
        'dpkts': int(zeek_json.get('resp_pkts', 0)),
        'sbytes': int(zeek_json.get('orig_ip_bytes', 0)),
        'dbytes': int(zeek_json.get('resp_ip_bytes', 0))
        # Expand this dictionary to include all features used in your Phase 1 training
    }


def map_zeek_to_ton_iot(zeek_json):
    """Translates Zeek's output into the strict IoT schema."""
    return {
        'dur': float(zeek_json.get('duration', 0.0)),
        'proto': str(zeek_json.get('proto', 'unknown')),
        'sbytes': int(zeek_json.get('orig_ip_bytes', 0)),
        # Expand based on the specific columns you retained in the ToN_IoT prep script
    }


def log_to_quarantine(raw_flow, xgb_pred, if_pred, device_type):
    """Appends flagged traffic features to a CSV buffer for human review."""
    file_exists = os.path.isfile(QUARANTINE_FILE)

    row = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'device_segment': device_type,
        'src_ip': raw_flow.get('id.orig_h', 'Unknown'),
        'dst_ip': raw_flow.get('id.resp_h', 'Unknown'),
        'port': raw_flow.get('id.resp_p', 'Unknown'),
        'xgb_score': xgb_pred,
        'iso_score': if_pred,

        # Core behavioral features mapped for retraining
        'dur': raw_flow.get('duration', 0.0),
        'proto': raw_flow.get('proto', 'unknown'),
        'service': raw_flow.get('service', '-'),
        'state': raw_flow.get('conn_state', '-'),
        'spkts': raw_flow.get('orig_pkts', 0),
        'sbytes': raw_flow.get('orig_ip_bytes', 0)
    }

    with open(QUARANTINE_FILE, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def evaluate_and_log(raw_flow, xgb_pred, if_pred, device_type):
    """Analyzes the AI outputs, prints alerts, and triggers quarantine logging."""
    if xgb_pred == 0 and if_pred == 0:
        return  # Traffic is benign. Skip to save CPU cycles.

    source_ip = raw_flow.get('id.orig_h', 'Unknown_IP')
    dest_ip = raw_flow.get('id.resp_h', 'Unknown_IP')
    port = raw_flow.get('id.resp_p', 'Unknown_Port')

    # Terminal Alerting
    if device_type == "IoT":
        print(f"[IOT ANOMALY] Rigid baseline violated by {source_ip} -> {dest_ip}:{port}")
    else:
        if xgb_pred == 1 and if_pred == 1:
            print(f"[CRITICAL THREAT] Both engines flagged {source_ip} -> {dest_ip}:{port}")
        elif xgb_pred == 1:
            print(f"[KNOWN ATTACK] XGBoost flagged {source_ip} -> {dest_ip}:{port}")
        elif if_pred == 1:
            print(f"[ANOMALY] Isolation Forest flagged {source_ip} -> {dest_ip}:{port}")

    # Write to Human-in-the-Loop buffer
    log_to_quarantine(raw_flow, xgb_pred, if_pred, device_type)


def route_and_infer(raw_flow, it_models, iot_models):
    """Routes traffic to the appropriate ML matrix based on network topology."""
    source_ip = raw_flow.get('id.orig_h', '')

    it_prep, it_xgb, it_iso = it_models
    iot_prep, iot_iso = iot_models

    # Branch 1: IoT Segment
    if ENABLE_IOT_SEGMENTATION and iot_prep and source_ip.startswith(IOT_SUBNET):
        ml_ready = map_zeek_to_ton_iot(raw_flow)
        scaled_flow = iot_prep.transform(pd.DataFrame([ml_ready]))
        iso_prediction = 1 if iot_iso.predict(scaled_flow)[0] == -1 else 0

        evaluate_and_log(raw_flow, 0, iso_prediction, "IoT")

    # Branch 2: Standard IT Segment
    else:
        ml_ready = map_zeek_to_unsw(raw_flow)
        scaled_flow = it_prep.transform(pd.DataFrame([ml_ready]))

        xgb_prediction = it_xgb.predict(scaled_flow)[0]
        iso_prediction = 1 if it_iso.predict(scaled_flow)[0] == -1 else 0

        evaluate_and_log(raw_flow, xgb_prediction, iso_prediction, "IT")


def realtime_inference_loop(it_models, iot_models):
    """Tails the live Zeek log and continuously feeds the routing engine."""
    print(f"3. Connecting to live Zeek stream at {ZEEK_LOG_PATH}...")

    if not os.path.exists(ZEEK_LOG_PATH):
        print(f"[CRITICAL] Zeek log not found at {ZEEK_LOG_PATH}. Is Zeek running?")
        exit(1)

    with open(ZEEK_LOG_PATH, 'r') as file:
        file.seek(0, os.SEEK_END)
        print(" -> Online and actively monitoring traffic. Press Ctrl+C to stop.\n")

        try:
            while True:
                line = file.readline()
                if not line:
                    time.sleep(0.05)
                    continue

                try:
                    raw_flow = json.loads(line)
                    route_and_infer(raw_flow, it_models, iot_models)
                except json.JSONDecodeError:
                    pass
                except Exception as e:
                    print(f"[!] Inference Error: {e}")
        except KeyboardInterrupt:
            print("\nShutting down AI Engine safely.")


if __name__ == "__main__":
    it_artifacts, iot_artifacts = load_segmented_brains()
    realtime_inference_loop(it_artifacts, iot_artifacts)