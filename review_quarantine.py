import pandas as pd
import os


def review_quarantine_buffer():
    buffer_path = "quarantine_buffer.csv"
    verified_path = "verified_history.csv"

    if not os.path.exists(buffer_path):
        print("Quarantine buffer is empty. No alerts to review.")
        return

    df = pd.read_csv(buffer_path)
    verified_records = []

    print(f"Initiating Human-in-the-Loop Review: {len(df)} records found.")

    for index, row in df.iterrows():
        print(f"\n--- Alert {index + 1} / {len(df)} ---")
        print(f"Time: {row['timestamp']} | IP: {row['src_ip']} -> {row['dst_ip']}:{row['port']}")
        print(f"Metrics: {row['dur']}s | {row['sbytes']} bytes | Proto: {row['proto']}")
        print(f"AI Flags -> XGBoost (Known): {row['xgb_score']} | IsoForest (Zero-Day): {row['iso_score']}")

        while True:
            decision = input("Verdict: [m]alicious, [b]enign, or [s]kip? ").strip().lower()
            if decision in ['m', 'b', 's']:
                break
            print("Invalid input. Use m, b, or s.")

        if decision == 's':
            continue

        # We strip the network context and append the binary label for the AI
        ml_record = row.drop(['timestamp', 'src_ip', 'dst_ip', 'port', 'xgb_score', 'iso_score']).to_dict()
        ml_record['label'] = 1 if decision == 'm' else 0
        verified_records.append(ml_record)

    # Append verifications to the continuous learning database
    if verified_records:
        verified_df = pd.DataFrame(verified_records)
        write_header = not os.path.exists(verified_path)
        verified_df.to_csv(verified_path, mode='a', header=write_header, index=False)
        print(f"\nSaved {len(verified_records)} verified records to {verified_path}.")

    # Clear the buffer to prevent duplicate reviews
    os.remove(buffer_path)
    print("Quarantine buffer cleared.")


if __name__ == "__main__":
    review_quarantine_buffer()