import pandas as pd
import os


def unify_and_sample(file_paths, output_filename, sample_fraction=0.1):
    print("1. Initializing unification protocol...")

    # adding headers/ coolumn names for the UNSW-NB15 according to the dataset documentation
    columns = [
        "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes",
        "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload", "Spkts", "Dpkts",
        "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
        "Sjit", "Djit", "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",
        "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
        "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
        "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label"
    ]
    # an array to hold the 4 dataset before concatenation
    # the dataset in itself is a matrix/ or a list, so we are storing the 4 datasets in a list of dataframes

    dataframes = []

    # Sequential Memory-Safe Loading
    for file in file_paths:
        if os.path.exists(file):
            print(f"Loading {file}..")
            # We enforce low_memory=False to prevent Pandas from crashing on mixed data types
            df_chunk = pd.read_csv(file, header=None, names=columns, low_memory=False)
            dataframes.append(df_chunk)
        else:
            print(f"{file} not found. Ensure the path is correct.")

    #  Concatenation
    print("Concatenating datasets.")
    # ignore_index = true will reset the index of the concatenated dataframe to a continuous range,
    # this prevents index duplication and ensures a clean, continuous index across the unified dataset.
    unified_df = pd.concat(dataframes, ignore_index=True)
    print(f"Total rows before sampling: {len(unified_df)}")

    #  Stratified Downsampling
    # 2.5 million rows is too heavy. We sample a fraction (e.g., 10%)
    # while ensuring the proportion of 'label' (0 or 1) remains identical.
    print(f"Downsampling to {sample_fraction * 100}% for local compute constraints...")

    # We use groupby on the label to ensure stratified sampling
    sampled_df = unified_df.groupby('label', group_keys=False).apply(
        lambda x: x.sample(frac=sample_fraction, random_state=42)
    )

    print(f"Final training rows: {len(sampled_df)}")

    # 5. Disk Serialization
    print(f"Saving optimized dataset to {output_filename}...")
    sampled_df.to_csv(output_filename, index=False)
    print("Unification complete.")


if __name__ == "__main__":
    raw_files = [
        "UNSW-NB15_1.csv",
        "UNSW-NB15_2.csv",
        "UNSW-NB15_3.csv",
        "UNSW-NB15_4.csv"
    ]

    unify_and_sample(raw_files, "../UNSW_NB15_optimized.csv", sample_fraction=0.1)
