# data_import.py

import pandas as pd
import gzip
import os
from datetime import datetime, timedelta
from config import (
    RAW_DATA_DIR,
    NORMALIZED_DATA_DIR,
    EXCHANGE,
    TARDIS_TYPE,
    START_DATE,
    END_DATE,
    INSTRUMENT
)

def normalize_tardis_data(exchange, tardis_type, start_date, end_date, instrument, raw_data_dir, normalized_data_dir):
    # Convert start and end dates to datetime objects
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')

    # File pattern for raw data
    raw_file_pattern = os.path.join(raw_data_dir, f"{exchange}_{tardis_type}_{instrument}_{{date}}.csv.gz")
    # File pattern for normalized data
    normalized_file = os.path.join(normalized_data_dir, f"{exchange}_{instrument}_normalized.csv")

    dfs = []
    
    # Generate a list of dates within the given range
    date_list = [start_date_dt + timedelta(days=x) for x in range((end_date_dt - start_date_dt).days)]
    
    for date in date_list:
        date_str = date.strftime('%Y-%m-%d')
        filename = raw_file_pattern.format(date=date_str)
        
        if os.path.exists(filename):
            with gzip.open(filename, 'rt') as f:
                df = pd.read_csv(f)

            # Convert 'timestamp' to datetime and set it as the index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us', utc=True)
            df = df.set_index('timestamp')

            # Resample to hourly frequency and take the last available entry
            df_hourly = df.resample('H').last()

            # Check if there are 24 data points; if not, skip the date
            if len(df_hourly) == 24:
                dfs.append(df_hourly)
                print(f"[PROCESSED] {date_str}")
            else:
                print(f"Skipping incomplete data for date: {date_str}")
        else:
            print(f"File not found: {filename}")

    # Combine all DataFrames if there are any, or return an empty DataFrame
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=False)
        # Save the normalized data
        combined_df.to_csv(normalized_file)
        print(f"Normalized data saved to {normalized_file}")
        return combined_df
    else:
        print("No valid data found in the given date range.")
        return pd.DataFrame()

if __name__ == "__main__":
    normalize_tardis_data(
        exchange=EXCHANGE,
        tardis_type=TARDIS_TYPE,
        start_date=START_DATE,
        end_date=END_DATE,
        instrument=INSTRUMENT,
        raw_data_dir=RAW_DATA_DIR,
        normalized_data_dir=NORMALIZED_DATA_DIR
    )