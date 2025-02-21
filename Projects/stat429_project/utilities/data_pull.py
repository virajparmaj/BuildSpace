# data_pull.py

import os
from datetime import datetime, timedelta
from tardis_dev import datasets
from config import (
    TARDIS_TYPE,
    START_DATE,
    END_DATE,
    EXCHANGE,
    INSTRUMENT,
    RAW_DATA_DIR
)

def get_api_key():
    api_key = os.environ.get('TARDIS_API_KEY')
    if not api_key:
        raise ValueError("TARDIS_API_KEY not found in environment variables")
    return api_key

def download_tardis_data(exchange, start_date, end_date, tardis_sym, instrument, save_dir):
    api_key = get_api_key()

    cur_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")

    while cur_date < end_date_dt:
        cur_str = cur_date.strftime("%Y-%m-%d")
        tom_str = (cur_date + timedelta(days=1)).strftime("%Y-%m-%d")

        filename = os.path.join(save_dir, f"{exchange}_{tardis_sym}_{instrument}_{cur_str}.csv.gz")
        if os.path.exists(filename):
            print(f"File already exists: {filename}")
            cur_date += timedelta(days=1)
            continue

        # Download data using the Tardis API
        datasets.download(
            exchange=exchange,
            data_types=[tardis_sym],
            from_date=cur_str,
            to_date=tom_str,
            symbols=[instrument],
            api_key=api_key,
            download_dir=save_dir
        )
        
        print(f"[DOWNLOADED] {cur_str}")
        cur_date += timedelta(days=1)

if __name__ == "__main__":
    download_tardis_data(
        exchange=EXCHANGE,
        start_date=START_DATE,
        end_date=END_DATE,
        tardis_sym=TARDIS_TYPE,
        instrument=INSTRUMENT,
        save_dir=RAW_DATA_DIR
    )