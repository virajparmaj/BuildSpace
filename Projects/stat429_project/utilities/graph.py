import matplotlib.pyplot as plt
import pandas as pd

def plot_funding_rate(df):
    # Make a copy of the DataFrame and convert the timestamp
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us', utc=True)
    df = df.set_index('timestamp')

    # Plot the funding rate, scaled for better visualization
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['funding_rate'] * 10_000, label='Funding Rate (bips)', color='blue')
    plt.xticks(rotation=45)
    plt.xlabel("Timestamp")
    plt.ylabel("Funding Rate (bips)")
    plt.title("Funding Rate vs. Time")
    plt.legend()
    plt.tight_layout()  # Adjust layout to avoid clipping
    plt.show()