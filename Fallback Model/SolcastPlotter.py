import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt

def process_and_plot_solar_data(folder_path):
    # Find all JSON files in the specified directory
    json_files = glob.glob(os.path.join(folder_path, "*.jsonl"))
    
    if not json_files:
        print(f"No JSON files found in directory: {folder_path}")
        return

    plt.figure(figsize=(14, 7))

    for file_path in json_files:
        filename = os.path.basename(file_path)
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON file: {filename}")
                continue

        # Each file contains a list of location objects
        for index, location in enumerate(data):
            lat = location.get("lat")
            lon = location.get("lon")
            time_series = location.get("data", [])

            if not time_series:
                continue

            # Convert location data to a DataFrame
            df = pd.DataFrame(time_series)
            
            # Parse timestamps to datetime objects
            df['period_end'] = pd.to_datetime(df['period_end'])

            # Label each line by file and coordinates/index for clarity
            label_prefix = f"{filename} [Loc {index+1} ({lat}, {lon})]"

            # Plot DNI and GHI
            plt.plot(df['period_end'], df['dni'], label=f"{label_prefix} - DNI", linestyle='--')
            plt.plot(df['period_end'], df['ghi'], label=f"{label_prefix} - GHI", linestyle='-')

    # Formatting the plot
    plt.title("Direct Normal Irradiance (DNI) & Global Horizontal Irradiance (GHI)")
    plt.xlabel("Timestamp (UTC)")
    plt.ylabel("Irradiance (W/m²)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Display plot
    plt.show()

if __name__ == "__main__":
    # Replace with the path to your folder containing the JSON files
    folder_directory = "Solar_real" 
    process_and_plot_solar_data(folder_directory)