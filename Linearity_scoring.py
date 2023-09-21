import numpy as np
import scipy.io as spio
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import csv
import os
import matplotlib.pyplot as plt
import time  # Importing the time module

# Constants and Parameters
contraction = 3 
ctp = 0.7 
time_band = contraction * ctp
time_band_half = round((0.5*time_band), 2) 
centers = np.array([4.5, 10.5, 16.5])
cent_samp = (centers*2000).astype(int)
plus_band = [round((i + time_band_half)*2000) for i in centers]
minus_band = [round((i-time_band_half)*2000) for i in centers]

def active_data(data):
    return [data[start:end] for start, end in zip(minus_band, plus_band)]

def apply_linear_regression_to_active_data(data, active_data_segments, channel):
    lr = LinearRegression()
    mse_values = []
    r2_values = []

    for idx, segment in enumerate(active_data_segments):
        X = np.arange(segment.shape[0]).reshape(-1, 1)
        y = segment[:, channel]
        lr.fit(X, y)
        predictions = lr.predict(X)
        
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        mse_values.append(mse)
        r2_values.append(r2)
        
        # Plotting the data
        #plt.plot(data[:, channel], color='lightgray', label='Full Channel Data' if idx == 0 else "")
        #plt.scatter(np.arange(minus_band[idx], plus_band[idx]), y, s=10, label='Active Data')
        #plt.plot(np.arange(minus_band[idx], plus_band[idx]), predictions, color='r', label=f'Best Fit Line\n$R^2 = {r2:.2f}$')
        #plt.xlabel('Time')
        #plt.ylabel('Amplitude')
        #plt.title(f'Channel {channel + 1}')
        #plt.legend()
        #plt.show()

    return mse_values, r2_values

def process_mat_file(mat_data):
    active_segments = active_data(mat_data)
    
    results = []
    for channel in range(8):
        mse_values, r2_values = apply_linear_regression_to_active_data(mat_data, active_segments, channel)
        avg_mse = np.mean(mse_values)
        avg_r2 = np.mean(r2_values)
        
        channel_results = {
            'channel': channel + 1,
            'mse_values': mse_values,
            'r2_values': r2_values,
            'avg_mse': avg_mse,
            'avg_r2': avg_r2
        }
        results.append(channel_results)
    
    return results

def append_to_csv(data, filename, identifier):
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            headers = ['Identifier', 'Channel', 'MSE_1', 'MSE_2', 'MSE_3', 'R2_1', 'R2_2', 'R2_3', 'Avg_MSE', 'Avg_R2']
            writer.writerow(headers)
    
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for channel_data in data:
            row = [
                identifier,
                channel_data['channel'],
                *channel_data['mse_values'],
                *channel_data['r2_values'],
                channel_data['avg_mse'],
                channel_data['avg_r2']
            ]
            writer.writerow(row)

def process_folder_structure(root_directory):
    output_csv_file = 'output.csv'
    
    for subdir, dirs, files in os.walk(root_directory):
        for file in files:
            if file.endswith('.mat'):
                mat_file_path = os.path.join(subdir, file)
                
                # Extract the relevant identifier from folder and filename
                parent_folder_name = os.path.basename(subdir)
                file_name_without_extension = os.path.splitext(file)[0]
                identifier = f"{parent_folder_name}_{file_name_without_extension}"
                
                data_raw = spio.loadmat(mat_file_path)
                data_matrix = data_raw['dataMatrix']
                
                results = process_mat_file(data_matrix)
                
                append_to_csv(results, output_csv_file, identifier)
                print(f"Processed {identifier} and saved to {output_csv_file}")

def main():
    start_time = time.time()  # Record the start time

    # Root directory containing all the participant folders
    root_directory = r''
    
    process_folder_structure(root_directory)
    
    end_time = time.time()  # Record the end time
    elapsed_time = end_time - start_time
    print(f"Time taken to complete: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
