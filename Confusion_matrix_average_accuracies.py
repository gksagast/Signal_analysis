import scipy.io as spio
import numpy as np
import os
import glob
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
import json
import pandas as pd
import matplotlib.pyplot as plt


contraction = 3 #The contraction time is 3 seconds
ctp = 0.7 #The time period we want to look at is 70% of the data
time_band = contraction * ctp #the time period is how much data of grasping data we will collect
time_band_half = round((0.5*time_band), 2) #This is going to be half the time band which will be used to find where the centers are
centers = np.array([4.5, 10.5, 16.5]) #The centers of the data. These are currently hardcoded in and set to be this
cent_samp = (centers*2000).astype(int) #This turns time into samples
plus_band = []
#This for loop and the next finds the time for the samples at both the high and low end of what we want to collect
for i in centers:
    plus_band.append(round((i + time_band_half)*2000))
minus_band = []
for i in centers:
    minus_band.append(round((i-time_band_half)*2000))
    
    
#Collect only the active data
def active_data(data):
    active_data = []
    #This for loop within a for loop collects data only during when the person was contracting
    for j in range(len(plus_band)):
        active_data.append(data[minus_band[j]:plus_band[j]])
    return active_data

#Put active data into one array
def only_one_act(almost_active):
    one_active = []
    active_data = almost_active[0]
    for j in range(1,3):
        active_data = np.concatenate((active_data, almost_active[j]))
    one_active.append(active_data)
    return one_active  # Return the one_active list


#Segmenting
def segmenting(one_data):
    segmented = []
    #print(len(one_data[0]))
    for i in range(len(one_data)):
        tinc = 100 #50 ms turned into samples
        bin_size = 400 #200 ms turned into samples
        end = 400
        start = 0
        segmented_inner = []
        #print(len(one_data[i]))
        while end <= len(one_data[i]):
            window = one_data[i][start:end]
            #print(window)
            segmented_inner.append(window)
            #print(segmented_inner)
            start = start + tinc
            end = start + bin_size
        segmented.append(segmented_inner)
        #print(segmented)
    return segmented

#Put in the right format so that channel 1 has all its windows and so does channel 2 and so forth
#rff stands for Ready for Features
def rff(segmented):
    rff = []
    for i in range(len(segmented)):
        ch1 = []
        ch2 = []
        ch3 = []
        ch4 = []
        ch5 = []
        ch6 = []
        ch7 = []
        ch8 = []
        for j in range(len(segmented[i])):
            ch1.append(segmented[i][j][:,0])
            ch2.append(segmented[i][j][:,1])
            ch3.append(segmented[i][j][:,2])
            ch4.append(segmented[i][j][:,3])
            ch5.append(segmented[i][j][:,4])
            ch6.append(segmented[i][j][:,5])
            ch7.append(segmented[i][j][:,6])
            ch8.append(segmented[i][j][:,7])
        feature_temp =[ch1, ch2, ch3, ch4, ch5, ch6, ch7, ch8]
        feature_temp_t = np.transpose(feature_temp, (1,0,2))
        rff.append(feature_temp_t)
    return rff

def MAV(window):
    temp_array = np.array(window)
    MAV = np.sum(abs(temp_array))/len(window)
    return MAV

def feature(rff):
    feature = []
    for i in range(len(rff)):
        fuck = []
        for j in range(len(rff[i])):
            fuck_but_inside = list([])
            for channel in rff[i][j]:
                dumb_math_list = list([MAV(channel)])
                fuck_but_inside = fuck_but_inside + dumb_math_list
            fuck.append(fuck_but_inside)
        feature.append(fuck)
    return feature

def grounded(data):
    ground_truth = []
    current = 0
    for i in range(len(data)):
        if i%123 == 0:
            current = current+1
            ground_truth.append(current)
        else:
            ground_truth.append(current)
    return ground_truth

def confusion_matrix(ground_truth, predicted):
    #Calculate confusion matrix
    from sklearn.metrics import confusion_matrix, accuracy_score
    result_confusion = confusion_matrix(ground_truth, predicted)
    
    return result_confusion

def accuracy(result_confusion):
    total = 0
    for i in range(len(result_confusion)):
        total = total+result_confusion[i][i]
    accuracy = total/sum(sum(result_confusion))
    return accuracy

def append_to_excel(acc, directory, output_directory, filename="output.xlsx"):
    # Extract the last two parts of the directory path
    new_dir_1 = os.path.basename(directory)
    parent_dir = os.path.dirname(directory)
    new_dir_2 = os.path.basename(parent_dir)
    
    # Concatenate the two parts to form new_dir
    new_dir = f"{new_dir_2}_{new_dir_1}"
    
    # Create the full path to the Excel file
    full_excel_path = os.path.join(output_directory, filename)
    
    # Check if the Excel file exists
    if os.path.exists(full_excel_path):
        # Load the existing Excel file
        df = pd.read_excel(full_excel_path, engine='openpyxl')
    else:
        # Create a new DataFrame if the Excel file doesn't exist
        df = pd.DataFrame(columns=['Person/Grasp', 'Accuracy'])

    # Append the new data
    new_data = pd.DataFrame({'Person/Grasp': [new_dir], 'Accuracy': [acc]})
    df = pd.concat([df, new_data], ignore_index=True)
    
    # Save the updated DataFrame back to the Excel file
    df.to_excel(full_excel_path, index=False, engine='openpyxl')


def append_to_csv(acc, directory, output_directory, filename="output.csv"):
    # Extract the last two parts of the directory path
    new_dir_1 = os.path.basename(directory)
    parent_dir = os.path.dirname(directory)
    new_dir_2 = os.path.basename(parent_dir)
    
    # Concatenate the two parts to form new_dir
    new_dir = f"{new_dir_2}_{new_dir_1}"
    
    # Create the full path to the CSV file
    full_csv_path = os.path.join(output_directory, filename)
    
    # Check if the CSV file exists
    if os.path.exists(full_csv_path):
        # Load the existing CSV file
        df = pd.read_csv(full_csv_path)
    else:
        # Create a new DataFrame if the CSV file doesn't exist
        df = pd.DataFrame(columns=['Person/Grasp', 'Accuracy'])

    # Append the new data
    new_data = pd.DataFrame({'Person/Grasp': [new_dir], 'Accuracy': [acc]})
    df = pd.concat([df, new_data], ignore_index=True)
    
    # Save the updated DataFrame back to the CSV file
    df.to_csv(full_csv_path, index=False)
    
    
# Sample usage
#data_directory = "C:\\Users\\Carlo\\Box\\00_BEAR Lab\\Projects\\Giancarlo Sagastume\\Testing_data\\Good_tests\\Par_2\\PY_TIA_2\\10grasp_PY_TIA_2\\WR"
#output_directory = "C:\\Path\\To\\Output\\Directory"  # Replace with your desired output directory
#acc = 0.95

#append_to_excel(acc, data_directory, output_directory)  # This will save the Excel file in the specified output directory#


# Define the root directory where participant folders are located
root_directory = r''

# Output directory for saving results
output_directory = r''

# Create an empty dictionary to store participant results
participant_results = {}

# Iterate through participant directories
for participant_folder in os.listdir(root_directory):
    participant_directory = os.path.join(root_directory, participant_folder)
    
    # Check if the item in the root directory is a directory
    if os.path.isdir(participant_directory):
        participant_data = {}
        
        # Iterate through subdirectories (XX_TIA_1, XX_UGA_1, XX_VD_1, etc.)
        for grasp_subfolder in os.listdir(participant_directory):
            grasp_subfolder_path = os.path.join(participant_directory, grasp_subfolder)
            
            # Check if the item in the subdirectory is a directory
            if os.path.isdir(grasp_subfolder_path):
                all_data = {}
                
                # Iterate through .mat files in the subdirectory
                for filename in glob.glob(os.path.join(grasp_subfolder_path, "*.mat")):
                    data_raw = spio.loadmat(filename)
                    data = data_raw['dataMatrix']
                    grasp_name = os.path.basename(filename).split('.')[0]
                    
                    data_temp = {
                        grasp_name: data
                    }
                    all_data = {**all_data, **data_temp}
                
                confusion_ready = []
                for grasp in all_data:
                    #print(all_data[grasp])
                    active_broken = active_data(all_data[grasp])
                    #print(active_broken)
                    active = only_one_act(active_broken)
                    segmented = segmenting(active)
                    ready_for_features = rff(segmented)
                    feature_vector = feature(ready_for_features)
                    confusion_ready = list(confusion_ready) + list(feature_vector[0])
                print(len(confusion_ready))

                ground_truth = grounded(confusion_ready)
                clf = LinearDiscriminantAnalysis()
                predicted = []
                for i in range(len(confusion_ready)):
                    data_sub = np.delete(confusion_ready, i, 0)
                    ground_sub = np.delete(ground_truth, i, 0)
                    the_forgotten = confusion_ready[i]
                    #print(i)
                    clf.fit(data_sub, ground_sub)
                    pred = int(clf.predict([the_forgotten]))
                    predicted.append(pred)
                
                conf_mat = confusion_matrix(ground_truth, predicted)
                #print(conf_mat)
                conf_mat_percent = np.around(conf_mat/123, 4)
                conf_mat_percent = np.around((conf_mat/123)*100, 2)
                print(conf_mat_percent)

                acc1 = accuracy(conf_mat)
                print(acc1)
                d1 = r''

                append_to_csv(acc1, filename, d1, filename="CF_accuracies.csv")
                keys = ['CR', 'CW', 'IF', 'IP', 'KP', 'PP', 'TP', 'WE', 'WF', 'WR']
               # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_mat_percent, display_labels = keys)
                #cm_display.plot(values_format='')
                #fig = cm_display.ax_.get_figure() 
                #fig.set_figwidth(15)
                #fig.set_figheight(15)
                #plt.show()
                
                Par1 = {'GC': 
                        {'Grasp Data': all_data, 'Confusion Matrix': conf_mat_percent, 'Accuracy': acc1}}
                
                print(Par1)
                
              #  class NpEncoder(json.JSONEncoder):
             #       def default(self, obj):
            #            if isinstance(obj, np.integer):
           #                 return int(obj)
          #              if isinstance(obj, np.floating):
         #                   return float(obj)
        #                if isinstance(obj, np.ndarray):
       #                     return obj.tolist()
      #                  return super(NpEncoder, self).default(obj)

                #Switch the name here
                #with open('GC_VD.json', 'w') as f_out:
                 #   json.dump(Par1,f_out, cls=NpEncoder)
                
                # Save results for this grasp subfolder
     #           participant_data[grasp_subfolder] = {
    #                'Grasp Data': all_data,
   #                 'Confusion Matrix': conf_mat_percent,
  #                  'Accuracy': acc1
 #               }
        
        # Save participant results
        participant_results[participant_folder] = participant_data

# Save participant results to JSON file
#with open('participant_results.json', 'w') as f_out:
#    json.dump(participant_results, f_out, cls=NpEncoder)