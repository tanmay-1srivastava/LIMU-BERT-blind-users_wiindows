import os
import pandas as pd
import numpy as np



def down_sample(data, target_sr, curr_sr, seq_len=120):
    factor = int(curr_sr / target_sr)
    data = data[::factor]
    
    total_pad = seq_len - len(data)
    if total_pad % 2 == 0:
        pad_start = pad_end = total_pad // 2
    else:
        pad_start = total_pad // 2
        pad_end = total_pad - pad_start
    data = np.pad(data, ((pad_start, pad_end), (0, 0)), 'constant')
    
    return data[:seq_len, :]


target_sr = 20
seq_len = 120
curr_sr = 100




def preprocess(path, path_save, version, raw_sr=100, target_sr=20, seq_len=120):
    data = []
    labels = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if file.endswith('.csv') and 'sw' in file.lower():
                    df = pd.read_csv(file_path)
                    labels_ = df["label"]
                    transition_indices = labels_.index[labels_ != labels_.shift(1)]
                    transition_indices = transition_indices.tolist()
                    for i in range(len(transition_indices) - 1):
                        if(i == len(transition_indices) - 1):
                            start_idx = transition_indices[i]
                            end_idx = len(df)
                        else:
                            start_idx = transition_indices[i]
                            end_idx = transition_indices[i + 1]
                        df_temp = df.iloc[start_idx:end_idx, :6]
                        label_curr = labels_[start_idx]
                        print("label_curr", label_curr)
                        try:
                            df_temp = down_sample(df_temp, target_sr, curr_sr)
                            print("new shape", df_temp.shape)
                            print("-------------------")
                            user_idx = int(folder[1:])
                            print("user_idx", user_idx)
                            data.append(df_temp)
                            label = np.array([[user_idx, label_curr]])
                            label = np.tile(label, (seq_len, 1))
                            labels.append(label)
                        except:
                            print("error in file", file)
    labels = np.array(labels)
    last_column = labels[:, :, 1]
    unique_strings = np.unique(last_column)
    string_to_int = {string: i+1 for i, string in enumerate(unique_strings)}

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            string_value = last_column[i, j]
            labels[i, j, 1] = string_to_int[string_value]


    print("shape of labels", np.shape(labels)) 
    data = np.stack(data, axis=0)

    
    print("shape of data", np.shape(data))

    np.save(r"C:\Users\sbupi\Desktop\LIMU-BERT-blind-users_wiindows\dataset\smart_watch\data_20_120.npy", np.array(data))
    np.save(r"C:\Users\sbupi\Desktop\LIMU-BERT-blind-users_wiindows\dataset\smart_watch\label_20_120.npy", np.array(labels))
    return data, labels


path_save = r'blind_user'
version = r'20_120'
DATASET_PATH = r'C:\Users\sbupi\Downloads\sp-sw-har-dataset-v1.0.0-r1\GeoTecINIT-sp-sw-har-dataset-25bcf90\DATA'

data, label = preprocess(DATASET_PATH, path_save, version, target_sr=20, seq_len=120)
print("no. of unique labels", (np.unique(label[:,:,1])))

