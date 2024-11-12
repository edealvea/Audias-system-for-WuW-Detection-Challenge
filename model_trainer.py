import pandas as pd
from tqdm import tqdm
import torch
import numpy as np
import os 
from torch import nn
import torchaudio
import torchaudio.functional as F
import argparse
from model_ResNet_2outs import *
import random
import csv

#import matplotlib.pyplot as plt

class ModelTrainer():
    """
    Class to train the model.
    Change this class to match the model you are using.
    """

    def __init__(self, args):
        self.device = args.device
        self.model = ResNet2MFCC(args).to(torch.device(args.device))
        self.threshold = args.threshold

    def warmup(self, window_size, n_warmup_samples=100):
        """
        Warmup the model with a given audio sample.
        """
        batch_size = 1
        for _ in tqdm(range(n_warmup_samples), desc="Warmup"):
            x = torch.randn(batch_size, window_size).to(torch.device(self.device))
            self.process_audio(x)

    def load_model(self, model_path):
        """
        Load a jit model from a given path.
        """
        model = torch.jit.load(model_path, map_location=torch.device(self.device))
        return model

    def preprocess(self, audio):
        """
        Preprocess audio to match the model input: peak normalization.
        """
        processed_audio = audio / torch.max(torch.abs(audio))
        return processed_audio

    def inference(self, audio):
        """
        Perform inference on a given audio sample: forward pass.
        """
        #with torch.no_grad():
        output = self.model(audio)
        return output

    def postprocess(self, output):
        """
        Postprocess model output to match the model output.

        Cambiar para mi modelo ya que queremos que devuelva 0 1 o 2

        Además para que nos devuelva la salida del batch entero no solo un 1
        Como hace [0][0] si hay más elementos en el batch no coge los de ese
        por lo que habría que hacer para que devolviese 1 en todos pero por el resto parece que hace bien el entrenamiento
        also habría que adaptarlo a mi tipo de modelo
        """
        wuw_probability = output
        #wuw_probability = F.softmax(output, dim=1) 
        #Esto está mal hay que cambiarlo devuelve solo con la probabilidad del primero
        predicted_label = torch.tensor([torch.argmax(e) if torch.max(e) > self.threshold else 0.0 for e in wuw_probability], dtype=torch.float32).to(torch.device(self.device))

        return wuw_probability, predicted_label
        

    def process_audio(self, audio ):
        """
        Process audio from a given path.
        """
        preprocessed = self.preprocess(audio)
        output = self.inference(preprocessed)
        return self.postprocess(output)
    
    def process_train(self, audio ):
        preprocessed = self.preprocess(audio)
        return self.inference(preprocessed)

    def train(self, args):
        clips_path = os.path.join(os.path.join(args.dataset_path), 'clips')
        # Audio configuration
        window_size = int(args.time_window * args.sampling_rate)
        hop = int(args.hop * args.sampling_rate)

        # Model
        #self.warmup(window_size)

        # Train data
        df = pd.read_csv(args.train_tsv, header=0, sep='\t')#Quitar el 3, esto es solo para probar
        noise_df = pd.read_csv(args.noise, header = 0, sep="\t")
        #Loading audios and preparing data

        audios, true_labs = get_feats_and_labs(df , clips_path, hop, window_size, args.sampling_rate)

        noise, _ = get_feats_and_labs(noise_df, args.noise_path + r"\train", hop, window_size, args.sampling_rate)

        audios, true_labs = augment_data(audios, true_labs, noise)

        audios = torch.cat(audios)



        true_labs_pos_index, true_labs_neg_index = splitter(true_labs)
        
        true_labs = torch.tensor(true_labs, dtype=torch.long)

        
        batches = get_balanced_batches(true_labs_pos_index, true_labs_neg_index, args.batch_size, len(audios) / args.batch_size, equil_fact = 0.2)
        #batches = np.array_split(np.arange(len(audios)), len(audios) / args.batch_size)

        #Validation data
        df_val = pd.read_csv(args.validation_tsv, header = 0, sep='\t')
        df_noise_val = pd.read_csv(args.noise_val, header = 0, sep="\t")
        auds_noise_val,_ = get_feats_and_labs(df_noise_val , args.noise_path + r"\dev", hop, window_size, args.sampling_rate)
        audios_val, true_labs_val = get_feats_and_labs(df_val , clips_path, hop, window_size, args.sampling_rate)
        audios_val, true_labs_val = augment_data(audios_val, true_labs_val, auds_noise_val)
        
        audios_val = torch.cat(audios_val)
        true_labs_val = torch.tensor(true_labs_val, dtype=torch.long)

        
        batches_val = np.array_split(np.arange(len(audios_val)), len(audios_val) / args.batch_size )


        #Define loss and optimizer
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = args.learning_rate)

        early_stopping = Callback(path = args.output_dir+"\ ".replace(" ","")+args.model_name+".pt",patience=5, min_delta=0.001)#Poner ese delta para que los valores altos no contaminen tanto la muestra

        for epoch in range(args.num_epochs):
            #Training phase
            self.model.train()
            cache_loss = 0 

            for batch in tqdm(batches, total=len(batches), ncols=60, desc=f"Training epoch {epoch}"):
                #print(len(batch))
                #print(true_labs[batch])
                optimizer.zero_grad()
                #raise Exception("Para")
                #Loading audio labels and features for the batch
                true_labs_batch = true_labs[batch].to(torch.device(self.device))
                #true_outs_batch = true_outs[batch].to(torch.device(self.device))
                audios_batch = audios[batch].to(torch.device(self.device))
                #Audio processing
                wuw_probs = self.process_train(audios_batch)

                #Loss and backpropagation
                loss = loss_function(wuw_probs, true_labs_batch)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                loss.backward() 
                cache_loss += loss.item() * len(batch)
                optimizer.step()
                # if (true_labs_batch >0).any():
                #    print(F.softmax(wuw_probs))
                #    print(true_labs_batch)

            # Validation phase
            self.model.eval()
            cache_loss_val = 0
            batches_sin = 0
            for batch in tqdm(batches_val, total=len(batches_val), ncols=60, desc=f"Validation in epoch {epoch}"):
                audios_val_batch = audios_val[batch].to(torch.device(self.device))
                true_labs_val_batch = true_labs_val[batch].to(torch.device(self.device))
                
                outputs = self.process_train(audios_val_batch )
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    #print("Aquí pasa algo raro:", outputs)
                    batches_sin += len(batch)
                #outputs+=1e-10
                else:

                    loss = loss_function(outputs, true_labs_val_batch)
                    cache_loss_val += loss.item() * len(batch)

            print("Loss: ", cache_loss / len(audios))
            print("Val Loss: ", cache_loss_val / (len(audios_val) - batches_sin))
            early_stopping(cache_loss_val / (len(audios_val) - batches_sin), self.model)
            if early_stopping.early_stop and epoch > 10:
                print(f"Training stopped in epoch {epoch} by the earlystopper")
                break


    """
    Modificar la siguiente función para adaptarla al training
    """
    def results(self, args):
        self.model.eval()
        if (os.path.isfile(args.model_path) or args.model_path == "None") and os.path.isfile(args.test_tsv) and os.path.isdir(os.path.dirname(args.output_dir)):
            clips_path = os.path.join(os.path.join(args.dataset_path), 'clips')
            
            # Audio configuration
            window_size = int(args.time_window * args.sampling_rate)
            hop = int(args.hop * args.sampling_rate)

            #Model
            #model_handler = ModelTrainer(args)
            #model_handler.warmup(window_size)

            # Test data
            df = pd.read_csv(args.test_tsv, header=0, sep='\t')
            results_list = []
            with torch.no_grad():
                self.model.to(torch.device(self.device))
                for _, row in tqdm(df.iterrows(), total=df.shape[0], ncols=60, desc="Streaming test"):
                    filename = row['Filename']
                    audio_path = os.path.join(clips_path, filename)
                    true_lab = row['Label']
                    sim = row["Similarity"]
                    true_start  = row["Start_Time"]
                    true_end = row["End_Time"]
                    predicted_probs = []
                    predicted_labels = []

                    # Load audio
                    info = torchaudio.info(audio_path)
                    steps = int((info.num_frames - window_size) / hop) + 1

                    
                    for i in range(steps):
                        start = int(i*hop)
                        audio, sr = torchaudio.load(
                            audio_path,
                            frame_offset=start,
                            num_frames=window_size,
                            )
                        
                        if(sr != args.sampling_rate):
                            raise Exception("Sampling rate mismatch")
                        audio = audio.to(torch.device(self.device))
                        wuw_p, wuw_label = self.process_audio(audio)
                        if torch.isnan(wuw_p).any():
                            print("Aquí pasa algo raro:", wuw_p)
                        predicted_probs.append(wuw_p)
                        predicted_labels.append(wuw_label)
                    #print(predicted_probs)
                    # Calculate results per sample
                    prob, label, start, end = calculate_results_outputs(
                        predicted_probs,
                        predicted_labels,
                        window_size,
                        hop,
                        args.n_positives,
                        args.sampling_rate
                        )

                    results_list.append([filename, prob, label, true_lab,sim, start, end, true_start, true_end])

            columns = ["Filename", "Probability", "Label_predicted", "Label","Similarity", "Start_Time", "End_Time", "True_Start_Time", "True_End_Time"]
            results_df = pd.DataFrame(results_list, columns=columns)
            results_df.to_csv(os.path.join(args.output_dir, args.sysid  + "_test.tsv"), index=None, sep="\t")
            return results_df
        else:
            raise Exception("Bad arguments for streaming test. Some of the provided paths are not valid.")
    
    def save_model(self, save_path):
        scripted_model = torch.jit.script(self.model)
        scripted_model.save(save_path)

    
    def results_extended(self, args):
        self.model.eval()
        if (os.path.isfile(args.model_path) or args.model_path == "None") and os.path.isfile(args.extended_test_tsv) and os.path.isdir(os.path.dirname(args.output_dir)):
            clips_path = os.path.join(os.path.join(args.extended_dataset_path), 'clips')
            
            # Audio configuration
            window_size = int(args.time_window * args.sampling_rate)
            hop = int(args.hop * args.sampling_rate)

            #Model
            #model_handler = ModelTrainer(args)
            #model_handler.warmup(window_size)

            # Test data
            df = pd.read_csv(args.extended_test_tsv, header=0, sep='\t')
            results_list = []
            with torch.no_grad():
                self.model.to(torch.device(self.device))
                for _, row in tqdm(df.iterrows(), total=df.shape[0], ncols=60, desc="Streaming extended test"):
                    filename = row['Sample_ID']
                    audio_path = os.path.join(clips_path, "extended_test_"+filename+".wav")
                    true_lab = row['Label']
                    #sim = row["Similarity"]
                    #true_start  = row["Start_Time"]
                    #true_end = row["End_Time"]
                    predicted_probs = []
                    predicted_labels = []

                    # Load audio
                    info = torchaudio.info(audio_path)
                    steps = int((info.num_frames - window_size) / hop) + 1

                    
                    for i in range(steps):
                        start = int(i*hop)
                        audio, sr = torchaudio.load(
                            audio_path,
                            frame_offset=start,
                            num_frames=window_size,
                            )
                        
                        if(sr != args.sampling_rate):
                            raise Exception("Sampling rate mismatch")
                        audio = audio.to(torch.device(self.device))
                        wuw_p, wuw_label = self.process_audio(audio)
                        if torch.isnan(wuw_p).any():
                            print("Aquí pasa algo raro:", wuw_p)
                        predicted_probs.append(wuw_p)
                        predicted_labels.append(wuw_label)
                    #print(predicted_probs)
                    # Calculate results per sample
                    prob, label, start, end = calculate_results_outputs(
                        predicted_probs,
                        predicted_labels,
                        window_size,
                        hop,
                        args.n_positives,
                        args.sampling_rate
                        )

                    results_list.append([filename, prob, label,true_lab, start, end])

            columns = ["Filename", "Probability", "Label_predicted", "Label", "Start_Time", "End_Time"]
            results_df = pd.DataFrame(results_list, columns=columns)
            results_df.to_csv(os.path.join(args.output_dir, args.sysid  + "_extended_test.tsv"), index=None, sep="\t")
            return results_df
        else:
            raise Exception("Bad arguments for streaming test. Some of the provided paths are not valid.")

def get_feats_and_labs(df, clips_path, hop, window_size, sampling_rate):
    audios = []
    labs = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], ncols=60, desc="Loading data"):
        filename = row['Filename']
        audio_path = os.path.join(clips_path, filename)
        # Load audio
        info = torchaudio.info(audio_path)
        steps = int((info.num_frames - window_size) / hop) + 1
        audios += get_feat(audio_path, steps, hop, window_size, sampling_rate)
        labs += get_lab(row, steps, hop, sampling_rate, window_size/sampling_rate)
        #print(row["Label"],row["Start_Time"],row["End_Time"],"\n",  labs)

    return audios, labs


def get_feat(audio_path, steps, hop, window_size, sampling_rate):
    audios = []
    for i in range(steps):
        start = int(i*hop)
        audio, sr = torchaudio.load(
            audio_path,
            frame_offset=start,
            num_frames=window_size,
            )
        if(sr != sampling_rate):
            raise Exception("Sampling rate mismatch")
        current_length = audio.size(1)

        if current_length < window_size:
            padding_length = window_size - current_length
            audio = torch.nn.functional.pad(audio, (0, padding_length), 'constant', 0)
        audios.append(audio)
    return audios

# def get_lab(row, steps, hop, sample_rate, window_size):
#     label = []
#     if row["Label"] == "WuW":
#         start_time = float(row["Start_Time"])
#         end_time = float(row["End_Time"])
#         for i in range(steps):
#             start = i * hop / sample_rate
#             #print(start, start_time, window_size / sample_rate)
#             if start  <= start_time and start + window_size > end_time: #Si ocupa toda la ventana
#                 label.append(1)
#             elif start >=start_time and start - hop < start_time:
#                 label.append(1)
#             elif start >=start_time and start + window_size < end_time + hop:
#                 label.append(1)
#             else:
#                 label.append(0)
#     else:
#         for i in range(steps):
#                 label.append(0)
#     return label    

def get_lab(row, steps, hop, sample_rate, window_size):
    label = []
    if row["Label"] == "WuW":
        start_time = float(row["Start_Time"])
        end_time = float(row["End_Time"])
        for i in range(steps):
            start = i * hop / sample_rate
            #print(start, start_time, window_size / sample_rate)
            if start  <= start_time and start > 0.85*(end_time - start_time) + start_time: #Si ocupa la mayor parte de la ventana el 85% salvo algo del final
                label.append(1)
            elif start >=start_time and start <  end_time - 0.85*(end_time - start_time):#Si ocupa la mayor parte de la ventana el 85% salvo algo del principio
                label.append(1)
            elif start >=start_time and start + window_size < end_time + hop: #Si está en la ventana
                label.append(1)
            else:
                label.append(0)
    else:
        for i in range(steps):
                label.append(0)
    return label    

def get_output(labels):
    outputs = []
    for lab in labels:
        if lab == 0:
            outputs.append([1,0])
        elif lab == 1:
            outputs.append([0,1])
        else:
            raise Exception("Error in the labels")
    return outputs


"""
    Modificar la siguiente función para adaptarla al training
"""
def calculate_results_outputs(probabilities, labels, window_size, hop, n_positives, sample_rate):
    """
    Calculate the final results of the model outputs.
    Here we should implement the logic to:
    - decide if a sample is positive or negative
    - calculate the probability of the whole sample to be positive
    - calculate the start and end time of the positive samples
    """
    detection, start_idx = look_detection_pattern(labels, n_positives)
    if detection:
        end_idx = start_idx + n_positives 
        prob = sum(probabilities[start_idx:start_idx + n_positives]) / n_positives
        start_time = (start_idx * hop)/sample_rate
        end_time = (end_idx * hop + window_size)/sample_rate
    else:
        prob = sum(probabilities) / len(probabilities)
        start_time = "Unknown"
        end_time = "Unknown"
    return prob, detection, start_time, end_time
    
def look_detection_pattern(labels, n_positives):
    """
    Look for a detection pattern in the labels.
    """
    sub_lst = [1]*(n_positives)
    len_sub_lst = len(sub_lst)
    for i in range(len(labels)):
        if sub_lst == labels[i:i+len_sub_lst]:
            return 1, i
    return 0, -1

def calculate_metrics(result_df, ref_df):
    """
    Calculate the Detect Cost Function (DCF) for the system.
    """
    cost_model = {
        'p_wuw' : 0.5,
        'c_miss' : 1,
        'c_fa' : 1.5
    }

    # Calculate misses

    #N_miss is number of WuW that the model has predicted as 0
    n_miss = result_df[(result_df["Label"] == "WuW") & (result_df["Label_predicted"] == 0) ].shape[0]

    #N_fa is number of non_WuW that the model has predicted as 1
    n_fa = result_df[(result_df["Label"] == "NonWuW") & (result_df["Label_predicted"] == 1) ].shape[0]

    #Calculate both probabilities
    p_miss = n_miss / result_df[result_df["Label"] == "WuW"].shape[0]
    p_fa = n_fa / result_df[result_df["Label"] == "NonWuW"].shape[0]

    #return the DFC
    #return Pwuw * Cmiss * Pmiss + (1- Pwuw) * Cfa * Pfa
    print(f"Misses: {n_miss}")
    print(f"Miss Rate: {p_miss}")
    print(f"False Positives: {n_fa}")
    print(f"False Alarm Rate: {p_fa}")

    print(f'Number of samples: {len(result_df)}')
    
    # calculate the cost of false positives and false negatives
    dcf = cost_model['c_miss'] * p_miss * cost_model["p_wuw"] + cost_model['c_fa'] * p_fa * (1- cost_model["p_wuw"])
    #Calculate the TEM
    res_copy = result_df[(result_df["Label"] == "WuW") & (result_df["Label_predicted"] == 1)]
    TEs = np.abs( np.array(res_copy["Start_Time"]) - np.array(res_copy["True_Start_Time"]))
    TEe = np.abs( np.array(res_copy["End_Time"]) - np.array(res_copy["True_End_Time"]))
    tem = np.median(TEs + TEe)
    
    # build a json with the results
    results = {
        'misses': n_miss,
        'false_positives': n_fa,
        'n_samples': len(ref_df),
        'p_miss': p_miss,
        'p_fa': p_fa,
        'dcf': dcf,
        'TEM':tem
    }

    return results

def calculate_metrics_ext(result_df, ref_df):
    """
    Calculate the Detect Cost Function (DCF) for the system.
    """
    cost_model = {
        'p_wuw' : 0.5,
        'c_miss' : 1,
        'c_fa' : 1.5
    }

    # Calculate misses

    #N_miss is number of WuW that the model has predicted as 0
    n_miss = result_df[(result_df["Label"] == "WuW") & (result_df["Label_predicted"] == 0) ].shape[0]

    #N_fa is number of non_WuW that the model has predicted as 1
    n_fa = result_df[(result_df["Label"] == "unknown") & (result_df["Label_predicted"] == 1) ].shape[0]

    #Calculate both probabilities
    p_miss = n_miss / result_df[result_df["Label"] == "WuW"].shape[0]
    p_fa = n_fa / result_df[result_df["Label"] == "unknown"].shape[0]

    #return the DFC
    #return Pwuw * Cmiss * Pmiss + (1- Pwuw) * Cfa * Pfa
    print(f"Misses: {n_miss}")
    print(f"Miss Rate: {p_miss}")
    print(f"False Positives: {n_fa}")
    print(f"False Alarm Rate: {p_fa}")

    print(f'Number of samples: {len(result_df)}')
    
    # calculate the cost of false positives and false negatives
    dcf = cost_model['c_miss'] * p_miss * cost_model["p_wuw"] + cost_model['c_fa'] * p_fa * (1- cost_model["p_wuw"])

    # build a json with the results
    results = {
        'misses': n_miss,
        'false_positives': n_fa,
        'n_samples': len(ref_df),
        'p_miss': p_miss,
        'p_fa': p_fa,
        'dcf': dcf
    }

    return results


def train_model(args):

    if (os.path.isfile(args.model_path) or args.model_path == "None") and os.path.isfile(args.train_tsv) and os.path.isdir(os.path.dirname(args.output_dir)):
        mt = ModelTrainer(args)
        mt.train(args)
        probar_modelo(args)
    else:
        raise Exception("Bad arguments for streaming test. Some of the provided paths are not valid.")
    
def probar_modelo(args):
    modelo = ModelTrainer(args)
    state_dict = torch.load(args.output_dir+"\ ".replace(" ","")+args.model_name+".pt", weights_only=True)
    modelo.model.load_state_dict(state_dict)
    results_df = modelo.results(args)
    ref_df = pd.read_csv(args.test_tsv, header = 0, sep='\t')  
    dcf = calculate_metrics(results_df, ref_df)

    print(f"DCF for the test of this model: {dcf}")
    
    results_extended = modelo.results_extended(args)
    ref_df = pd.read_csv(args.test_tsv, header = 0, sep = "\t")
    dcf_ext = calculate_metrics_ext(results_extended, ref_df)
    print(f"DCF for the extended test of this model: {dcf_ext}")
    write_tsv(args, dcf, dcf_ext)



def write_tsv(args, dcf, dcf_ext):
    row = [args.model_name, args.hop, args.time_window, args.threshold, dcf["dcf"], dcf_ext["dcf"], 
           dcf["p_miss"], dcf_ext["p_miss"], dcf["p_fa"], dcf_ext["p_fa"], dcf["TEM"]]
    with open("Results.tsv", "a", newline="") as tsv:
        writer = csv.writer(tsv, delimiter="\t")
        writer.writerow(row)


def splitter(true_labs):
    #Im going to split the labels to equilbrate each batch
    true_labs_pos_index = []
    true_labs_neg_index = []
    for i in range(len(true_labs)):
        if true_labs[i] == 1:
            true_labs_pos_index.append(i)
        elif true_labs[i] == 0:
            true_labs_neg_index.append(i)
        else:
            print(f"Skipped index {i} because has incorrect labeling")
    return true_labs_pos_index, true_labs_neg_index

# def augment_data(audios, labs, noise, rir = torchaudio.load("Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo.wav")):
#     labs_aug = []
#     audios_aug = []
#     rir_a = rir[0][:, int(rir[1] * 1.01) : int(rir[1] * 1.3)]
#     rir_a = rir_a / torch.linalg.vector_norm(rir_a, ord=2)
#     rir_a =torchaudio.transforms.Resample(orig_freq=rir[1], new_freq=16000)(rir_a)

#     for i in range(len(labs)):
#         lab = labs[i]
#         audio = audios[i]

#         if random.random() < 0.75:
#             vol_down = audio * random.random() 
            
#             noised = audio * (1+random.random()) 
#             noised2 = torchaudio.functional.add_noise(audio, random.choice(noise),torch.tensor([0]))

#             noised_up = audio * (1+random.random())  +  0.5* random.choice(noise) + 0.5 * random.choice(noise)

#             noised_rev = torchaudio.functional.fftconvolve(audio, rir_a, mode="same")[:,:24000]

#             noised_up3 = torchaudio.functional.add_noise(audio, random.choice(noise),torch.tensor([10]))

#             noised_down = torchaudio.functional.add_noise(audio, random.choice(noise),torch.tensor([20]))

            
#             #noised_down2 = audio * vold + random.choice(noise)
#             #noised_down3 = audio * vold + random.choice(noise)

#             labs_aug += [lab, lab, lab, lab, lab, lab, lab, lab]
#             audios_aug += [audio, vol_down, noised, noised_up,noised_rev, noised_down, noised_up3, noised2]
#         else:
#             labs_aug += [lab]
#             audios_aug += [audio]
#     assert len(audios_aug) == len(labs_aug)
#     return audios_aug, labs_aug

def get_rir(rir_path):
    lista_doc = os.listdir(rir_path)
    lista_rir = [doc for doc in lista_doc if doc.endswith(".wav") ]
    auds = []
    for file in lista_rir:
        rir = torchaudio.load(rir_path + "\ ".replace(" ","") + file)
        rir_a = rir[0][:, int(rir[1] * 1.01) : int(rir[1] * 1.3)]
        rir_a = rir_a / torch.linalg.vector_norm(rir_a, ord=2)
        rir_a =torchaudio.transforms.Resample(orig_freq=rir[1], new_freq=16000)(rir_a)
        auds.append(rir_a)
    return auds


def augment_data(audios, labs, noise, rir_path = r"room-response\impulse"):
    labs_aug = []
    audios_aug = []
    rir_auds = get_rir(rir_path)

    for i in range(len(labs)):
        lab = labs[i]
        audio = audios[i]

        if random.random() < 0.75:
            rir_a = random.choice(rir_auds)

            vol_down = audio * random.random() 
            
            noised = audio * (1+random.random()) 
            noised2 = torchaudio.functional.add_noise(audio, random.choice(noise),torch.tensor([0]))

            noised_up = audio * (1+random.random())  +  0.5* random.choice(noise) + 0.5 * random.choice(noise)

            noised_rev = torchaudio.functional.fftconvolve(audio, rir_a, mode="same")[:,:24000]

            noised_up3 = torchaudio.functional.add_noise(audio, random.choice(noise),torch.tensor([10]))

            noised_down = torchaudio.functional.add_noise(audio, random.choice(noise),torch.tensor([20]))

            
            #noised_down2 = audio * vold + random.choice(noise)
            #noised_down3 = audio * vold + random.choice(noise)

            labs_aug += [lab, lab, lab, lab, lab, lab, lab, lab]
            audios_aug += [audio, vol_down, noised, noised_up,noised_rev, noised_down, noised_up3, noised2]
        else:
            labs_aug += [lab]
            audios_aug += [audio]
    assert len(audios_aug) == len(labs_aug)
    return audios_aug, labs_aug

def get_balanced_batches(positives, negatives, batch_size, num_batches, equil_fact = 0.5):
    batches = []
    #Copy the lists to not modify the originals
    pos = positives.copy()
    neg = negatives.copy()
    random.shuffle(pos)
    random.shuffle(neg)
    num_pos = int(np.ceil(batch_size * equil_fact))
    num_negs = batch_size - num_pos
    init_pos = 0 
    end_pos = num_pos
    init_neg = 0 
    end_neg = num_negs
    not_ended_pos = 1
    not_ended_neg = 1
    while not_ended_pos + not_ended_neg:
        if random.random() < 0.15: #Try only a non positive batch to reduce the number of false acceptance
            positive_batch = []
            negative_batch = random.choices(neg, k=num_negs)
        else:    
            batch = []
            if end_pos < len(pos):
                #If we took all indexes we just oversample
                positive_batch = pos[init_pos:end_pos]
                init_pos += num_pos
                end_pos += num_pos
            elif init_pos < len(pos): #the end is more than the number of samples
                #take the rest and take randomly some more 
                positive_batch = pos[init_pos:]
                positive_batch += random.choices(positives, k=num_pos-(len(pos)-init_pos))
                init_pos += num_pos
            else:
                positive_batch = random.choices(pos, k=num_pos)
                not_ended_pos = 0
            
            if end_neg < len(neg):
                #If we took all indexes we just oversample
                negative_batch = neg[init_neg:end_neg]
                init_neg += num_negs
                end_neg += num_negs
            elif init_neg < len(neg): #the end is more than the number of samples
                #take the rest and take randomly some more 
                negative_batch = neg[init_neg:]
                negative_batch += random.choices(negatives, k=num_negs-(len(neg)-init_neg))
                init_neg += num_negs
            else:
                negative_batch = random.choices(neg, k=num_negs)
                not_ended_neg = 0
            
            #negative_batch = random.choices(negatives, k=num_negs)
        batch = positive_batch + negative_batch
        random.shuffle(batch)
        batches.append(batch)
    return batches


class Callback():
    def __init__(self, patience=7, min_delta=0.005, path='checkpoint.pt', verbose=False, shape=[1,16000]):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.Inf
        self.early_stop = False
        self.best_model = None
        self.shape = shape

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = model.state_dict()
            torch.save(self.best_model, self.path)
            #script = torch.jit.script(self.best_model) 
            #torch.jit.save(script, self.path)
            self.early_stop = False
            # if isinstance(self.best_model, dict):
            #     for key, model in self.best_model.items():
            #         # Script the individual model
            #         scripted_model = torch.jit.script(model)  # Ensure each item is a ScriptModule
                    
            #         # Save each scripted model with a unique name
            #         torch.jit.save(scripted_model, f"{self.path}_{key}.pt")
            # else:
            #     # If it's a single model
            #     scripted_model = torch.jit.script(self.best_model)
            #     torch.jit.save(scripted_model, self.path)
            #     torch.jit.save(script, self.path)
            if self.verbose:
                print(f'Validation loss decreased. Saving model to {self.path}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'Validation loss did not decrease. Counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                print('Early stopping triggered.')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a given model')
    parser.add_argument('--model_path', default="None", required=False, help='model path')
    parser.add_argument('--output_dir', default='', required=True, help='output path')
    parser.add_argument('--train_tsv', default='', required=True, help='train tsv file path')
    parser.add_argument('--extended_test_tsv', default='', required=True, help='tsv file containing the extended test')
    parser.add_argument('--dataset_path', default='', required=True, help='dataset path')
    parser.add_argument('--test_tsv', default='', required=True, help = 'test tsv file path')
    parser.add_argument('--extended_dataset_path', default='', required=True, help='dataset for the extended clips')
    parser.add_argument('--validation_tsv', default='', required=True, help = 'dev tsv file path')

    # Training configuration
    #parser.add_argument('--logits', default=False, required=False, help="If you want BCE or BCE_logits for the loss" )
    parser.add_argument('--learning_rate', default=0.001, required=False, help= "Learning rate for the optimizer")
    parser.add_argument('--model_name', required=True, help="Name for saving the model")
    parser.add_argument('--num_epochs', default=10, required=False, help="Number of epochs for the training")
    parser.add_argument('--batch_size', default=16, required=False, help="Batch size for the training")
    parser.add_argument('--device', default="cpu", required=False, help="device in which you want to calc")

    # Audio configuration
    parser.add_argument('--sampling_rate', type=int, default=16000, metavar='SR', help='sampling rate of the audio')
    parser.add_argument('--time_window', type=float, default=1.5, metavar='TW', help='time window covered by every data sample')
    parser.add_argument('--hop', type=float, default=0.256, metavar='H', help='hop between windows')
    parser.add_argument('--nmels', type=int, default=128, metavar='NM', help='number of mels')
    parser.add_argument('--nmfcc', required=False, type=int, default=13, metavar='NMfcc', help='number of MFCC')

    # Detection arguments
    parser.add_argument('--n_positives', type=int, default=1, metavar='NP', help='number of positive windows to decide a positive sample')
    parser.add_argument('--threshold', type=float, default=0.5, metavar='TH', help='detection threshold')

    # SisId
    parser.add_argument('--sysid', required=True, help='system identifier of the model')

    args = parser.parse_args()

    train_model(args)
