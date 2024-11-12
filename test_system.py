#This code is to generate an evaluation to a WuW model
#used for the 2024 Albayzin Evaluations for the Wake-up Word Detection challenge
#___________________________________________________
#site: AUDIAS UAM
#Author: Enrique Ernesto de Alvear Doñate
#GitHub: https://github.com/edealvea
#___________________________________________________


import os
import argparse

import pandas as pd
import csv
from tqdm import tqdm
import torch
import torchaudio
import numpy as np
import time 

#import matplotlib.pyplot as plt

class ModelHandler():
    """
    Class to handle the model.
    Change this class to match the model you are using.
    """

    def __init__(self, model_path, threshold):
        self.model = self.load_model(model_path)
        self.threshold = threshold
        self.model.eval()

    def warmup(self, window_size, n_warmup_samples=100):
        """
        Warmup the model with a given audio sample.
        """
        batch_size = 1
        for _ in range(n_warmup_samples):
            x = torch.randn(batch_size, window_size)
            self.process_audio(x)

    def load_model(self, model_path):
        """
        Load a jit model from a given path.
        """
        model = torch.jit.load(model_path, map_location=torch.device('cpu'))
        #print(model.state_dict())
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
        with torch.no_grad():
            output = self.model(audio)
        return output

    def postprocess(self, output):
        """
        Postprocess model output to match the model output.
        """
        wuw_probability = output[0][1].data.item()#If baseline use#output[0][0].item()
        predicted_label = 1 if wuw_probability >= self.threshold else 0
        return wuw_probability, predicted_label
        

    def process_audio(self, audio):
        """
        Process audio from a given path.
        """
        preprocessed = self.preprocess(audio)
        output = self.inference(preprocessed)
        return self.postprocess(output)
    

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
        end_idx = start_idx + n_positives - 1
        prob = sum(probabilities[start_idx:start_idx + n_positives]) / n_positives
        start_time = (start_idx * hop)/sample_rate
        end_time = (end_idx * hop + window_size)/sample_rate
    else:
        prob = sum(probabilities) / len(probabilities)
        start_time = "Unknown"
        end_time = "Unknown"
    return prob, detection, start_time, end_time


def calculate_results(args):

    if os.path.isfile(args.model_path) and os.path.isfile(args.test_tsv) and os.path.isdir(os.path.dirname(args.output_dir)):
        #clips_path = os.path.join(os.path.dirname(args.test_tsv), 'clips')
        clips_path = os.path.join(os.path.join(args.dataset_path), 'clips')
        # Audio configuration
        window_size = int(args.time_window * args.sampling_rate)
        hop = int(args.hop * args.sampling_rate)

        # Model
        model_handler = ModelHandler(args.model_path, args.threshold)
        model_handler.warmup(window_size)

        # Test data
        df = pd.read_csv(args.test_tsv, header=0, sep='\t')
        results_list = []

        for _, row in tqdm(df.iterrows(), total=df.shape[0], ncols=60, desc="Streaming test"):
            filename = row['Filename']
            audio_path = os.path.join(clips_path, filename)
            label = row['Label']
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
                
                wuw_p, wuw_label = model_handler.process_audio(audio)
                predicted_probs.append(wuw_p)
                predicted_labels.append(wuw_label)

            # Calculate results per sample
            prob, label_pred, start, end = calculate_results_outputs(
                predicted_probs,
                predicted_labels,
                window_size,
                hop,
                args.n_positives,
                args.sampling_rate
                )
            results_list.append([filename, prob, label, label_pred, start, end, true_start, true_end])

        columns = ["Filename", "Probability", "Label", "Label_predicted", "Start_Time", "End_Time", "True_Start_Time", "True_End_Time"]
        results_df = pd.DataFrame(results_list, columns=columns)
        return results_df
    else:
        raise Exception("Bad arguments for streaming test. Some of the provided paths are not valid.")


def calculate_results_ext(args):
    if os.path.isfile(args.model_path) and os.path.isfile(args.extended_test_tsv) and os.path.isdir(os.path.dirname(args.output_dir)):
        #clips_path = os.path.join(os.path.dirname(args.test_tsv), 'clips')
        clips_path = os.path.join(os.path.join(args.extended_dataset_path), 'clips')
        # Audio configuration
        window_size = int(args.time_window * args.sampling_rate)
        hop = int(args.hop * args.sampling_rate)

        # Model
        model_handler = ModelHandler(args.model_path, args.threshold)
        model_handler.warmup(window_size)

        # Test data
        df = pd.read_csv(args.extended_test_tsv, header=0, sep='\t')
        results_list = []

        for _, row in tqdm(df.iterrows(), total=df.shape[0], ncols=60, desc="Streaming extended test"):
            filename = row['Sample_ID']
            audio_path = os.path.join(clips_path, "extended_test_"+filename+".wav")
            label = row['Label']
            
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
                
                wuw_p, wuw_label = model_handler.process_audio(audio)
                predicted_probs.append(wuw_p)
                predicted_labels.append(wuw_label)

            # Calculate results per sample
            prob, label_pred, start, end = calculate_results_outputs(
                predicted_probs,
                predicted_labels,
                window_size,
                hop,
                args.n_positives,
                args.sampling_rate
                )
            results_list.append([filename, prob, label, label_pred, start, end])

        columns = ["Filename", "Probability", "Label", "Label_predicted", "Start_Time", "End_Time"]
        results_df = pd.DataFrame(results_list, columns=columns)
        return results_df
    else:
        raise Exception("Bad arguments for streaming test. Some of the provided paths are not valid.")

def look_detection_pattern(labels, n_positives):
    """
    Look for a detection pattern in the labels.
    """
    sub_lst = [1]*n_positives
    len_sub_lst = len(sub_lst)
    for i in range(len(labels)):
        if sub_lst == labels[i:i+len_sub_lst]:
            return 1, i
    return 0, -1

def metric_DCF(result_df,  Pwuw = 0.5, Cmiss = 1, Cfa = 1.5):

    #N_miss is number of WuW that the model has predicted as 0
    N_miss = result_df[(result_df["Label"] == "WuW") & (result_df["Label_predicted"] == 0) ].shape[0]

    #N_fa is number of non_WuW that the model has predicted as 1
    N_fa = result_df[(result_df["Label"] == "NonWuW") & (result_df["Label_predicted"] == 1) ].shape[0]

    #Calculate both probabilities
    Pmiss = N_miss / result_df[result_df["Label"] == "WuW"].shape[0]
    Pfa = N_fa / result_df[result_df["Label"] == "NonWuW"].shape[0]

    #Calculate the TEM
    res_copy = result_df[(result_df["Label"] == "WuW") & (result_df["Label_predicted"] == 1)]
    TEs = np.abs( np.array(res_copy["Start_Time"]) - np.array(res_copy["True_Start_Time"]))
    TEe = np.abs( np.array(res_copy["End_Time"]) - np.array(res_copy["True_End_Time"]))
    tem = np.median(TEs + TEe)
    return Pwuw * Cmiss * Pmiss + (1- Pwuw) * Cfa * Pfa, Pmiss, Pfa, tem

def metric_DCF_ext(result_df,  Pwuw = 0.5, Cmiss = 1, Cfa = 1.5):

    #N_miss is number of WuW that the model has predicted as 0
    N_miss = result_df[(result_df["Label"] == "WuW") & (result_df["Label_predicted"] == 0) ].shape[0]

    #N_fa is number of non_WuW that the model has predicted as 1
    N_fa = result_df[(result_df["Label"] == "unknown") & (result_df["Label_predicted"] == 1) ].shape[0]

    #Calculate both probabilities
    Pmiss = N_miss / result_df[result_df["Label"] == "WuW"].shape[0]
    Pfa = N_fa / result_df[result_df["Label"] == "unknown"].shape[0]

    #return the DFC
    return Pwuw * Cmiss * Pmiss + (1- Pwuw) * Cfa * Pfa , Pmiss, Pfa


def main(args):
        start_time = time.time()
        results_df = calculate_results(args)
        
        dcf, pmiss, p_fa, tem = metric_DCF(results_df)
        end_time = time.time()
        time_test = end_time - start_time

        
        start_time = time.time()
        results_df_ext = calculate_results_ext(args)

        dcf_ext, pmiss_ext, p_fa_ext = metric_DCF_ext(results_df_ext)
        end_time = time.time()
        time_extended = end_time - start_time
        
        row = [args.sysid, args.hop, args.time_window, args.threshold, args.n_positives , dcf, dcf_ext, pmiss,  pmiss_ext, p_fa, p_fa_ext, tem, time_test, time_extended]
        with open("Results.tsv", "a", newline="") as tsv:
            writer = csv.writer(tsv, delimiter="\t")
            writer.writerow(row)
        results_df.to_csv(os.path.join(args.output_dir, args.sysid  + "_eval.tsv"), index=None, sep="\t")
        results_df_ext.to_csv(os.path.join(args.output_dir, args.sysid  + "_extended_eval.tsv"), index=None, sep="\t")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate evaluation results for a given model')
    parser.add_argument('--model_path', default='', required=True, help='model path')
    parser.add_argument('--output_dir', default='', required=True, help='output path')
    parser.add_argument('--test_tsv', default='', required=True, help='tsv file containing the extended test')
    parser.add_argument('--dataset_path', default='', required=True, help='dataset path')
    parser.add_argument('--extended_dataset_path', default='', required=True, help='dataset for the extended clips')
    parser.add_argument('--extended_test_tsv', default='', required=True, help='tsv file containing the extended test')
    parser.add_argument('--device', default="cpu", required=False, help="device in which you want to calc")

    # Audio configuration
    parser.add_argument('--sampling_rate', type=int, default=16000, metavar='SR', help='sampling rate of the audio')
    parser.add_argument('--time_window', type=float, default=1.5, metavar='TW', help='time window covered by every data sample')
    parser.add_argument('--hop', type=float, default=0.256, metavar='H', help='hop between windows')
    

    # Detection arguments
    parser.add_argument('--n_positives', type=int, default=2, metavar='NP', help='number of positive windows to decide a positive sample')
    parser.add_argument('--threshold', type=float, default=0.5, metavar='TH', help='detection threshold')

    # SisId
    parser.add_argument('--sysid', required=True, help='system identifier of the model')

    args = parser.parse_args()

    main(args)
    

