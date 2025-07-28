import pandas as pd
import re

def method_1(RT):
    dead_volume = 3.25666666666666
    time = [0,10,40,55,60]
    gradient = [6,11,21,31,61]
    corrected_rt = RT/60 - dead_volume
    for start_index in range(len(time)-1):
        end_index = start_index +1
        start_time = time[start_index]
        end_time = time[start_index+1]
        start_gradient = gradient[start_index]
        end_gradient = gradient[end_index]
        slope = (end_gradient-start_gradient)/(end_time-start_time)
        if start_time < corrected_rt and corrected_rt < end_time:
            b_percentage = start_gradient + slope * (corrected_rt-start_time)
            return b_percentage


def method_2(RT):
    dead_volume = 3.25666666666666
    time = [0,10,40,55,70]
    gradient = [6,11,21,31,91]
    corrected_rt = RT/60 - dead_volume
    for start_index in range(len(time)-1):
        end_index = start_index +1
        start_time = time[start_index]
        end_time = time[start_index+1]
        start_gradient = gradient[start_index]
        end_gradient = gradient[end_index]
        slope = (end_gradient-start_gradient)/(end_time-start_time)
        if start_time < corrected_rt and corrected_rt < end_time:
            b_percentage = start_gradient + slope * (corrected_rt-start_time)
            return b_percentage
        

def method_3(RT):
    dead_volume = 3.25666666666666
    time = [0,10,40,55,70,80]
    gradient = [6,11,21,31,91,91]
    corrected_rt = RT/60 - dead_volume
    for start_index in range(len(time)-1):
        end_index = start_index +1
        start_time = time[start_index]
        end_time = time[start_index+1]
        start_gradient = gradient[start_index]
        end_gradient = gradient[end_index]
        slope = (end_gradient-start_gradient)/(end_time-start_time)
        if start_time < corrected_rt and corrected_rt < end_time:
            b_percentage = start_gradient + slope * (corrected_rt-start_time)
            return b_percentage



def clean_psm(input_path, output_path, method=None):
    if input_path[-4:]==".tsv":
        df = pd.read_csv(input_path, sep="\t")
    elif input_path[-4:]==".csv":
        df = pd.read_csv(input_path, sep=",")
    df = df[["Protein","Peptide","Retention","Intensity","Observed Mass","Delta Mass","Charge"]]
    df = df[df['Intensity'] > 0]
    df = df[abs(df['Delta Mass']) < 0.1]

    start = None
    end = None
    if ("Phe" in input_path) or ("2plex" in input_path):
        start = "F"
        if "K-term" in input_path:
            end = "KF"
        elif "R-term" in input_path:
            end = "R"
    elif "LF" in input_path:
        if "K-term" in input_path:
            end = "K"
        elif "R-term" in input_path:
            end = "R"
    
    if start:
        df['start'] = df["Peptide"].apply(lambda x:x[0])
        df = df[df['start']==start]
        df.drop('start', axis=1, inplace=True)

    if end:
        if len(end)==1:
            df['end'] = df["Peptide"].apply(lambda x:x[-1])
            df = df[df['end']==end]
            df.drop('end', axis=1, inplace=True)
        elif len(end)==2:
            df['end'] = df["Peptide"].apply(lambda x:x[-2:])
            df = df[df['end']==end]
            df.drop('end', axis=1, inplace=True)

    if ("D-Phe" in input_path) and ("K-term" in input_path):
        df['Peptide'] = df['Peptide'].apply(lambda x:'f'+x[1:-1]+'f')
    if ("D-Phe" in input_path) and ("R-term" in input_path):
        df['Peptide'] = df['Peptide'].apply(lambda x:'f'+x[1:])
    if method=="method_1":
        df['B'] = df["Retention"].apply(method_1)
    elif method=="method_2":
        df['B'] = df["Retention"].apply(method_2)
    else:
        df['B'] = df["Retention"].apply(method_3)

    df = df[["Protein","Peptide","Retention","B","Intensity","Observed Mass","Charge"]]
    df = df.loc[df.groupby("Protein")["Intensity"].idxmax()]
    df.sort_values(by="Protein", inplace=True)
    df = df[["Peptide","Retention","B","Observed Mass","Charge"]]
    df = df.rename(columns={"Retention":"RT","Observed Mass":"M","Charge":"Z"})
    df.to_csv(output_path, index=False)
