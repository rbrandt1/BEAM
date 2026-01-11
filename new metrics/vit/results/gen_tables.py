import pandas as pd
import numpy as np

showstds = True

folder = "./../"
num_runs = 3


def rename_and_remove(combined_df):

    if 'Time' in combined_df.index:
        combined_df.drop("Time", inplace=True)
        
    combined_df.drop("_soft_PR_abs", inplace=True) # These are the errorous versions of the soft PR, RE and F1 metrics without conversion of GTs to binary. The correct versions of these metrics (without leading _ in their name) are included in the tables.
    combined_df.drop("_soft_RE_abs", inplace=True)
    combined_df.drop("_soft_F1_abs", inplace=True)
    combined_df.drop("_soft_PR_pos", inplace=True)
    combined_df.drop("_soft_RE_pos", inplace=True)
    combined_df.drop("_soft_F1_pos", inplace=True)
    combined_df.drop("_soft_PR_neg", inplace=True)
    combined_df.drop("_soft_RE_neg", inplace=True)
    combined_df.drop("_soft_F1_neg", inplace=True)

    combined_df.drop("Atten_C", inplace=True) # These two metrics are removed because they are identical to EBPG, which is already in the table.
    combined_df.drop("soft_PR_abs", inplace=True)

    rename_map = {
        "Deletion": "Del \cite{petsiuk2018rise}",
        "Insertion": "Ins \cite{petsiuk2018rise}",
        "MuFidelity": "MF \cite{bhatt2020evaluating}",
        "CosSim": "COS \cite{GUIDOTTI2021103428}",
        "_CosSim": "COS* \cite{GUIDOTTI2021103428}",
        "SSIM": "SSIM \cite{hassan2012structural}",
        "Conciseness": "CSN \cite{Amparore_2021}",
        "F1": "F1 \cite{GUIDOTTI2021103428}",
        "IoU": "IoU \cite{GUIDOTTI2021103428}",
        "PR": "PR \cite{GUIDOTTI2021103428}",
        "RE": "RE \cite{GUIDOTTI2021103428}",
        "EBPG": "EBPG \cite{DBLP:journals/corr/abs-1910-01279}",
        "RRA": "RRA \cite{arras2021ground}",
        "soft_PR_abs": "sPR \cite{zhang2023attributionlab}",
        "soft_RE_abs": "sRE \cite{zhang2023attributionlab}",
        "soft_F1_abs": "sF1 \cite{zhang2023attributionlab}",
        "soft_PR_pos": "sPR+ \cite{zhang2023attributionlab}",
        "soft_RE_pos": "sRE+ \cite{zhang2023attributionlab}",
        "soft_F1_pos": "sF1+ \cite{zhang2023attributionlab}",
        "soft_PR_neg": "sPR- \cite{zhang2023attributionlab}",
        "soft_RE_neg": "sRE- \cite{zhang2023attributionlab}",
        "soft_F1_neg": "sF1- \cite{zhang2023attributionlab}",
        "Atten_NC": "A\%NC \cite{zhou2022feature}",
        "cor_nosign": "cor$_{ns}$",
        "cor_sign": "cor$_{s}$"
    }
    combined_df.rename(index=rename_map, inplace=True)

    combined_df.reset_index(inplace=True)

    return combined_df


for name in ['table_results_','table_times_']: 

    print("*******************************************")

    dfs = []

    for i in range(num_runs):
        dfs.append(pd.read_csv(r''+folder+name+'True_'+str(i)+'.csv', index_col=False))
       
    #print(dfs)
    #exit()
       
    df = pd.concat(dfs)
    df.rename(columns={"Unnamed: 0": "Metric"}, inplace=True)

    by_row_index = df.groupby(df.Metric, sort=False)
    
    df_means = by_row_index.mean()
    
    #print(df_means)
    #exit()
    
    if name == 'table_times_':
        df_means = df_means.mean(axis=1)
     
    #print(df_means)
    #exit()
     
    df_std = by_row_index.std()

    #print(df_std)
    #exit()

    if name == 'table_times_':
        df_std = df_std.mean(axis=1)

    #print(df_std)
    #exit()

    if showstds:
        df = df_means.round(2).astype(str) + u"\u00B1" + df_std.round(2).astype(str)
    else:
        df = df_means.round(2).astype(str) 
    
    #print(df)
    
    if name == 'table_times_':
        df = pd.DataFrame(df)
     
        df.insert(1, "Means", df_means, True)
        
        #print(df)
        #exit()
        
        df = df.sort_values(by=['Means'])
        
        #print(df)
        #exit()
        
        df =  df.drop(columns=['Means'])
  
    else:
                          
        df = df.apply(lambda x: x.apply(lambda x: x.lstrip('0')))
        

    df = rename_and_remove(df)

    print(df.to_latex(index=False,escape=False))
