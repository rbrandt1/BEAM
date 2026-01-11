import pandas as pd
import numpy as np
import glob
import os
import csv


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


def process_table_results():
 
    run_ids = [0, 1, 2]
    turn_binary_options = [True, False]

    grouped_dfs = {True: [], False: []}
    
    for turn in turn_binary_options:
        for rid in run_ids:
            file_pattern = f"./../table_results_True_{rid}_{turn}.csv"
            if os.path.exists(file_pattern):
                df = pd.read_csv(file_pattern)
                grouped_dfs[turn].append(df)
            else:
                print(f"Warning: {file_pattern} not found.")

    avg_false = pd.concat(grouped_dfs[False])
    avg_false.rename(columns={"Unnamed: 0": "Metric"}, inplace=True)
    by_row_index = avg_false.groupby(avg_false.Metric, sort=False)
    avg_false = by_row_index.mean()

    avg_true = pd.concat(grouped_dfs[True])
    avg_true.rename(columns={"Unnamed: 0": "Metric"}, inplace=True)
    by_row_index = avg_true.groupby(avg_true.Metric, sort=False)
    avg_true = by_row_index.mean()

    final_table = pd.DataFrame(index=avg_false.index)


    # 1st Col: abs(new_1 - new_0) from turn_binary=False
    final_table['CR'] = (avg_false['new_1'] - avg_false['new_0']).abs().round(4)

    # 2nd Col: abs(new_3 - new_2) from turn_binary=False
    final_table['NCR'] = (avg_false['new_3'] - avg_false['new_2']).abs().round(4)

    # 3rd Col: abs(new_4 - new_5) from turn_binary=False
    final_table['Inv'] = (avg_false['new_4'] - avg_false['new_5']).abs().round(4)

    # 4th Col: abs(new_4 from False - new_4 from True)
    final_table['Bin'] = (avg_false['new_4'] - avg_true['new_4']).abs().round(4)

    return final_table

if __name__ == "__main__":
    df = process_table_results()

    df = rename_and_remove(df)

    print(df.to_latex(escape = False,index=False))