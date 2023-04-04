import yaml
import argparse
import os
import sys
import time
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from src.datasets.views_structure import DataViews, load_structure
from src.metrics.metrics import ClassificationMetrics, SoftClassificationMetrics
from src.visualizations.utils import save_results, gt_mask
from src.visualizations.tools import plot_prob_dist_bin, plot_conf_matrix

def classification_metric(
                preds_p_run, 
                indexs_p_run, 
                data_ground_truth, 
                ind_save, 
                show=True, 
                runs_agg = True,
                plot_runs = False, 
                train_data = [], 
                include_metrics = [],
                dir_folder = ""
                ):
    R = len(preds_p_run)

    df_runs = []
    df_runs_diss = []
    time_runs = []
    for r in range(R):        
        y_true, y_pred = gt_mask(data_ground_truth, indexs_p_run[r]), preds_p_run[r]
        y_true = np.squeeze(y_true)
        y_pred_prob = np.squeeze(y_pred)
        y_pred_prob_no_missing = y_pred_prob[y_true != -1]
        y_true_no_missing = y_true[y_true != -1]
        
        y_pred_no_missing = np.argmax(y_pred_prob_no_missing, axis = -1)
        
        d_me = ClassificationMetrics()
        dic_res = d_me(y_pred_no_missing, y_true_no_missing)
        d_me_aux = SoftClassificationMetrics()
        dic_res.update(d_me_aux(y_pred_prob_no_missing, y_true_no_missing))
        
        d_me = ClassificationMetrics(["F1 none", "R none", "P none", "ntrue", 'npred'])
        dic_des = d_me(y_pred_no_missing, y_true_no_missing)
        df_des = pd.DataFrame(dic_des)
        df_des.index = ["label-"+str(i) for i in range(len(dic_des["N TRUE"]))]
        df_runs_diss.append(df_des)

        d_me = ClassificationMetrics(["confusion"])
        cf_matrix = d_me(y_pred_no_missing, y_true_no_missing)["MATRIX"]        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4), squeeze=False)
        plot_conf_matrix(ax[0,0], cf_matrix, "test set")
        save_results(f"{dir_folder}/plots/{ind_save}_preds_r{r:02d}", plt)
        plt.close()
        
        if "f1 bin" in include_metrics:
            dic_res["F1 bin"] = dic_des["F1 NONE"][1]
        if "p bin" in include_metrics:
            dic_res["P bin"] = dic_des["P NONE"][1]
        df_res = pd.DataFrame(dic_res, index=["test"])
        
        if len(d_me.n_samples) == 2 and plot_runs:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,3), squeeze=False)
            plot_prob_dist_bin(ax[0,0], y_pred_prob, y_true_no_missing, f"test run-{r}")
            plt.close()
        
        if len(train_data) != 0:
            y_true, y_pred = gt_mask(train_data[2], train_data[1][r]), train_data[0][r]
            y_true = np.squeeze(y_true)
            y_pred_prob = np.squeeze(y_pred)
            y_pred_prob_no_missing = y_pred_prob[y_true != -1]
            y_true_no_missing = y_true[y_true != -1]

            y_pred_no_missing = np.argmax(y_pred_prob_no_missing, axis = -1)

            d_me = ClassificationMetrics()
            dic_res = d_me(y_pred_no_missing, y_true_no_missing)
            d_me_aux = SoftClassificationMetrics()
            dic_res.update(d_me_aux(y_pred_prob_no_missing, y_true_no_missing))
            df_res.loc["train"] = dic_res
            
            if len(d_me.n_samples) == 2:
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,3), squeeze=False)
                plot_prob_dist_bin(ax[0,0], y_pred_prob, y_true_no_missing, f"train run-{r}")
                plt.close()
                
            d_me = ClassificationMetrics(["F1 none", "R none", "P none", "ntrue", 'npred'])
            dic_des = d_me(y_pred_no_missing, y_true_no_missing)
            df_des = pd.DataFrame(dic_des)
            df_des.index = ["label-"+str(i) for i in range(len(dic_des["N TRUE"]))]
            if plot_runs:
                print("train set", df_des)

            d_me = ClassificationMetrics(["confusion"])
            cf_matrix = d_me(y_pred_no_missing, y_true_no_missing)["MATRIX"]             
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4), squeeze=False)
            plot_conf_matrix(ax[0,0], cf_matrix, "train set")
            plt.close()
        if plot_runs:
            print(f"Run {r} being shown")
            print(df_res.round(4).to_markdown())
        df_runs.append(df_res)
            
    if runs_agg:
        df_concat = pd.concat(df_runs).groupby(level=0)
        df_mean = df_concat.mean()
        df_std = df_concat.std()
        
        save_results(f"{dir_folder}/plots/{ind_save}_preds_mean", df_mean)
        save_results(f"{dir_folder}/plots/{ind_save}_preds_std", df_std)
        if show:
            print(f"################ Showing the {ind_save} ################")
            print(df_mean.round(4).to_markdown()) 
            print(df_std.round(4).to_markdown()) 
            
        df_concat_diss = pd.concat(df_runs_diss).groupby(level=0)
        df_mean_diss = df_concat_diss.mean()
        df_std_diss = df_concat_diss.std()
        
        save_results(f"{dir_folder}/plots/{ind_save}_preds_ind_mean", df_mean_diss)
        save_results(f"{dir_folder}/plots/{ind_save}_preds_ind_std", df_std_diss)
        if show:
            print(df_mean_diss.round(4).to_markdown())
         
    return df_mean,df_std
  
def load_data_sup(data_name, method_name, dir_folder="", **args):
    files_load = [str(v) for v in Path(f"{dir_folder}/pred/{data_name}/{method_name}").glob(f"*.csv")]
    files_load.sort()
    
    preds_p_run = []
    indxs_p_run = []
    for file_n in files_load:
        data_views = pd.read_csv(file_n, index_col=0) #load_structure(file_n)
        preds_p_run.append(data_views.values)
        indxs_p_run.append(list(data_views.index))
    return preds_p_run,indxs_p_run

def calculate_metrics(df_summary, df_std, data_tr,data_te,data_name, method, **args):
    preds_p_run_tr, indexs_p_run_tr = load_data_sup(data_name+"/train", method, **args )
    preds_p_run_te, indexs_p_run_te = load_data_sup(data_name+"/test", method, **args)
    
    df_aux, df_aux2= classification_metric(
                        preds_p_run_te, 
                        indexs_p_run_te, 
                        data_te, 
                        ind_save=f"{data_name}/{method}/", 
                        show=True,
                        train_data = [preds_p_run_tr,indexs_p_run_tr, data_tr], 
                        **args
                        )
    df_summary[method] = df_aux.loc["test"]
    df_std[method] = df_aux2.loc["test"]

def ensemble_avg(method_names, df_summary, df_std, data_tr,data_te,data_name, **args):
    method = "EnsembleAVG"
    preds_p_run_tr, indexs_p_run_tr = [], []
    preds_p_run_te, indexs_p_run_te = [], []
    for method_n in method_names:
        preds_p_run_tr_a, indexs_p_run_tr = load_data_sup(data_name+"/train", method_n, **args )
        preds_p_run_te_a, indexs_p_run_te = load_data_sup(data_name+"/test", method_n, **args )
        preds_p_run_tr.append(preds_p_run_tr_a)
        preds_p_run_te.append(preds_p_run_te_a)
    preds_p_run_tr = np.mean(preds_p_run_tr, axis = 0)
    preds_p_run_te = np.mean(preds_p_run_te, axis = 0)

    df_aux, df_aux2= classification_metric(
                        preds_p_run_te, 
                        indexs_p_run_te, 
                        data_te, 
                        f"{data_name}/{method_n}/", 
                        show=True,
                        train_data = [preds_p_run_tr,indexs_p_run_tr, data_tr], 
                        **args
                        )
    df_summary[method] = df_aux.loc["test"]
    df_std[method] = df_aux2.loc["test"]

def main_evaluation(config_file):
    input_dir_folder = config_file["input_dir_folder"]
    output_dir_folder = config_file["output_dir_folder"]
    data_name = config_file["data_name"]
    include_metrics = ["f1 bin", "p bin"]

    data_tr = load_structure(f"{input_dir_folder}/{data_name}_train.nc")
    data_te = load_structure(f"{input_dir_folder}/{data_name}_test.nc")

    if config_file.get("methods_to_plot"):
        methods_to_plot = config_file["methods_to_plot"]
    else:
        methods_to_plot = sorted(os.listdir(f"{output_dir_folder}/pred/{data_name}/test"))

    df_summary_sup, df_summary_sup_s = pd.DataFrame(), pd.DataFrame()
    pool_names = []
    for method in methods_to_plot:
        print(f"Evaluating method {method}")
        calculate_metrics(df_summary_sup, df_summary_sup_s, 
                        data_tr, data_te,
                        data_name, 
                        method, 
                        include_metrics=include_metrics,
                        plot_runs=config_file.get("plot_runs"),
                        dir_folder=output_dir_folder,
                        )
        if "pool_" in method.lower():
            pool_names.append(method)
    if len(pool_names) != 0:
        ensemble_avg(pool_names, df_summary_sup, df_summary_sup_s, data_tr,data_te,data_name,
                    include_metrics=include_metrics,
                    plot_runs=config_file.get("plot_runs"),
                    dir_folder=output_dir_folder
                    )

    #all figures were saved in output_dir_folder/plots
    print(">>>>>>>>>>>>>>>>> Mean across runs on test set")
    print((df_summary_sup.T*100).round(2).to_markdown())
    print(">>>>>>>>>>>>>>>>> Std across runs on test set")
    print((df_summary_sup_s.T*100).round(2).to_markdown())
    df_summary_sup.T.to_csv(f"{output_dir_folder}/plots/{data_name}/summary_mean.csv")
    df_summary_sup_s.T.to_csv(f"{output_dir_folder}/plots/{data_name}/summary_std.csv")
    

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--settings_file",
        "-s",
        action="store",
        dest="settings_file",
        required=True,
        type=str,
        help="path of the settings file",
    )
    args = arg_parser.parse_args()
    with open(args.settings_file) as fd:
        config_file = yaml.load(fd, Loader=yaml.SafeLoader)
    
    main_evaluation(config_file)