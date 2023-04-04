import yaml
import argparse
import os
import sys
import time
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.utils import class_weight

from src.training.utils import preprocess_views
from src.training.learn_pipeline import PoolEnsemble_train
from src.datasets.views_structure import DataViews, load_structure
from src.datasets.utils import _to_loader

OVERWRITE=True
def main_run(config_file):
    start_time = time.time()
    input_dir_folder = config_file["input_dir_folder"]
    output_dir_folder = config_file["output_dir_folder"]
    data_name = config_file["data_name"]
    view_names = config_file["view_names"]
    runs = config_file["experiment"]["runs"]
    preprocess_args = config_file["experiment"]["preprocess"]
    val_size = config_file["experiment"]["val_size"]
    mlflow_runs_exp = config_file["experiment"]["mlflow_runs_exp"]
    method_name = "Pool"
    if config_file["training"].get("early_stop_args"):
        if config_file["training"].get("early_stop_args").get("min_delta"):
            config_file["training"]["early_stop_args"]["min_delta"] *= len(view_names)

    try:
        data_views_tr = load_structure(f"{input_dir_folder}/{data_name}_train")
    except:
        data_views_tr = load_structure(f"{input_dir_folder}/{data_name}")
    raw_dims = {}
    for view_name in data_views_tr.get_view_names():
        aux_data = data_views_tr.get_view_data(view_name)["views"]
        raw_dims[view_name] = {"raw": list(aux_data.shape[1:]), "flatten": int(np.prod(aux_data.shape[1:]))}
    funcs = preprocess_views(data_views_tr, **preprocess_args)
    views_tr = data_views_tr.generate_full_view_data(views_first=True, view_names=view_names)
    
    data_views_te = load_structure(f"{input_dir_folder}/{data_name}_test")
    preprocess_views(data_views_te, train=False, funcs=funcs, **preprocess_args)
    views_te = data_views_te.generate_full_view_data(views_first=True, view_names=view_names)
        
    if "loss_args" not in config_file["training"]: 
        config_file["training"]["loss_args"] = {}
    config_file["training"]["loss_args"]["name"] = "ce" if "name" not in config_file["training"]["loss_args"] else config_file["training"]["loss_args"]["name"]
    train_data_target = views_tr["target"].astype(int).flatten()
    config_file["training"]["loss_args"]["weight"]=class_weight.compute_class_weight(class_weight='balanced',classes= np.unique(train_data_target), y=train_data_target)
    
    run_id_mlflow = None 
    metadata_r = {"epoch_runs":[], "prediction_time":[], "training_time":[], "best_score":[]}
    for r in range(runs):
        if os.path.isfile(f"{output_dir_folder}/pred/{data_name}/test/{method_name}/out_run-{r:02d}.csv") and not OVERWRITE:
            print(f"run {r} already created.. so skipping")
            continue
        if mlflow_runs_exp:
            run_id_mlflow = "ind"
        print(f"Executing model on run {r}")
        
        if val_size!= 0:
            mask_train = np.random.rand(len(data_views_tr.get_all_identifiers())) <= (1-val_size)
            indx_train = np.arange(len(mask_train))[~mask_train]
            data_views_tr.set_test_mask(indx_train, reset=True)

            train_data = data_views_tr.generate_full_view_data(train = True, views_first=True, view_names=view_names)
            val_data = data_views_tr.generate_full_view_data(train = False, views_first=True, view_names=view_names)
        else:
            train_data = views_tr 
            val_data = None
        
        start_aux = time.time()
        method, trainer = PoolEnsemble_train(train_data, val_data=val_data,run_id=r,
                                                      method_name=method_name, run_id_mlflow = run_id_mlflow, **config_file)
        mlf_logger, run_id_mlflow = trainer.loggers[0], trainer.loggers[0].run_id 
        mlf_logger.experiment.log_dict(run_id_mlflow, raw_dims, "original_data_dim.yaml")
        mlf_logger.experiment.log_dict(run_id_mlflow, config_file, "config_file.yaml")
        metadata_r["training_time"].append(time.time()-start_aux)
        metadata_r["epoch_runs"].append(trainer.callbacks[0].stopped_epoch)
        metadata_r["best_score"].append(trainer.callbacks[0].best_score.cpu())
        print("Training done")
        
        pred_time_Start = time.time()
        BS = config_file["training"]["batch_size"]
        outputs_tr = method.transform(_to_loader(views_tr, batch_size=BS, train=False), output=True, out_norm=True)    
        outputs_te = method.transform(_to_loader(views_te, batch_size=BS, train=False), output=True, out_norm=True)
        metadata_r["prediction_time"].append(time.time()-pred_time_Start)
        
        for view_n, values in outputs_tr["views:prediction"].items():
            data_save_tr = DataViews([values], identifiers=views_tr["identifiers"], view_names=[f"out_run-{r:02d}"])
            data_save_tr.save(f"{output_dir_folder}/pred/{data_name}/train/{method_name}_{view_n}", ind_views=True, xarray=False)
            mlf_logger.experiment.log_artifact(run_id_mlflow, f"{output_dir_folder}/pred/{data_name}/train/{method_name}_{view_n}/out_run-{r:02d}.csv",
                                              artifact_path=f"preds/train/{method_name}_{view_n}")
        
        for view_n, values in outputs_te["views:prediction"].items():
            data_save_te = DataViews([values], identifiers=views_te["identifiers"], view_names=[f"out_run-{r:02d}"])
            data_save_te.save(f"{output_dir_folder}/pred/{data_name}/test/{method_name}_{view_n}", ind_views=True, xarray=False)
            mlf_logger.experiment.log_artifact(run_id_mlflow, f"{output_dir_folder}/pred/{data_name}/test/{method_name}_{view_n}/out_run-{r:02d}.csv",
                                              artifact_path=f"preds/test/{method_name}_{view_n}")
        print(f"Run {r:02d} of {method_name} finished...")
        
    if type(run_id_mlflow) != type(None):
        mlf_logger.experiment.log_metric(run_id_mlflow, "mean_tr_time", np.mean(metadata_r["training_time"]))
        mlf_logger.experiment.log_metric(run_id_mlflow, "mean_pred_time", np.mean(metadata_r["prediction_time"]))
        mlf_logger.experiment.log_metric(run_id_mlflow, "mean_epoch_runs", np.mean(metadata_r["epoch_runs"]))
        mlf_logger.experiment.log_metric(run_id_mlflow, "mean_best_score", np.mean(metadata_r["best_score"]))
        pd.DataFrame(metadata_r).to_csv(f"{output_dir_folder}/metadata_runs.csv")
        mlf_logger.experiment.log_artifact(run_id_mlflow, f"{output_dir_folder}/metadata_runs.csv",)
        os.remove(f"{output_dir_folder}/metadata_runs.csv")
    print("Epochs for %s runs on average for %.2f epochs +- %.3f"%(method_name,np.mean(metadata_r["epoch_runs"]),np.std(metadata_r["epoch_runs"])))
    print(f"Finished whole execution of {runs} runs in {time.time()-start_time:.2f} secs")    

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
    
    main_run(config_file)
