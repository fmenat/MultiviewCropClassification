import pandas as pd
import numpy as np
from IPython.display import display
from pathlib import Path

def save_results(path_data, object_save):
    name_path_, _, file_name_ = path_data.rpartition("/") 
    path_ = Path(name_path_)
    path_.mkdir(parents=True, exist_ok=True)
    path_view = str(path_/file_name_)
    if type(object_save) == type(pd.DataFrame()):
        object_save.to_csv(f"{path_view}.csv", index=True)
    else:
        object_save.savefig(f'{path_view}.pdf')  

def views_metrics_to_matrix(data, view_names):
    n_views = len(view_names)
    matrix = np.zeros((n_views,n_views))
    for i, v1 in enumerate(view_names):
        for j, v2 in enumerate(view_names):
            if j > i:
                matrix[i,j] = np.mean(data[f"{v1}--{v2}"])
    return matrix

def gt_mask(data_views, indexs): #to ipython tools
    return data_views.get_target(indexs, matrix=True)["target"]