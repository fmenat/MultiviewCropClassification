from cropharvest.datasets import CropHarvestLabels, CropHarvest
from cropharvest.utils import NoDataForBoundingBoxError
from cropharvest import countries
from cropharvest.datasets import Task
import numpy as np

from typing import List, Union, Dict

from .views_loader import DataViews_torch

def _to_loader(data: Union[List[np.ndarray],Dict[str,np.ndarray]], batch_size=32, train=True , args_loader={}, **args_structure):
    if type(data) == dict:
        aux_str = DataViews_torch(**data, **args_structure)
    else:
        aux_str = DataViews_torch(data, **args_structure)
    return aux_str.get_torch_dataloader(batch_size = batch_size, train=train, **args_loader)

def get_label_names(DATA_DIR):
    labels_name = CropHarvestLabels(DATA_DIR)._labels["label"].unique()
    print("The number of possible labels are (the library is under this) =",len(labels_name))
    labels_name2 = CropHarvestLabels(DATA_DIR)._labels["classification_label"].unique()
    print("The number of standarized labels are =",len(labels_name2))
    return labels_name,labels_name2


def get_whole_country(DATA_DIR, name: str, label: str ='crop', val: float = 0.0, balance_negative_crops: bool = False,test: bool =False ):
    """
        return the array X and Y for the whole country indicating by name and target label
        label: [] https://github.com/nasaharvest/cropharvest/blob/561c670868e4b66e93938c28ff4023abe114987a/cropharvest/columns.py -- it works over "label" not classification label
        name: https://github.com/nasaharvest/cropharvest/blob/main/datasets.md
        balance_negative_crops: it is more useful when label is not crop
    """
    if (label != 'crop') and (label not in get_country_labels(DATA_DIR, name)):
        print("The country does not have the requested label, run get_country_labels(DATA_DIR, country_name) to check")
        return
    X_country = []
    Y_country= []
    if name.lower() == "global":
        country_bboxes = [None]
    else:
        country_bboxes = countries.get_country_bbox(name)
    for country_bbox in country_bboxes:
        task = Task(country_bbox, label, False, f"{name}_{label}", True) #get all data, afterwards the model will look at what to do
        print("Found bbox: ",task.bounding_box, task.target_label)
        try:
            data_country  = CropHarvest(DATA_DIR, task, val_ratio=val, download= True)
            if balance_negative_crops:
                n_samples = min(map(len, data_country._get_positive_and_negative_indices()))*2+1
            else:
                n_samples = -1 #all
            if test:
                print('Not ready yet')
                #_, test_instances = data_country.test_data(flatten_x=False)
                #return test_instances
                #X_aux, y_aux = test_instances.x, test_instances.y
                return
            else:
                X_aux, y_aux = data_country.as_array(flatten_x=False, num_samples=n_samples)
            X_country.append(X_aux)
            Y_country.append(y_aux)
        except NoDataForBoundingBoxError:
            print("Not data found for the specific country bbox")
        except Exception as e:
            print(e)
    return np.concatenate(X_country),np.concatenate(Y_country)

def get_country_labels(DATA_DIR, name: str):
    C_lab = CropHarvestLabels(DATA_DIR)._labels
    country_bboxes = countries.get_country_bbox(name)
    labels_return = []
    for bbox in country_bboxes:
        labels_return += list(CropHarvestLabels.filter_geojson(C_lab, bbox)['label'])
    labels_return = np.asarray(labels_return)
    labels_return[labels_return==None] = 'single label'
    return np.unique(labels_return)

from cropharvest import countries
def get_countries():
    return countries.get_countries()