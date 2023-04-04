# MultiviewCropClassification
Code used for the crop classification (CropHarvest) based on multi-view data fusion


### Training
* To train a single-view learning model (e.g. Input-level fusion):  
```
python train_singleview.py -s config/singleview_ex.yaml
```
* To train all the views individually with single-view learning (e.g. for single-view predictions or Ensemble-based fusion):  
```
python train_singleview_pool.py -s config/singleviewpool_ex.yaml
```
* To train a multi-view learning model (e.g. Feature-level fusion, Decision-level fusion, Gated Fusion, Feature-level fusion with MultiLoss):  
```
python train_multiview.py -s config/multiview_ex.yaml
```

### Evaluation
* To evaluate the model by its predictions (performance):
```
python evaluate_predictions.py -s config/evaluation_ex.yaml
```


## Installation
Please install the required packages with the following command:
```
pip install -r requirements.txt
```

> for torch 
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

## Data
The data used comes from https://github.com/nasaharvest/cropharvest. However we also share the structures that we used on Google Drive: https://drive.google.com/drive/folders/1aPlctAL8B5dXSdpM55fr3-RUmAHO3quj


## Source
Public repository of our IGARSS 2023 submission.

## Citation
Not yet..

## Licence
Copyright (C) 2022  authors of this github.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
