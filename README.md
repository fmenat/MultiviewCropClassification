# MultiviewCropClassification
<a href="https://github.com/fmenat/mvlearning">  <img src="https://img.shields.io/badge/Package-mvlearning-blue"/>  </a>
[![paper](https://img.shields.io/badge/arXiv-2308.05407-D12424)](https://www.arxiv.org/abs/2308.05407) 
[![DOI:10.1109/IGARSS52108.2023.10282138](http://img.shields.io/badge/DOI-10.1109/IGARSS52108.2023.10282138-blue.svg)](https://doi.org/10.1109/IGARSS52108.2023.10282138)

> Public repository of our work [*A comparative assessment of multi-view fusion learning for crop classificatio*](https://doi.org/10.1109/IGARSS52108.2023.10282138)
---

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
The data used comes from https://github.com/nasaharvest/cropharvest. However, we also share the structures that we used on Google Drive: https://drive.google.com/drive/folders/1aPlctAL8B5dXSdpM55fr3-RUmAHO3quj



# 🖊️ Citation and more

Mena, Francisco, et al. "*A comparative assessment of multi-view fusion learning for crop classification.*" IGARSS 2023-2023 IEEE International Geoscience and Remote Sensing Symposium. IEEE, 2023.
```bibtex
@inproceedings{mena2023comparativeassessmentmultiview,
  title = {A {{Comparative Assessment}} of {{Multi-view Fusion Learning For Crop Classification}}},
  booktitle = {{{IGARSS}} 2023 - 2023 {{IEEE International Geoscience}} and {{Remote Sensing Symposium}}},
  author = {Mena, Francisco and Arenas, Diego and Nuske, Marlon and Dengel, Andreas},
  date = {2023},
  publisher = {{IEEE}},
  doi={10.1109/IGARSS52108.2023.10282138}
}
```
> [!NOTE]
> * [Presentation](https://github.com/fmenat/fmenat/blob/main/presentations/2023_IGARSS_MVC.pdf)
