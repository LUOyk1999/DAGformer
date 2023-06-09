### Python environment setup with Conda

```bash
conda create -n graphgps python=3.9
conda activate graphgps

conda install pytorch=1.10 torchvision torchaudio -c pytorch -c nvidia
conda install pyg=2.0.4 -c pyg -c conda-forge

# RDKit is required for OGB-LSC PCQM4Mv2 and datasets derived from it.  
conda install openbabel fsspec rdkit -c conda-forge

pip install torchmetrics
pip install performer-pytorch
pip install ogb
pip install tensorboardX
pip install wandb

conda clean --all
```

### Running DAG+GraphGPS
```bash
conda activate graphgps

# Running DAG+GPS
python main.py --cfg configs/GPS/ogbg-code2-GPS.yaml  wandb.use False
```