# Transformer over Directed Acyclic Graph 

Once you have activated the environment and installed all dependencies, run:

```bash
source s
```

Datasets will be downloaded via OGB package.

## Train DAG_transformer (DAG+models) on ogbg-code2

After having run `source s`, run `cd experiments`. 

```bash
# Train DAG+Transformer based on MPNN (save time)
python train_code2.py --use-edge-attr --use_mpnn true
# Train DAG+Transformer based on MPNN with Receptive Field N_k (k=6)
python train_code2.py --use-edge-attr --use_mpnn true --k 6
# Train DAG+Transformer based on mask 
python train_code2.py --use-edge-attr
# Train DAG+SAT based on MPNN (save time)
python train_code2.py --gnn-type gcn --use-edge-attr --use_mpnn true --SAT true
# Train DAG+SAT based on MPNN with Receptive Field N_k (k=6)
python train_code2.py --gnn-type gcn --use-edge-attr --use_mpnn true --k 6 --SAT true
# Train DAG+SAT based on mask 
python train_code2.py --gnn-type gcn --use-edge-attr --SAT true
```

Train DAG+GPS:

```bash
cd GraphGPS
```