# Hyperparameter search for transformer-based models

# DAG+Nodeformer
python -u main_node.py --conv_name NodeFormer --nhid 32 --num_heads 4 --num_layers 2 --lr 0.005

# Nodeformer
python -u main_node_nodeformer.py --conv_name NodeFormer --nhid 64 --num_heads 4 --num_layers 2 --lr 0.001

# Transformer
python -u main.py --nhid 128 --num_heads 4 --num_layers 2 --lr 0.0005 --DAG_attention 0 --pe none
python -u main.py --nhid 128 --num_heads 4 --num_layers 3 --lr 0.0005 --DAG_attention 0 --pe none
python -u main.py --nhid 128 --num_heads 4 --num_layers 4 --lr 0.0005 --DAG_attention 0 --pe none
# Graph Transformer
python -u main.py --nhid 128 --num_heads 4 --num_layers 3 --lr 0.001 --DAG_attention 0 --pe Eigvecs
python -u main.py --nhid 128 --num_heads 4 --num_layers 3 --lr 0.0005 --DAG_attention 0 --pe Eigvecs

# SAT
python -u main.py --nhid 128 --num_heads 4 --num_layers 2 --lr 0.0005 --SAT 1  --pe Eigvecs  --DAG_attention 0
python -u main.py --nhid 128 --num_heads 4 --num_layers 2 --lr 0.0005 --SAT 1  --pe RWPE  --DAG_attention 0
python -u main.py --nhid 128 --num_heads 4 --num_layers 2 --lr 0.0005 --SAT 1  --pe none  --DAG_attention 0
python -u main.py --nhid 128 --num_heads 4 --num_layers 2 --lr 0.001 --SAT 1  --pe Eigvecs  --DAG_attention 0
python -u main.py --nhid 128 --num_heads 4 --num_layers 2 --lr 0.001 --SAT 1  --pe RWPE  --DAG_attention 0
python -u main.py --nhid 128 --num_heads 4 --num_layers 2 --lr 0.001 --SAT 1  --pe none  --DAG_attention 0

# DAG+transformer
python -u main.py --nhid 128 --num_heads 4 --num_layers 2 --lr 0.0005

# Ablation Study
# Transformer+RWPE
python -u main.py --nhid 128 --num_heads 4 --num_layers 2 --lr 0.0005 --DAG_attention 1 --pe RWPE
# Transformer+LapPE
python -u main.py --nhid 128 --num_heads 4 --num_layers 2 --lr 0.0005 --DAG_attention 1 --pe Eigvecs
# Transformer+DAGPE
python -u main.py --nhid 128 --num_heads 4 --num_layers 2 --lr 0.0005 --DAG_attention 0
# Transformer+DAG_attention
python -u main.py --nhid 128 --num_heads 4 --num_layers 2 --lr 0.0005 --DAG_attention 1 --pe none

# SAT+DAG
python -u main.py --nhid 128 --num_heads 4 --num_layers 2 --lr 0.001 --SAT 1

# Ablation Study
# SAT+DAGPE
python -u main.py --nhid 128 --num_heads 4 --num_layers 2 --lr 0.001 --SAT 1 --DAG_attention 0
# SAT+DAG_attention
python -u main.py --nhid 128 --num_heads 4 --num_layers 2 --lr 0.001 --SAT 1 --pe none

# DAG+GraphGPS
python -u main.py --nhid 128 --num_heads 4 --num_layers 2 --lr 0.0005 --gps 2

# GraphGPS
python -u main.py --nhid 128 --num_heads 4 --num_layers 2 --lr 0.0005 --gps 1 --pe none
