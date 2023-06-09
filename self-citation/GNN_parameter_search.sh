# Hyperparameter search for message-passing GNNs

# GCN+virtual node
python -u main.py --conv_name GCNConv --num_layers 2 --lr 0.001 --VN 1
python -u main.py --conv_name GCNConv --num_layers 3 --lr 0.001 --VN 1
python -u main.py --conv_name GCNConv --num_layers 4 --lr 0.001 --VN 1
python -u main.py --conv_name GCNConv --num_layers 5 --lr 0.001 --VN 1

# GIN+virtual node
python -u main.py --conv_name GINConv --num_layers 2 --lr 0.001 --VN 1
python -u main.py --conv_name GINConv --num_layers 3 --lr 0.001 --VN 1
python -u main.py --conv_name GINConv --num_layers 4 --lr 0.001 --VN 1
python -u main.py --conv_name GINConv --num_layers 5 --lr 0.001 --VN 1

# GCN+virtual node
python -u main.py --conv_name GCNConv --num_layers 2 --lr 0.0005 --VN 1
python -u main.py --conv_name GCNConv --num_layers 3 --lr 0.0005 --VN 1
python -u main.py --conv_name GCNConv --num_layers 4 --lr 0.0005 --VN 1
python -u main.py --conv_name GCNConv --num_layers 5 --lr 0.0005 --VN 1

# GIN+virtual node
python -u main.py --conv_name GINConv --num_layers 2 --lr 0.0005 --VN 1
python -u main.py --conv_name GINConv --num_layers 3 --lr 0.0005 --VN 1
python -u main.py --conv_name GINConv --num_layers 4 --lr 0.0005 --VN 1
python -u main.py --conv_name GINConv --num_layers 5 --lr 0.0005 --VN 1

python -u main.py --conv_name GCNConv --num_layers 2 --lr 0.001
python -u main.py --conv_name GCNConv --num_layers 3 --lr 0.001
python -u main.py --conv_name GCNConv --num_layers 4 --lr 0.001
python -u main.py --conv_name GCNConv --num_layers 5 --lr 0.001

python -u main.py --conv_name GINConv --num_layers 2 --lr 0.001
python -u main.py --conv_name GINConv --num_layers 3 --lr 0.001
python -u main.py --conv_name GINConv --num_layers 4 --lr 0.001
python -u main.py --conv_name GINConv --num_layers 5 --lr 0.001

python -u main.py --conv_name GATConv --num_layers 2 --lr 0.001
python -u main.py --conv_name GATConv --num_layers 3 --lr 0.001
python -u main.py --conv_name GATConv --num_layers 4 --lr 0.001
python -u main.py --conv_name GATConv --num_layers 5 --lr 0.001

python -u main.py --conv_name PNAConv --num_layers 2 --lr 0.001
python -u main.py --conv_name PNAConv --num_layers 3 --lr 0.001
python -u main.py --conv_name PNAConv --num_layers 4 --lr 0.001
python -u main.py --conv_name PNAConv --num_layers 5 --lr 0.001

python -u main.py --conv_name DAGNN --num_layers 2 --lr 0.001
python -u main.py --conv_name DAGNN --num_layers 3 --lr 0.001
python -u main.py --conv_name DAGNN --num_layers 4 --lr 0.001
python -u main.py --conv_name DAGNN --num_layers 5 --lr 0.001

