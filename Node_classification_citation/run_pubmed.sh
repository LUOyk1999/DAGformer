# nodeformer
python main.py --dataset pubmed --rand_split --metric acc --method nodeformer --lr 0.001 \
--weight_decay 5e-2 --num_layers 2  --hidden_channels 16 --num_heads 2 --rb_order 2 --rb_trans sigmoid --lamda 1.0 \
--M 30 --K 10 --use_bn --use_residual --use_gumbel --runs 10 --epochs 1000 --device 1 --seed 42

# dag+nodeformer
python -u main.py --dataset pubmed --rand_split --metric acc --method dag_nodeformer --lr 0.001 \
--weight_decay 5e-3 --num_layers 2 --hidden_channels 16 --num_heads 2 \
--rb_order 2 --use_bn --use_residual --runs 10 --epochs 1000 --device 0 --lamda 1.0 --seed 42

# dag+transformer
python -u main.py --dataset pubmed --rand_split --metric acc --method dag_nodeformer --lr 0.01 \
--weight_decay 5e-3 --num_layers 3 --hidden_channels 16 --num_heads 2 \
--rb_order 2 --use_bn --use_residual --runs 10 --epochs 1000 --device 0 --lamda 0.0 --seed 42

# dag+sat
python -u main.py --dataset pubmed --rand_split --metric acc --method dag_nodeformer --lr 0.01 \
--weight_decay 5e-3 --num_layers 2 --hidden_channels 16 --num_heads 2 \
--rb_order 2 --use_bn --use_residual --runs 10 --epochs 1000 --device 0 --lamda 0.0 --seed 42 --sat 1

# for model in gcn gat mixhop
# for model in dag_nodeformer
# do
# for lr in 0.01 0.001
# do
# for hidden_channels in 16 32
# do
# for lamda in 1.0
# do
# for num_layers in 2 3 4
# do
# for num_heads in 2
# do
# python -u main.py --dataset pubmed --rand_split --metric acc --method $model --lr $lr \
# --weight_decay 5e-3 --num_layers $num_layers --hidden_channels $hidden_channels --num_heads $num_heads \
# --rb_order 2 --use_bn --use_residual --runs 3 --epochs 1000 --device 0 --lamda $lamda --seed 42
# done
# done
# done
# done
# done
# done