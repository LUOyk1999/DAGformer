# nodeformer
python main.py --dataset citeseer --rand_split --metric acc --method nodeformer --lr 0.001 \
--weight_decay 5e-3 --num_layers 2 --hidden_channels 32 --num_heads 2 --rb_order 2 --rb_trans sigmoid \
--lamda 1.0 --M 30 --K 10 --use_bn --use_residual --use_gumbel --runs 5 --epochs 1000 --device 0

# dag+nodeformer
python -u main.py --dataset citeseer --rand_split --metric acc --method dag_nodeformer --lr 0.01 \
--weight_decay 5e-3 --num_layers 4 --hidden_channels 16 --num_heads 4 \
--rb_order 2 --use_bn --use_residual --runs 10 --epochs 1000 --device 0

# dag+transformer
python -u main.py --dataset citeseer --rand_split --metric acc --method dag_nodeformer --lr 0.001 \
--weight_decay 5e-3 --num_layers 5 --hidden_channels 64 --num_heads 2 \
--rb_order 2 --use_bn --use_residual --runs 10 --epochs 1000 --device 0 --lamda 0.0

# transformer
python -u main.py --dataset citeseer --rand_split --metric acc --method transformer --lr 0.001 \
--weight_decay 5e-3 --num_layers 5 --hidden_channels 64 --num_heads 2 \
--rb_order 2 --use_bn --use_residual --runs 10 --epochs 1000 --device 0

# dag+sat
python -u main.py --dataset citeseer --rand_split --metric acc --method dag_nodeformer --lr 0.01 \
--weight_decay 5e-3 --num_layers 4 --hidden_channels 32 --num_heads 2 \
--rb_order 2 --use_bn --use_residual --runs 10 --epochs 1000 --device 0 --lamda 0.0 --sat 1

# sat
python -u main.py --dataset citeseer --rand_split --metric acc --method transformer --lr 0.01 \
--weight_decay 5e-3 --num_layers 4 --hidden_channels 32 --num_heads 2 \
--rb_order 2 --use_bn --use_residual --runs 10 --epochs 1000 --device 0 --sat 1


# for lr in 0.001 0.01
# do
# for hidden_channels in 16 32 64
# do
# for lamda in 0.5 1.0
# do
# for num_layers in 2 3 4
# do
# for num_heads in 2 4
# do
# python -u main.py --dataset citeseer --rand_split --metric acc --method dag_nodeformer --lr $lr \
# --weight_decay 5e-3 --num_layers $num_layers --hidden_channels $hidden_channels --num_heads $num_heads \
# --rb_order 2 --use_bn --use_residual --runs 10 --epochs 1000 --device 0 --lamda $lamda
# done
# done
# done
# done
# done
