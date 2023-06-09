## Train DAG_transformer on NA

On the NA dataset, DAG attention is implemented based on the mask matrix:

```bash
# NA
# Train DAG+transformer
bash ./scripts/na_train_dag_transformer.sh 0 DAGNN
# eval DAG+transformer
bash ./scripts/na_eval_dag_transformer.sh 0 DAGNN 100
# Train DAG+SAT
bash ./scripts/na_train_dag_SAT.sh 0 DAGNN
# eval DAG+SAT
bash ./scripts/na_eval_dag_SAT.sh 0 DAGNN 100
# Train DAG+gps
bash ./scripts/na_train_dag_gps.sh 0 DAGNN
# eval DAG+gps
bash ./scripts/na_eval_dag_gps.sh 0 DAGNN 100
# Train transformer
bash ./scripts/na_train_transformer.sh 0 DAGNN
# eval transformer
bash ./scripts/na_eval_transformer.sh 0 DAGNN 100
# Train graph transformer
bash ./scripts/na_train_graph_transformer.sh 0 DAGNN
# eval graph transformer
bash ./scripts/na_eval_graph_transformer.sh 0 DAGNN 100
# Train SAT
bash ./scripts/na_train_SAT.sh 0 DAGNN
# eval SAT
bash ./scripts/na_eval_SAT.sh 0 DAGNN 100
# Train gps
bash ./scripts/na_train_gps.sh 0 DAGNN
# eval gps
bash ./scripts/na_eval_gps.sh 0 DAGNN 100
```

