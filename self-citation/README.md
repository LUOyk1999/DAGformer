## Train DAG_transformer on self-citation

Dataset: `./data/data_origin`
In a self-citation networks, each node is a paper of this scholar and each directed edge indicates that one paper is cited by another one. 
For a scholar, the edges are shown in:
[influence_arc_*.csv] :

| Field         | Type         | Null | Key  | Default | Extra |
| ------------- | ------------ | ---- | ---- | ------- | ----- |
| citingpaperID | varchar(100) | NO   | MUL  | NULL    |       |
| citedpaperID  | varchar(100) | NO   | MUL  | NULL    |       |
| oldProb       | float        | YES  |      | NULL    |       |

The nodes are shown in:
[papers_arc_*.csv] :
| Field         | Type         | Null | Key  | Default                | Extra             |
| ------------- | ------------ | ---- | ---- | ---------------------- | ----------------- |
| paperID       | varchar(100) | NO   | MUL  | NULL                   |                   |
| year          | int          | NO   |      | NULL                   |                   |
| citationCount | int          | NO   |      | -1 (means missing)     | -2(means seleted) |
| label         | Int          | NO   |      | -2 (means not seleted) |                   |



Train:

On the self-citation dataset, DAG attention is implemented based on MPNN:

```bash
# Hyperparameter for GNN models
sh GNN_parameter_seach.sh
# Hyperparameter for transformer-based models
sh parameter_search.sh
```