Here is the source code  for the submission paper "Retrieving GNN Architecture  for Collaborative Filtering ".  Thank you for your interest in our work!

First we will construct meta-train database for both tasks.

```shell
bash space_for_cf/run_batch.sh
```

Then we will do the meta train and searching.

```shell
bash run.sh
```

The generate yaml file move to configs in  space_for cf and rerun the configs based on space_for cf.

the meta feature extractor in space_for_cf/datasets/dataset_to_meta.py