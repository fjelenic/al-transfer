# On Dataset Transferability in Active Learning for Transformers

Code for the paper [On Dataset Transferability in Active Learning for Transformers](https://aclanthology.org/2023.findings-acl.144/), published in the Findings of ACL 2023.

## Abstract

Active learning (AL) aims to reduce labeling costs by querying the examples most beneficial for model learning. While the effectiveness of AL for fine-tuning transformer-based pre-trained language models (PLMs) has been demonstrated, it is less clear to what extent the AL gains obtained with one model transfer to others. We consider the problem of transferability of actively acquired datasets in text classification and investigate whether AL gains persist when a dataset built using AL coupled with a specific PLM is used to train a different PLM. We link the AL dataset transferability to the similarity of instances queried by the different PLMs and show that AL methods with similar acquisition sequences produce highly transferable datasets regardless of the models used. Additionally, we show that the similarity of acquisition sequences is influenced more by the choice of the AL method than the choice of the model.

## Run the Experiments

### Get the Experimental Results

- <code>run.py</code> runs the experiments and produces <code>results.pkl</code> pickle file that stores the raw results of the experiments.
- <code>create-dataset.py</code> formats the data in the format that is easier to analyze and dumps it into <code>data.csv</code> CSV file.

### Analyzing results

- <code>experiments.ipynb</code> is used to compute all the results from the original paper and create all the figures.
- <code>anova-model-vs-method.ipynb</code> is used to run two Kruskal-Wallis H-tests to demonstrate that the distributions of ASM values are more alike when grouped by the AL methods than when grouped by the model pairings.
