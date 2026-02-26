# Multimodal_Learning

This study project evaluates different fusion approaches in the context of multimodal learning. Runs are tacked using weights&biases and visualizations are obtained by using fiftyone.

Weights & Biases
username:      zeihenri
project links:
- https://wandb.ai/zeihenri-hasso-plattner-institut/fusion_architecture_comparison?nw=nwuserzeihenri
- https://wandb.ai/zeihenri-hasso-plattner-institut/maxpool_vs_strided_ablation?nw=nwuserzeihenri
- https://wandb.ai/zeihenri-hasso-plattner-institut/contrastive%20pre%20training?nw=nwuserzeihenri


## fusion architecture experiment analysis

The experiment was performed with the following constant hyperparameters and settings:
- Batch Size:32
- Epochs:20
- n_train: 15,998
- n_val: 4,000
->  Decided to use the full dataset
- learning rate: 1e-4 (constant)
- Optimizer: torch.optim Adam
![alt text](results/w&b%20screenshots/fusion_comparision.png)
We can see that overall, the intermediate fusion approaches outperformed the late fusion appproaches. The validation losses were lower and f1 scores was the same between late fusion and intermediate fusion with matrix multiplication, while the other intermediate fusion approaches achieved a perfect f1 score. The best performance was achieved using the concatenation intermediate fusion model. Intuitively it makes sense that it outperforms addition and multiplication as for a given concatenation there is exactly one pair of matrices which lead to that result. On the flipside, it does result in the first layer of the shared embedder having to learn twice as many parameters. Although for the experiment to be conclusive more experiments e.g. in regards to parameters would need to be performed, we could show that the intermediate fusion concatenation approach significantly outperformed the other (also well-performing) models, indicating 
1. That learning the correlation between rgb and lidar data improved validation loss
2. That among the Intermediate fusion strategies, concatenation is the most effective

## MaxPool2d to Strided Convolution Ablation Analysis

![alt text](results/w&b%20screenshots/ablation_part1.png)

![alt text](results/w&b%20screenshots/ablation_part2.png)

The experiment was performed with the following constant hyperparameters and settings:
- Batch Size:32
- Epochs:20
- n_train: 15,998
- n_val: 4,000
->  Decided to use the full dataset
- learning rate: 1e-4 (constant)
- Optimizer: torch.optim Adam
The training of the strided models logically took longer, as there were more parameters to learn (by around 1/3). The results in regards to loss and accuracy are mixed. For the intermediate fusion concatenation and addition model max pooling led to significantly better results. Meanwhile, intermediate fusion by multiplication strided convolutions performed slightly better and for late fusion it was mixed between accuracy and loss. Overall the differences for the latter 2 were quite small and likely not significant and robustly reproducible. The differences seen for the concatenation model were largest. That might be due to the concatenation model having more parameters anyway leading to the dataset size becoming insufficient to learn proper downsampling through strided convolutions in paralel.
