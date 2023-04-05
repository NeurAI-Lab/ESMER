# ESMER
Official Repository for ICLR'23 Paper "[Error Sensitivity Modulation based Experience Replay: Mitigating Abrupt Representation Drift in Continual Learning](https://openreview.net/pdf?id=zlbci7019Z3)"

We extended the [CLS-ER](https://github.com/NeurAI-Lab/CLS-ER) repo with our method

## Setup

+ Use `python main.py` to run experiments.
+ Use argument `--load_best_args` to use the best hyperparameters for each of the evaluation setting from the paper.
+ To reproduce the results in the paper run the following  

    `python main.py --dataset <dataset> --model <model> --buffer_size <buffer_size> --load_best_args`

  ## Examples:

    ```
    python main.py --dataset seq-cifar10 --model esmer --buffer_size 200 --load_best_args
    
    python main.py --dataset seq-cifar100 --model esmer --buffer_size 200 --load_best_args
    ```

  ## For GCIL-CIFAR-100 Experiments

    ```
    python main.py --dataset gcil-cifar100 --weight_dist unif --model esmer --buffer_size 200 --load_best_args
    
    python main.py --dataset gcil-cifar100 --weight_dist longtail --model esmer --buffer_size 200 --load_best_args
    ```

## Requirements

- torch==1.7.0
- torchvision==0.9.0 
