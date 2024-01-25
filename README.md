# Report Case Classification for “12345”

> [Mian Wu*](https://github.com/NothernSJTU)
> <br>SJTU<br>


## Results on 2023 "12345" Dataset

<!-- results with basic recipe (s.d. = stochastic depth)   | -->


results with improved recipe

| model        | Train_Acc | Validation_Acc |
|:------------|:-----:|:------:|
| Bert     |  98.1  | 97.8  | 


## Installation
Please check [INSTALL.md](INSTALL.md) for installation instructions. 

<!-- ## Training

## Basic Recipe
Download the model(pth）then place in the /model, then run main.py.
If you provide a dataset, type in "1" and input the path.
If you want to get a classification of a complaint, tap "2," then type in the “Contents of calls” section of the 12345 report.
### Basic Recipe
We list commands for early dropout, early stochastic depth on `ViT-T` and late stochastic depth on `ViT-B`.
- For training other models, change `--model` accordingly, e.g., to `vit_tiny`, `mixer_s32`, `convnext_femto`, `mixer_b16`, `vit_base`.
- Our results were produced with 4 nodes, each with 8 gpus. Below we give example commands on both multi-node and single-machine setups.

**Early dropout**

multi-node
```
python run_with_submitit.py --nodes 4 --ngpus 8 \
--model vit_tiny --epochs 300 \
--batch_size 128 --lr 4e-3 --update_freq 1 \
--dropout 0.1 --drop_mode early --drop_schedule linear --cutoff_epoch 50 \
--data_path /path/to/data/ \
--output_dir /path/to/results/
```

single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model vit_tiny --epochs 300 \
--batch_size 128 --lr 4e-3 --update_freq 4 \
--dropout 0.1 --drop_mode early --drop_schedule linear --cutoff_epoch 50 \
--data_path /path/to/data/ \
--output_dir /path/to/results/
``` -->




### Improved Recipe
<!-- Our improved recipe extends training epochs from `300` to `600`, and reduces both `mixup` and `cutmix` to `0.3`. -->


## Acknowledgement
This repository is built using the [Lee Meng](https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html#%E7%94%A8-BERT-fine-tune-%E4%B8%8B%E6%B8%B8%E4%BB%BB%E5%8B%99) codebase.

## License
This project is released under the CC-BY-NC 4.0 license. Please see the [LICENSE](LICENSE) file for more information.

## Citation
If you find this repository helpful, please consider citing:
```bibtex
@inproceedings{wu2024rcc,
  title={Report Case Classification},
  author={Mian Wu},
  year={2024},
}
```
