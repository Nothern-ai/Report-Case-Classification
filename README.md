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

## Basic Recipe
Download the model(pth） place it in the /model, then run main.py.
If you provide a dataset, type in "1" and input the path.
If you want to get a classification of a complaint, tap "2," then type in the “Contents of calls” section of the 12345 report.

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
