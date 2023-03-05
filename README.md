# COPNER

Source code and relevant scripts for our COLING 2022 paper: "[COPNER: Contrastive Learning with Prompt Guiding for Few-shot Named Entity Recognition](https://aclanthology.org/2022.coling-1.222/)".

## Requirements

- Python 3.7.1
- PyTorch (tested version 1.8.1)
- Transformers (tested version 4.8.0)
- seqeval

You can install all required Python packages with `pip install -r requirements.txt`

## Running COPNER

Run `train_demo.py`. The main arguments are presented below.

```shell
-- mode                 training mode, must be in [inter, intra, supervised, i2b2, conll, wnut, mit-movie]
-- task                 training mode, must be in [cross-label-space, domain-transfer, in-label-space]
-- trainN               N in train
-- N                    N in val and test
-- K                    K shot
-- Q                    Num of query per class
-- support_num          the id number of support set of domain-transfer and in-label-space
-- batch_size           batch size
-- train_iter           num of iters in training
-- val_iter             num of iters in validation
-- test_iter            num of iters in testing
-- val_step             val after training how many iters
-- zero_shot            whether to evaluate zero-shot task or not
-- prompt               the prompt id
-- max_length           max length of tokenized sentence
-- lr                   learning rate
-- weight_decay         weight decay
-- grad_iter            accumulate gradient every x iterations
-- load_ckpt            path to load model
-- save_ckpt            path to save model
-- adapt_step           the adaption steps for target label space
-- only_test            no training process, only test
-- ckpt_name            checkpoint name
-- seed                 random seed
-- pretrain_ckpt        bert pre-trained checkpoint
-- dot                  use dot instead of L2 distance in distance calculation
-- tau                  the temperature rate for contrastive learning
# only for structshot
-- struct                     Whether to using viterbi decode or not
-- stuct_tau                  StructShot parameter to re-normalizes the transition probabilities
```

## Citation

If you use our work, please cite:

```bibtex
@inproceedings{huang-etal-2022-copner,
    title = "{COPNER}: Contrastive Learning with Prompt Guiding for Few-shot Named Entity Recognition",
    author = "Huang, Yucheng  and  He, Kai  and  Wang, Yige  and  Zhang, Xianli  and  Gong, Tieliang  and  Mao, Rui  and  Li, Chen",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.222",
    pages = "2515--2527"
}
```
