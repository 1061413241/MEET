# MEET

This is an official pytorch implementation of our paper "MEET: Multi-band EEG Transformer". In this repository, we provide PyTorch code for training and testing our proposed MEET model. MEET provides an efficient EEG classification framework that achieves state-of-the-art results on several EEG benchmarks such as SEED.

If you find MEET useful in your research, please use the following BibTeX entry for citation.



# Model Variants

We provide the following three variants of MEET.

| Model      | Depth | Heads | Hidden size | MLP size | Params |
| ---------- | ----- | ----- | ----------- | -------- | ------ |
| MEET-Small | 3     | 3     | 768         | 3072     | 30M    |
| MEET-Base  | 6     | 12    | 768         | 3072     | 61M    |
| MEET-Large | 12    | 16    | 1024        | 4096     | 215M   |

To simplify the model and provide a cleaner code repository, we have made minor adjustments to the model's implementation. Therefore, there might be a small difference in performance compared to the results reported in the paper.



# Usage

You can use MEET as follows:

```python
import torch
from meet.models.vit import meet_small_patch8 as create_model

model = create_model(num_classes=3)
dummy_eeg = torch.randn(8, 5, 6, 32, 32) # (batch x bands x frames x height x width)
pred = model(dummy_eeg) # (8, 3)

assert pred.shape == (8, 3)
```

