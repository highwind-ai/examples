# HuggingFace Machine Translation Example

This folder contains the instructions to train and deploy a translation model (from English to French) on Highwind.

The process is based on this tutorial: https://huggingface.co/docs/transformers/tasks/translation

A Jupyter notebook: `huggingface-en-fr.ipynb` is provided which has the code to run the entire tutorial from preprocessing -> training -> inference.

## Folder structure

```bash
.
├── deployment
├── tf_saved_model
└── training
```

1. The `deployment` folder contains instructions on how to create a Kserve custom predictor to be deployed on Highwind.
    1. The `main.py` contains the inference and deployment code from `huggingface-en-fr.ipynb`.
1. The `tf_saved_model` folder contains the pretrained Tensorflow model with tokenizer configurations.
1. The `training` folder contains the instructions on how to create a Kubeflow Pipelines training pipeline to be run on Highwind.
    1. The `pipeline.py` contains the training code from `huggingface-en-fr.ipynb`.
