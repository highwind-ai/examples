## Usage

1. Compile the pipeline
2. Upload the pipeline (IR YAML) to Kubeflow

The pipeline is written using the [Lightweight Python Component](https://v1-7-branch.kubeflow.org/docs/components/pipelines/v2/components/lightweight-python-components/) approach.

### Compile the pipeline

```python
poetry install --no-root
poetry shell
python -m pipeline
# A new file tensorflow_machine_translation_training_pipeline.yaml is created.
# This file is the IR YAML file that can be uploaded to Highwind.
```

