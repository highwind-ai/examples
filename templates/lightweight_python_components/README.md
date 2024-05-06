# Template details

## Overview
- This folder contains sub-folders for:
    - `notebooks`: This contains the exploratory  notebooks
    - `deployment`: This will house all files, folders, scripts..etc for deployment
    - `saved_model` This will contain the saved model which can be created from your Notebook as part of saving the model
    - `training`: This will contain a single `pipeline` script

## Getting Started
1. Start by creating a folder structure for your project or copy the following folder `lightweight_python_components`

2. After creating/copying the folders, you can proceed to create the notebooks for exploratory analysis and testing purposes within the `notebooks` folder.

3. Proceed to populate the `pipeline.py` within the `training` folder file based on your project specfics
   - See existing list of examples in the example section below as a reference. 

4. `Deployment folder`: Please see the `deployment` folder set-up for the examples listed below when making updates for deployment
    - For the `main.py` script, you are required to inlcude `predict` and `model load` functions
    - The rest of the files in the folder can be copied and adjusted to your project requirements

5. Generate the IR YAML file
   - Proceed to execute the script to compile the IR YAML file using (Please see `Kubeflow Caveats`):
    ```python
            python -m pipeline
           # A new yaml file is created.
    ```

6.  This file is the IR YAML file that can be uploaded to Highwind.

7. **OPTIONAL**: Local testing of the KFP pipeline
   - **NB:** Please see Highwind user manual for local testing steps
   - This step is optional, but if your pipeline runs successfully locally, you can then proceed to upload the generated IR YAML file to Highwind
   - Ensure you swicth your kfp version from `2.7.0` back to `2.0.0b15` prior to generating the IR YAML file as local testing requires kfp version `2.7.0`.


### Kubeflow Caveats
- **Caveat 1 - Optional Args:** After compiling the IR YAML file, please ensure it does not contain `isOptional: true` as this will result in an invalid IR YAML.

- **Caveat 2 - Pipeline Description:** Before compiling the IR YAML file, please ensure within the `@dsl.pipeline` section, there is no description included as this will result in a failure to submit the IR YAML onto Highwind due to formatting issues.
    - This can also be resolved after compiling the IR YAML by searching for `description` which should appear within the section called `pipelineInfo` if it is present, then remove description and save the IR YAML.


### Python Containerised Examples
1. California Housing

### Dependency Management
- The various examples makes use of poetry for dependency management
- For further details on poetry, please refer to [python poetry website](https://python-poetry.org/docs/)
- `NB:` You are free to use any dependency management tools help manage project dependencies and package installations.