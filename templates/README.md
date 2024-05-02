# Template details

### Overview
- This folder contains templates for KFP components:
    1. `Lightweight Python Components`: This is a component from a self-contained Python function ie: A stand-alone Python function which will contain all your project logic within a single script.  

    2. `Containerized Python Components`: These components extend `Lightweight Python Components` by relaxing the constraint that `Lightweight Python Components` be hermetic (i.e., fully self-contained). This means `Containerized Python Component` functions can depend on symbols defined outside of the function, imports outside of the function, code in adjacent Python modules, etc. 

- **NB:** For further details please refer to [Kubeflow v2 Components](https://www.kubeflow.org/docs/components/pipelines/v2/components/)


### Choice of template
- The selection of a template depends on the project's complexity.

- For simple projects which can be done within a single script, the `Lightweight Python Components` template is sufficent. 

- For more complex projects which require more than one script, the `Containerized Python Components` should be employed.


### Template usage
- Inside each template folder, you'll find a `ReadMe` document containing instructions for setting up and utilizing the template.

- Once you've selected a template, you can duplicate the corresponding folder and proceed to configure and initialize your project according to the instructions in the `ReadMe`.