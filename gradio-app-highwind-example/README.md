# Gradio App Highwind Example

This example shows how to create a [Gradio App](https://www.gradio.app/) and deploy it
using Highwind.

## Step 1: Create a Gradio App Asset

1. On Highwind, create a new Asset and select "Gradio App" as the Asset type.
2. Upload a single Python file that follows the format of `demo.py` in this folder.

Things to ensure before you upload the Python file:

1. Ensure that your inference function initializes Highwind with the user's access token:

    ```py
    import highwind

    def my_inference_function(
        ..., # your model inputs
        request: gr.Request,
    ):
        highwind.GradioApp.setup_with_request(request)
    ```

2. Ensure that your Gradio app is named "demo":

    ```py
    # Example 1: with Gradio Blocks
    with gr.Blocks() as demo:
      # ... build your Gradio App

    # Example 2: with Gradio Interface
    demo = gr.Interface(...)
    ```

3. Ensure that you **DO NOT** launch the demo in your file:

    ```py
    # Please either comment out this line, or remove it entirely:
    # demo.launch()
    ```

## Step 2: Create a Use Case

1. On Highwind, create a new Use Case
2. Add the Asset that you created to the Use Case
3. When viewing the Use Case, the Gradio App should now be visible
