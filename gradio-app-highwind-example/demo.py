from typing import Dict

import gradio as gr
import highwind

# Replace this with the unique id of your UseCase on Highwind
# This can be found by viewing the UseCase on the Highwind frontend and extracting it from the URL:
# https://frontend.dev.highwind.cloud/use_cases/{{use-case-id-to-copy}}
USE_CASE_ID: str = "..."

OUTPUT_MAPPING: Dict[int, str] = {
    0: "Bad input",
    1: "Iris-Setosa",
    2: "Iris-Versicolour",
    3: "Iris-Virginica",
}


def classify_flower(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float,
    request: gr.Request,
) -> str:
    """
    This is the function that takes input from the Gradio App, constructs the inference
    payload, and sends to the Highwind UseCase.

    Once a response is received, it can also format the output nicely for display on
    the Gradio App.

    +----------------------------------------------------------------------------------+
    | NOTE: The final named parameter is the Request object which needs to be used to  |
    | extract the User's Highwind access token.                                        |
    +----------------------------------------------------------------------------------+

    Parameters:
        - sepal_length -> float: The iris flower's Sepal length (in cm)
        - sepal_width -> float: The iris flower's Sepal width (in cm)
        - petal_length -> float: The iris flower's Petal length (in cm)
        - petal_width -> float: The iris flower's Petal width (in cm)

    Returns:
        - flower_name -> str: The name of the flower or "Bad input" if the flower type
                              cannot be determined.
    """
    # Initialize Highwind to work with this Gradio App.
    # This sets up and returns the Highwind client, but it is not necessary to use the
    # client. It will be used automatically in the background.
    highwind.GradioApp.setup_with_request(request)

    # Initialize the Highwind UseCase
    # If the user does not have access to this UseCase (for example, if they did not
    # create it, or have not purchased it, this will raise a HTTP 403 exception)
    use_case: highwind.UseCase = highwind.UseCase(id=USE_CASE_ID)

    inference_payload: Dict = {
        "inputs": [
            {
                "name": "input-0",
                "shape": [1],
                "datatype": "BYTES",
                "parameters": None,  # This model does not accept any additional parameters
                "data": [
                    [
                        sepal_length,
                        sepal_width,
                        petal_length,
                        petal_width,
                    ]  # The model accepts an Array of Arrays of floats/numerics
                ],
            }
        ]
    }

    inference_result: Dict = use_case.run_inference(inference_payload)
    result_index: int = inference_result["outputs"][0]["data"][0]
    flower_name: str = OUTPUT_MAPPING[result_index]

    return f"This is a '{flower_name}'."


# Here, we define the Gradio App using the Blocks API
# See: https://www.gradio.app/docs/gradio/blocks
#
# +------------------------------------------------------------------------------------+
# | NOTE: The Gradio App **HAS TO** be named "demo"!                                   |
# +------------------------------------------------------------------------------------+
#
with gr.Blocks() as demo:
    gr.Markdown("# Highwind Iris Classifier Demo")
    with gr.Row():
        with gr.Column():
            sepal_length = gr.Slider(0.5, 10, label="Sepal length (cm)", value=6.1)
            sepal_width = gr.Slider(0.5, 10, label="Sepal width (cm)", value=2.8)
            petal_length = gr.Slider(0.5, 10, label="Petal length (cm)", value=4.7)
            petal_width = gr.Slider(0.5, 10, label="Petal width (cm)", value=1.2)
            btn = gr.Button("What iris flower is this?")
        with gr.Column():
            flower_type = gr.Textbox(label="Flower type")

    btn.click(
        fn=classify_flower,  # There is no need to pass the Request - this is passed automatically
        inputs=[sepal_length, sepal_width, petal_length, petal_width],
        outputs=flower_type,
    )

    # We include a logout button for when the Gradio App is hosted on Highwind. This
    # button does nothing when running this app locally.
    gr.Button("Logout", link="/logout")

# You can uncomment the flowing line in order to test the Gradio App locally before
# Deploying on Highwind.
#
# +------------------------------------------------------------------------------------+
# | NOTE: When deploying this to Highwind, the demo should not be launched here!       |
# | Highwind wraps the Gradio App in a FastAPI application, and launching it here will |
# | break the application on Highwind!                                                 |
# +------------------------------------------------------------------------------------+
#
# demo.launch()
