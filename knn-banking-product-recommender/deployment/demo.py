from typing import Dict

import gradio as gr
import highwind

# Replace this with the unique id of your UseCase on Highwind
# This can be found by viewing the UseCase on the Highwind frontend and extracting it from the URL:
# https://frontend.dev.highwind.cloud/use_cases/{{use-case-id-to-copy}}
USE_CASE_ID: str = "..."


def get_mock_product_recommendation(
    user_id: str,
    request: gr.Request,
) -> str:
    """
    Mock version of the get_product_recommendation function.
    Can be used for simple testing where you don't want to run inference on Highwind.
    """
    recommended_items: list[str] = [
        "Credit Card",
        "Personal Loan",
        "Investment Account",
    ]

    # Format the output as a Markdown bullet point list
    formatted_output: str = f"- {'\n- '.join(recommended_items)}"
    return formatted_output


def get_product_recommendation(
    user_id: str,
    request: gr.Request,
) -> str:
    """
    This function takes a user ID and returns a list of product recommendations for them.

    Args:
        user_id (str): The ID of the user.
        request (gr.Request): The request object.

    Returns:
        str: The list of product recommendations as a Markdown bullet point list.
    """
    # Initialize Highwind to work with this Gradio App.
    highwind.GradioApp.setup_with_request(request)
    use_case: highwind.UseCase = highwind.UseCase(id=USE_CASE_ID)

    # Construct inference payload
    inference_payload: Dict = {
        "inputs": [
            {
                "name": "input-0",
                "shape": [1],
                "datatype": "INT32",
                "data": [int(user_id)],
            }
        ]
    }

    # Run inference
    inference_result: Dict = use_case.run_inference(inference_payload)
    recommended_items: list[str] = inference_result["outputs"][0]["data"]

    # Format the output as a Markdown bullet point list
    formatted_output: str = f"- {'\n- '.join(recommended_items)}"
    return formatted_output


# The Gradio App, called demo
with gr.Blocks() as demo:
    gr.Markdown("<center><h1>Banking Product Recommender</h1></center>")
    gr.Markdown(
        "<center>This app looks up pre-computed, personalised banking products for a user with a given user ID.</center>"
    )
    user_id = gr.Textbox(
        label="User ID",
        placeholder="123456",
        info="The user who you want product recommendations for",
    )
    output = gr.Textbox(
        label="Product Recommendations",
        lines=5,
        info="The products recommended for the user",
    )
    submit_button = gr.Button("Submit", variant="primary")
    submit_button.click(
        fn=get_product_recommendation,  # Choose mock or real inference function
        inputs=[user_id],
        outputs=[output],
    )

# For local testing
# demo.launch()
