# %% Setting up

import gradio as gr

from donaldjpt.inference import load_trained_model, prompt_model
from donaldjpt.utils import get_project_root

# Loading the pre-trained tokenizer and model
tokenizer, model = load_trained_model()

# %% Building the interface

# Logo to be displayed on top of the app
ROOT = get_project_root()
logo_path = ROOT.joinpath("assets/logo.png")


# Function to interact with "inference.py"
def prompt_model_via_app(prompt: str, max_new_tokens: int) -> str:
    result = prompt_model(
        tokenizer, model, prompt=prompt, max_new_tokens=max_new_tokens
    )
    return result


# Build the app
with gr.Blocks() as demo:
    # Render the logo first
    gr.Image(
        logo_path,
        show_label=False,
        container=False,
        height=120,
    )

    # Create the interface inside the Blocks context
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Prompt")
            input_tokens = gr.Number(
                value=50, minimum=1, maximum=125, label="Max New Tokens"
            )
            submit_btn = gr.Button("Generate")

    output_text = gr.Textbox(label="Output")

    # Connect the button to the function
    submit_btn.click(
        fn=prompt_model_via_app,
        inputs=[input_text, input_tokens],
        outputs=output_text,
        api_name="predict",
    )

# Launch the app
demo.launch()


# %%
