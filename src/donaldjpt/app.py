# %% Setting up

import gradio as gr

from donaldjpt.inference import load_trained_model, prompt_model

# Loading the pre-trained tokenizer and model
tokenizer, model = load_trained_model()

# %% Building the interface


def prompt_model_via_app(prompt: str) -> str:
    result = prompt_model(tokenizer, model, prompt)
    return result


demo = gr.Interface(
    fn=prompt_model_via_app, inputs=["text"], outputs=["text"], api_name="predict"
)

demo.launch()
