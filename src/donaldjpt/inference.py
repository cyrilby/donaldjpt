# %% Setting up
from transformers import AutoTokenizer, GPT2LMHeadModel

from donaldjpt.utils import get_project_root

# %% Fn to load pre-trained model and tokenizer


def load_trained_model():
    """
    Loads the pre-trained model and tokenizer from the
    local '.model' folder so they can be used for inference.
    """

    # Finding the project's root folder
    ROOT = get_project_root()
    model_path = ROOT.joinpath(".model")

    # Loading tokenizer and pre-trained model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    return tokenizer, model


# %% Fn to prompt model and return output


def prompt_model(
    tokenizer,
    model,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> str:

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    return generated_text


# %% Testing the pipeline

if __name__ == "__main__":

    # Loading the pre-trained tokenizer and model
    tokenizer, model = load_trained_model()

    # Prompt the model
    prompt = "China is"
    prompt_model(tokenizer, model, prompt)


# %%
