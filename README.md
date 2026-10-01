# DonaldJPT: a minimalistic language model example

- Author: github.com/cyrilby
- Last meaningful update: 01-10-2026

This repository contains a minimalistic pipeline for training a small language model based on custom data imported from a CSV file and then using the model for inference via a tiny web app. Essentially, this is a custom implementation of the `GPT2LMHeadModel` model.

## Requirements

- Python 3.14
- Dependencies as listed in `pyproject.toml` (can be installed on Windows by running `requirements_install.bat`)
- Dataset `speeches.csv` downloaded from the source (see next section for more info on that)

## Data

### Data sources

The model is trained exclusively on data from US president Donald J. Trump's various speeches. The dataset can be downloaded [here](https://huggingface.co/datasets/coastalcph/populism-trump-chronos/blob/main/train.csv).

### Data processing

- Import from CSV
- Combine separate sentences into a single corpus

## Model training

Training is designed to be very minimalistic on purpose. Therefore, the number of hidden layers, embeddings, etc. is significantly lower compared to what you would normally choose if you want to train a proper, useful model.

Model training (including data import and tokenization) is implemented via the `train_gpt2_model()` function in the `training.py` script and consists of the following steps:

1. Import data from CSV file and combine separate speeches into a single corpus.
2. Tokenize the corpus using a local copy of the GPT2 `AutoTokenizer`, incl. minor adjustments such as `pad_token = eos_tolen`, which are needed for GPT2 model training.
3. Convert the output to a `Dataset` that can be used by the `transofmers` library directly.
4. Specify custom configurations for the `GPT2LMHeadModel` model via `GPT2Config`.
5. Specify `TrainingArguments`, `DataCollator` and `Trainer`, the run the actual training process.
6. Export the final model and its tokenizer to the `.model` folder so that they can be used for inference.

## Inference

Inference can be called programmatically via the `prompt_model()` function. However, to make it easier to use for non-technical people, a demo web app based on the `gradio` framework has been added.

You can run the `app.py` script or (if you're running Windows), by running the `launch_app.bat` script.