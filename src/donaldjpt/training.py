# %% Setting up

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
)

from donaldjpt.utils import clean_dir_if_exists, get_project_root

# %% Importing the data


def load_raw_data(
    file_name: str = "train.csv",
    data_col: str = "sentence",
    data_points_limit: int | None = None,
) -> str:

    # Finding the project's root folder
    ROOT = get_project_root()

    # Loading dataset from CSV file
    data_file = ROOT.joinpath(f"data/{file_name}")
    raw_data = pd.read_csv(data_file, encoding="latin1", sep="	")

    # Optional limiting the number of data points for the model
    if data_points_limit:
        raw_data = raw_data.head(data_points_limit)

    # Turning sentences into a single corpus
    text = "\n".join(raw_data[data_col].tolist())

    return text


# %% Tokenizing the data


def download_tokenizer(
    tokenizer_name: str = "openai-community/gpt2",
) -> None:

    # Define path for storing tokenizer
    ROOT = get_project_root()
    local_path = ROOT.joinpath(".tokenizer")
    clean_dir_if_exists(local_path)

    # Download the tokenizer from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Note: GPT-2 doesn't have a pad token by default but it
    # is needed for training a model, so we have to add it
    tokenizer.pad_token = tokenizer.eos_token

    # Save the tokenizer
    tokenizer.save_pretrained(local_path)

    print(f"Tokenizer saved to {local_path}.")


def load_local_tokenizer(
    tokenizer_path: str = ".tokenizer",
) -> AutoTokenizer:

    # Define path for storing tokenizer
    ROOT = get_project_root()
    local_path = ROOT.joinpath(tokenizer_path)

    # Load the tokenizer from the local path
    tokenizer = AutoTokenizer.from_pretrained(local_path)

    # GPT-2 doesn't have a pad token by default.
    # tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def tokenize_data(
    tokenizer: AutoTokenizer,
    text: str,
    context_length: int = 128,
) -> Dataset:

    # Tokenize the entire corpus
    tokenized = tokenizer(
        text,
        return_attention_mask=False,
    )
    input_ids = tokenized["input_ids"]

    # Discard the final incomplete chunk
    total_length = (len(input_ids) // context_length) * context_length
    input_ids = input_ids[:total_length]

    # Split into fixed-size chunks
    chunks = [
        input_ids[i : i + context_length]
        for i in range(0, total_length, context_length)
    ]

    return Dataset.from_dict(
        {
            "input_ids": chunks,
        }
    )


# %% Preparing the data for the model


def train_gpt2_model(
    data_points_limit: int | None = None,
    context_length: int = 128,
    update_tokenizer: bool = False,
    num_train_epochs: int = 3,
):

    # Prepare paths for data/model storage
    ROOT = get_project_root()
    training_dir = ROOT.joinpath(".training")
    model_folder = ROOT.joinpath(".model")
    clean_dir_if_exists(training_dir)
    clean_dir_if_exists(model_folder)

    # Load the data
    main_corpus = load_raw_data(data_points_limit=data_points_limit)

    # Load the tokenizer
    if update_tokenizer:
        download_tokenizer()
    tokenizer = load_local_tokenizer()

    # Tokenize the data
    train_dataset = tokenize_data(
        tokenizer,
        main_corpus,
    )

    print(f"Training examples: {len(train_dataset)}")
    print(f"Tokens per example: {context_length}")

    # Configure the GPT2 model
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=context_length,
        n_ctx=context_length,
        # Tiny model
        n_embd=128,
        n_layer=4,
        n_head=4,
        n_inner=512,
        # Special tokens
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Intiate the model with the configs above
    model = GPT2LMHeadModel(config)
    print(f"Parameters: " f"{model.num_parameters():,}")

    # Initiate the data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Prepare the training arguments for the model
    training_args = TrainingArguments(
        output_dir=training_dir,
        num_train_epochs=num_train_epochs,
        # Increase/decrease according to your hardware
        per_device_train_batch_size=4,
        learning_rate=5e-4,
        logging_steps=10,
        save_steps=500,
        save_total_limit=1,
        report_to="none",
    )

    # Specify the model trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Run the actual training process
    print("Training model...")
    trainer.train()
    print("Training complete.")

    # Save the model and the tokenizer to the designated folder
    trainer.save_model(model_folder)
    tokenizer.save_pretrained(model_folder)

    print(f"Model saved to {model_folder}.")
    print("DONE.")


# %% Testing the pipeline

if __name__ == "__main__":

    # Test the model on a tiny subset of the data
    # and only running one epoch
    train_gpt2_model(num_train_epochs=1)


# %%
