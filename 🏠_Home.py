import streamlit as st
from datasets import load_dataset
import openai
import json
import os
import tiktoken
import numpy as np
from collections import defaultdict
import pygsheets


openai.api_key = "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"

st.set_page_config(
    page_title="GPT AutoTuner",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="auto",
)

if "uploaded_metadata" not in st.session_state:
    st.session_state["uploaded_metadata"] = None

if "fine_tune_metadata" not in st.session_state:
    st.session_state["fine_tune_metadata"] = None

if "progress_metadata" not in st.session_state:
    st.session_state["progress_metadata"] = None

if "api_key" not in st.session_state:
    st.session_state["api_key"] = None

st.subheader("Enter API Key")
st.session_state["api_key"] = st.text_input("Enter your OpenAI API Key", "")
login = st.button("Done")
if login:
    openai.api_key = st.session_state["api_key"]


st.subheader("Dataset")
st.text_input(
    "Enter the path of the Hugging Face [dataset repository](https://huggingface.co/datasets)",
    "databricks/databricks-dolly-15k",
)
col1, col2 = st.columns(2)
user_col = col1.text_input(
    "The name of column containing the user prompt", "instruction"
)
assistant_col = col2.text_input(
    "The name of column containing the assitant response", "response"
)
load = st.button("Load")


def basic_prompt_format(user, assistant):
    desired_format = {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }
    return json.dumps(desired_format)


def load_json(json_file_path, user_col, assistant_col):
    with open(json_file_path, "r") as json_file, open(
        f"{json_file_path.split('.')[0]}.jsonl", "w"
    ) as jsonl_file:
        for line in json_file:
            data = json.loads(line)
            user = data[user_col]
            assistant = data[assistant_col]
            converted_line = basic_prompt_format(user, assistant)
            jsonl_file.write(converted_line + "\n")


def check_file_format():
    # We start by importing the required packages

    # Next, we specify the data path and open the JSONL file

    data_path = "train.jsonl"

    # Load dataset
    with open(data_path) as f:
        dataset = [json.loads(line) for line in f]

    # We can inspect the data quickly by checking the number of examples and the first item

    # Initial dataset stats
    print("Num examples:", len(dataset))
    print("First example:")
    for message in dataset[0]["messages"]:
        print(message)

    # Now that we have a sense of the data, we need to go through all the different examples and check to make sure the formatting is correct and matches the Chat completions message structure

    # Format error checks
    format_errors = defaultdict(int)

    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(k not in ("role", "content", "name") for k in message):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in ("system", "user", "assistant"):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            if not content or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        print("Found errors:")
        for k, v in format_errors.items():
            print(f"{k}: {v}")
    else:
        print("No errors found")

    # Beyond the structure of the message, we also need to ensure that the length does not exceed the 4096 token limit.

    # Token counting functions
    encoding = tiktoken.get_encoding("cl100k_base")

    # not exact!
    # simplified from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    def num_assistant_tokens_from_messages(messages):
        num_tokens = 0
        for message in messages:
            if message["role"] == "assistant":
                num_tokens += len(encoding.encode(message["content"]))
        return num_tokens

    def print_distribution(values, name):
        print(f"\n#### Distribution of {name}:")
        print(f"min / max: {min(values)}, {max(values)}")
        print(f"mean / median: {np.mean(values)}, {np.median(values)}")
        print(f"p5 / p95: {np.quantile(values, 0.1)}, {np.quantile(values, 0.9)}")

    # Last, we can look at the results of the different formatting operations before proceeding with creating a fine-tuning job:

    # Warnings and tokens counts
    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []

    for ex in dataset:
        messages = ex["messages"]
        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages))
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages))

    print("Num examples missing system message:", n_missing_system)
    print("Num examples missing user message:", n_missing_user)
    print_distribution(n_messages, "num_messages_per_example")
    print_distribution(convo_lens, "num_total_tokens_per_example")
    print_distribution(assistant_message_lens, "num_assistant_tokens_per_example")
    n_too_long = sum(l > 4096 for l in convo_lens)
    print(
        f"\n{n_too_long} examples may be over the 4096 token limit, they will be truncated during fine-tuning"
    )

    # Pricing and default n_epochs estimate
    MAX_TOKENS_PER_EXAMPLE = 4096

    MIN_TARGET_EXAMPLES = 100
    MAX_TARGET_EXAMPLES = 25000
    TARGET_EPOCHS = 3
    MIN_EPOCHS = 1
    MAX_EPOCHS = 25

    n_epochs = TARGET_EPOCHS
    n_train_examples = len(dataset)
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

    n_billing_tokens_in_dataset = sum(
        min(MAX_TOKENS_PER_EXAMPLE, length) for length in convo_lens
    )
    print(
        f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training"
    )
    print(f"By default, you'll train for {n_epochs} epochs on this dataset")
    print(
        f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens"
    )
    print("See pricing page to estimate total costs")


if load:
    dataset = load_dataset("databricks/databricks-dolly-15k")
    dataset["train"].to_json("train.json")
    load_json("train.json", user_col, assistant_col)
    check_file_format()
    st.session_state["uploaded_metadata"] = openai.File.create(
        file=open("train.jsonl", "rb"), purpose="fine-tune"
    )
    if st.session_state["uploaded_metadata"]["status"] == "uploaded":
        st.success("Your file has successfully been uploaded")
    st.text(st.session_state["uploaded_metadata"])

st.subheader("Fine-tuning")
fine_tune = st.button("Start Fine-tuning")
if fine_tune:
    try:
        st.session_state["fine_tune_metadata"] = openai.FineTuningJob.create(
            training_file=st.session_state["uploaded_metadata"]["id"],
            model="gpt-3.5-turbo",
        )
        st.text(st.session_state["fine_tune_metadata"])
    except:
        st.warning(
            f"File {st.session_state['uploaded_metadata']['id']} is not ready. Come back later."
        )


check_progress = st.button("Check Progress")
if check_progress:
    if not st.session_state["fine_tune_metadata"]:
        st.warning("The fine-tuned job has not started.")
    else:
        st.session_state["progress_metadata"] = openai.FineTuningJob.retrieve(
            st.session_state["fine_tune_metadata"]["id"]
        )
        st.info(
            f"The status of the fine-tuning job is {st.session_state['progress_metadata']['status']}"
        )
        st.text(progress_metadata)
