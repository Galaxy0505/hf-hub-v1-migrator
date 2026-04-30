from huggingface_hub import hf_hub_download


path = hf_hub_download(
    "bert-base-uncased", "config.json",
)
