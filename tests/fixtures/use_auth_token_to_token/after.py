from huggingface_hub import hf_hub_download


path = hf_hub_download(
    repo_id="bert-base-uncased",
    filename="config.json",
    token=True,
)
