from huggingface_hub.file_download import cached_download, hf_hub_url


path = cached_download(
    hf_hub_url("bert-base-uncased", "config.json"),
    resume_download=True,
)
