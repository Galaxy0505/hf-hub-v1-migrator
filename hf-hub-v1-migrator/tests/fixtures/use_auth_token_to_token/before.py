from huggingface_hub import hf_hub_download


path = hf_hub_download(
    repo_id="bert-base-uncased",
    filename="config.json",
    use_auth_token=True,
    resume_download=True,
)
