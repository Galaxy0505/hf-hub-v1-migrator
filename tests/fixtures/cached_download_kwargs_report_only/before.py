from huggingface_hub import cached_download


path = cached_download(
    "https://example.com/config.json",
    use_auth_token=True,
    resume_download=True,
)
