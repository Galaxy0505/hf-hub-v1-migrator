from huggingface_hub import cached_download, hf_hub_url


path = cached_download(hf_hub_url("huggingface/label-files", "imagenet-1k-id2label.json", repo_type="dataset"))
