from huggingface_hub import hf_hub_download


path = hf_hub_download("huggingface/label-files", "imagenet-1k-id2label.json", repo_type="dataset")
