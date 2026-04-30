from huggingface_hub import list_models


models = list_models(task="text-classification", library="pytorch")
