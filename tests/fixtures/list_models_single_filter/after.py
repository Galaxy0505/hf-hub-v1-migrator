from huggingface_hub import list_models


models = list_models(filter="text-classification")
