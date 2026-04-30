from huggingface_hub.constants import hf_cache_home


def read_cache(hf_cache_home: str):
    return hf_cache_home


hf_cache_home = "local-cache"
local = hf_cache_home
