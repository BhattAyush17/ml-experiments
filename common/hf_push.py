from huggingface_hub import HfApi

def push_to_hf(local_path, repo_id):
    api = HfApi()
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=local_path,
        repo_id=repo_id,
        repo_type="model"
    )
