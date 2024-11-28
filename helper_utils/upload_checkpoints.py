from huggingface_hub import hf_hub_download, login, upload_file

def authenticate_huggingface(api_token):
    try:
        login(token=api_token)
        print("Authentication successful!")
    except Exception as e:
        print(f"Authentication failed: {e}")

def download_from_huggingface(repo_id, filename, repo_type, api_token, dest_path=None):
    """
    Downloads a file from a Hugging Face repository and optionally moves it to a destination path.
    
    Args:
        repo_id (str): Repository ID in the format "username/repo_name".
        filename (str): Name of the file to download.
        repo_type (str): Type of the repository ("model", "dataset", etc.).
        api_token (str): Hugging Face API token.
        dest_path (str): Optional. Destination directory where the file will be saved.
    
    Returns:
        str: Final path to the downloaded file.
    """
    try:
        
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            token =api_token,
            local_dir=dest_path
        )
        print(f"File downloaded to: {file_path}")
        return file_path
    except Exception as e:
        print(f"Download failed: {e}")
        return None


def upload_to_huggingface(local_path, remote_path, repo_id, repo_type, commit_message, api_token):
    """
    Uploads a file to a Hugging Face repository.
    
    Args:
        local_path (str): Path to the local file to be uploaded.
        remote_path (str): Path in the repository to upload the file.
        repo_id (str): Repository ID in the format "username/repo_name".
        repo_type (str): Type of the repository ("model", "dataset", etc.).
        commit_message (str): Commit message for the upload.
        api_token (str): Hugging Face API token.
    """
    try:
        upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message=commit_message,
            token=api_token
        )
        print(f"File uploaded to: https://huggingface.co/{repo_id}/blob/main/{remote_path}")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":

    TOKEN = ""

    # authenticate_huggingface(TOKEN)

    # download_from_huggingface(
    #     repo_id="shredder-31/GAR",
    #     filename="outputs.zip",
    #     repo_type="model",
    #     api_token=TOKEN,
    #     dest_path="/teamspace/studios/this_studio/Group-Activity-Recognition/modeling/baseline 3/outputs"
    # )

    # upload_to_huggingface(
    #     local_path="/kaggle/working/outputs.zip",
    #     remote_path="outputs.zip", 
    #     repo_id="shredder-31/GAR",
    #     repo_type="model",
    #    commit_message="upload model checkpoints", 
    #    api_token=TOKEN
    # )
