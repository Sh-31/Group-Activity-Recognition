import os
import kagglehub
from huggingface_hub import login, upload_file, list_repo_files, hf_hub_download


def authenticate_huggingface(api_token):
    try:
        login(token=api_token)
        print("Authentication successful!")
    except Exception as e:
        print(f"Authentication failed: {e}")

def upload_directory_to_huggingface(local_dir, remote_dir, repo_id, repo_type, commit_message, api_token):
    """
    Uploads an entire directory to a Hugging Face repository.

    Args:
        local_dir (str): Path to the local directory to be uploaded.
        remote_dir (str): Path in the repository to mirror the directory.
        repo_id (str): Repository ID in the format "username/repo_name".
        repo_type (str): Type of the repository ("model", "dataset", etc.).
        commit_message (str): Commit message for the upload.
        api_token (str): Hugging Face API token.
    """
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            # Create a relative path for the remote location
            relative_path = os.path.relpath(local_path, start=local_dir)
            remote_path = os.path.join(remote_dir, relative_path).replace("\\", "/")  # Ensure Unix-style paths

            try:
                upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=remote_path,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    commit_message=commit_message,
                    token=api_token
                )
                print(f"Uploaded: {local_path} -> {remote_path}")
            except Exception as e:
                print(f"Failed to upload {local_path}: {e}")

def download_entire_repo(repo_id, local_folder, repo_type, api_token):
    """
    Downloads all files from the root of a Hugging Face repository to a local directory.

    Args:
        repo_id (str): Repository ID in the format "username/repo_name".
        local_folder (str): Local directory where the files will be saved.
        repo_type (str): Type of the repository ("model", "dataset", etc.).
        api_token (str): Hugging Face API token.
    """
    try:
        files = list_repo_files(repo_id=repo_id, repo_type=repo_type, token=api_token)
        print(f"Files in repo: {files}")
        
        os.makedirs(local_folder, exist_ok=True)

        for remote_file in files:
            local_file_path = os.path.join(local_folder, remote_file)
            local_file_dir = os.path.dirname(local_file_path)
            os.makedirs(local_file_dir, exist_ok=True)  # Ensure the directory exists
            
            # Download file
            hf_hub_download(
                repo_id=repo_id,
                filename=remote_file,
                repo_type=repo_type,
                token=api_token,
                local_dir=local_file_dir
            )
            print(f"Downloaded: {remote_file} -> {local_file_path}")
    except Exception as e:
        print(f"Download failed: {e}")



if __name__ == "__main__":
   
    TOKEN = "hf_"  
    LOCAL_DIRECTORY = "/teamspace/studios/this_studio/Group-Activity-Recognition/modeling"
    REMOTE_DIRECTORY = ""  # Path in the Hugging Face repository
    REPO_ID = "shredder-31/GAR"  # Replace with your repo ID
    REPO_TYPE = "model"  # Type of repository (e.g., "model", "dataset")
    COMMIT_MESSAGE = "Upload baseline 4 model files and outputs"

    # Authenticate
    # authenticate_huggingface(TOKEN)

    # # Upload directory to Hugging Face
    # upload_directory_to_huggingface(
    #     local_dir=LOCAL_DIRECTORY,
    #     remote_dir=REMOTE_DIRECTORY,
    #     repo_id=REPO_ID,
    #     repo_type=REPO_TYPE,
    #     commit_message=COMMIT_MESSAGE,
    #     api_token=TOKEN
    # )

    # download_entire_repo(
    #     repo_id=REPO_ID,
    #     local_folder=LOCAL_DIRECTORY,
    #     repo_type=REPO_TYPE,
    #     api_token=TOKEN
    # )

    import kagglehub

    # Download latest version
    path = kagglehub.dataset_download("ahmedai31/b8-final")

    print("Path to dataset files:", path)