#!/usr/bin/python3

import os
from typing import Optional
from huggingface_hub import HfApi, upload_folder, upload_file, snapshot_download
from requests.exceptions import HTTPError
from wasabi import Printer
from pathlib import Path

msg = Printer()

class HuggingFaceUtils():
    def __init__(self):
        pass

    def check_repo(self, repo_id, token):
        repo_url = HfApi().create_repo(
            repo_id=repo_id,
            token=token,
            private=True,
            exist_ok=True,
        )
        return repo_url

    def load_online_model(self, project_name, config_file_name, log_dir_name):
        online_model = None
        try:
            self.pull_from_hub(
                repo_id=project_name+"/"+config_file_name,
                filename=log_dir_name,
            )
            online_model_exists = True
        except HTTPError as http_err:
            if http_err.response.status_code == 404:
                online_model_exists = False
            else:
                print(f"A HTTP error occurred while trying to load the online model: {http_err}")
                raise ValueError
        except Exception as general_err:
            print(f"Exception occurred while trying to load the online model: {general_err}")
            raise ValueError
        
        return online_model_exists, online_model
    
    def pull_from_hub(self, repo_id: str, filename: str):
        msg.info(f"Pulling repo {repo_id} from the Hugging Face Hub")
        filename_path = os.path.abspath(filename)
        snapshot_download(
            repo_id=repo_id,
            local_dir=filename_path
            )
        msg.good(
            f"The repo has been pulled successfully from the Hub, you can find it here: {filename_path}"
        )

    def push_to_hub(
        self,
        repo_id: str,
        filename: str,
        commit_message: str,
        token: Optional[str] = None,
    ):
        repo_url = self.check_repo(repo_id, token)

        # Add the model
        filename_path = os.path.abspath(filename)

        msg.info(f"Pushing repo {repo_id} to the Hugging Face Hub")
        repo_url = upload_folder(
            repo_id=repo_id,
            folder_path=filename_path,
            path_in_repo="",
            commit_message=commit_message,
            token=token,
        )

        msg.good(
            f"Your model has been uploaded to the Hub, you can find it here: {repo_url}"
        )
        return repo_url
    
    def push_file_to_hub(
        self,
        repo_id: str,
        filename: str,
        commit_message: str,
        token: Optional[str] = None,
    ):
        repo_url = self.check_repo(repo_id, token)

        # Add the model
        filename_path = os.path.abspath(filename)

        msg.info(f"Pushing repo {repo_id} to the Hugging Face Hub")
        repo_url = upload_file(
            repo_id=repo_id,
            path_or_fileobj=filename_path,
            path_in_repo=Path(filename_path).name,
            commit_message=commit_message,
            token=token,
        )

        msg.good(
            f"Your model has been uploaded to the Hub, you can find it here: {repo_url}"
        )
        return repo_url