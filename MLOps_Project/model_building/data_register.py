# Script for registering the data in HF data repo
import os
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo

#Repository details
repo_id = "KavinPrasathK/Tourism_Package_Prediction"
repo_type = "dataset"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))    #HF_TOKEN is stored in the GitHub secrets

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")

# Step 2: If repo doesn't exist : create repo   
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# Pushing the data to the repo
api.upload_folder(
    folder_path="MLOps_Project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
