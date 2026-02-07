
# Importing necessary libraries
from huggingface_hub import HfApi
import os

# Uploading the deployment folder to Hugging Face
api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="MLOps_Project/deployment",               
    repo_id="KavinPrasathK/Tourism_Package_Prediction",           # Target Repo
    repo_type="space",                                            # Repo type              
    path_in_repo="",                          
)
