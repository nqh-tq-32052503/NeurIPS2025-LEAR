def download_all_dataset(HF_TOKEN: str):
    import os
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    from huggingface_hub import snapshot_download

    hf_data = "./data"
    if not os.path.exists(hf_data):
        os.makedirs(hf_data)

    list_hf_datasets = [
        "nguyenhieu3205xt/chest-x-data",
        "nguyenhieu3205xt/cub-200-data",
        "nguyenhieu3205xt/imagenet-r-data",
        "nguyenhieu3205xt/tiny-imagenet-data",
        "nguyenhieu3205xt/euro-sat-data",
        "nguyenhieu3205xt/resisc-45-data",
        "nguyenhieu3205xt/crop-disease-data"
        
    ]


    for dataset_id in list_hf_datasets:
        print(f"Downloading {dataset_id}....")
        snapshot_download(
            repo_id=dataset_id, # nguyenhiext/your-dataset-name
            repo_type="dataset", # hoặc "model"
            local_dir=hf_data,
            token=HF_TOKEN
        )