import copy
reproduced_datasets = "seq-cifar10,seq-cropdisease,seq-mnist,seq-resisc45,seq-eurosat-rgb,seq-tinyimg,seq-cifar100,seq-chestx,seq-imagenet-r,seq-cub200"
origin_params = {
    "list_datasets" : reproduced_datasets,
    "dataset" : "seq-cifar10",
    "ncls_per_task" : 0,
    "model" : "LEAR",
    "lr" : 0.03,
    "batch_size" : 64,
    "n_epochs" : 5,
    "num_workers" : 8,
    "backbone" : "lear",
    "transform_type" : "weak",
    "n_frq" : 9000,
    "n_experts" : 32,
    "topk" : 3
}


list_params = []
params = copy.deepcopy(origin_params)
list_params.append(params)
    
list_cmds = []
for params in list_params:
    cmd = "python -u main_domain.py"
    for key in params:
        param_cmd = f" --{key} {params[key]}"
        cmd += param_cmd
    list_cmds.append(cmd)

import os

for cmd in list_cmds:
    try:
        print("[INFO CMD] Running: ", cmd)    
        os.system(cmd)
    except Exception as e:
        print(e)