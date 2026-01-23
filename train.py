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
    "use_bilora" : 1
}

list1 = ["both"]
list2 = ["separate"]
list3 = [1]
combinations = [(x, y, z) for x in list1 for y in list2 for z in list3]

#     "apply_bilora_for" : "local",
#    "bilora_mode" : "aggregate",
#    "skip_task_0" : 0

list_params = []
for combination in combinations:
    params = copy.deepcopy(origin_params)
    params["apply_bilora_for"] = combination[0]
    params["bilora_mode"] = combination[1]
    params["skip_task_0"] = combination[2]
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