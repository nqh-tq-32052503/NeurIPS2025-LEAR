from main_domain import parse_args, extend_args, check_args
from datasets import get_dataset_names, get_all_datasets


reproduced_datasets = "seq-cifar10,seq-cropdisease,seq-mnist,seq-resisc45,seq-eurosat-rgb,seq-tinyimg,seq-cifar100,seq-chestx,seq-imagenet-r,seq-cub200"
import sys, shlex
cmd = f"--list_datasets {reproduced_datasets} --dataset seq-cifar100 --model LEAR --lr 0.03 --batch_size 32 --n_epochs 2 --num_workers 0 --backbone lear --ncls_per_task 0"
sys.argv = ['main_domain.py'] + shlex.split(cmd)
args = parse_args()

datasets = get_all_datasets(args)
extend_args(args, datasets[0])
check_args(args, dataset=datasets[0])


list_dataloaders = []
for dataset in datasets:
    try:
        a, b = dataset.get_all_data_loaders()
        list_dataloaders.append(a)
        print("[SUCCESS]", dataset)
    except Exception as e:
        print("[FAIL]", dataset)
        print(e)
    print("-------------------")
     