# --apply_bilora_for local --bilora_mode separate --skip_task_0 0
[INFO CMD] Running:  python main_domain.py --list_datasets seq-cifar100,seq-cifar100,seq-cifar100 --dataset seq-cifar100 --ncls_per_task 5 --model LEAR --lr 0.03 --batch_size 32 --n_epochs 5 --num_workers 8 --backbone lear --transform_type weak --use_bilora 1 --apply_bilora_for local --bilora_mode separate --skip_task_0 0
[INFO] 10-Jan-26 17:49:24 - Running Mammoth! on 04dc83e5ff9c. (if you see this message more than once, you are probably importing something wrong)
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'repr' attribute with value False was provided to the `Field()` function, which has no effect in the context it was used. 'repr' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'frozen' attribute with value True was provided to the `Field()` function, which has no effect in the context it was used. 'frozen' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
[INFO] 10-Jan-26 17:49:34 - NumExpr defaulting to 4 threads.
/usr/local/lib/python3.11/dist-packages/deeplake/util/check_latest_version.py:32: UserWarning: A newer version of deeplake (4.4.4) is available. It's recommended that you update to the latest version using `pip install -U deeplake`.
  warnings.warn(
[WARNING] 10-Jan-26 17:49:38 - Trying to load default configuration for model LEAR but no configuration file found in None.
[WARNING] 10-Jan-26 17:49:38 - Default configuration file not found for dataset seq-cifar100. Using the defaults specified in the dataset class (if available).
[INFO] 10-Jan-26 17:49:38 - `lr_scheduler` set to multisteplr, overrides default from dataset.
[INFO] 10-Jan-26 17:49:38 - Using device cuda:0
Loading dataset: seq-cifar100
Loading dataset: seq-cifar100
Loading dataset: seq-cifar100
[INFO] 10-Jan-26 17:49:38 - `wandb_entity` and `wandb_project` not set. Disabling wandb.
[WARNING] 10-Jan-26 17:49:38 - Label noise is not available with multi-label datasets. If this is not multi-label, ignore this warning.
[INFO] 10-Jan-26 17:49:39 - Loading pretrained weights from Hugging Face hub (timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)
[INFO] 10-Jan-26 17:49:45 - [timm/vit_base_patch16_224.augreg2_in21k_ft_in1k] Safe alternative available for 'pytorch_model.bin' (as 'model.safetensors'). Loading weights using safetensors.
[INFO] 10-Jan-26 17:49:45 - Missing keys (head.weight, head.bias) discovered while loading pretrained weights. This is expected if model is being adapted.
[INFO] 10-Jan-26 17:49:46 - Loading pretrained weights from Hugging Face hub (timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)
[INFO] 10-Jan-26 17:49:47 - [timm/vit_base_patch16_224.augreg2_in21k_ft_in1k] Safe alternative available for 'pytorch_model.bin' (as 'model.safetensors'). Loading weights using safetensors.
[INFO] 10-Jan-26 17:49:47 - Missing keys (head.weight, head.bias) discovered while loading pretrained weights. This is expected if model is being adapted.
[INFO] 10-Jan-26 17:49:47 - Using backbone: lear

Using LEAR as backbone
Use BiLORA:  True
Apply BiLORA for:  local
Namespace(dataset='seq-cifar100', model='LEAR', backbone='lear', load_best_args=False, dataset_config=None, model_config='default', list_datasets='seq-cifar100,seq-cifar100,seq-cifar100', transform_type='weak', ncls_per_task=5, use_bilora=1, apply_bilora_for='local', skip_task_0=0, bilora_mode='separate', num_classes=100, seed=None, permute_classes=False, base_path='./data/', results_path='results/', device=device(type='cuda', index=0), notes=None, eval_epochs=None, non_verbose=False, disable_log=False, num_workers=8, enable_other_metrics=False, debug_mode=False, inference_only=False, code_optimization=0, distributed='no', savecheck=None, save_checkpoint_mode='safe', loadcheck=None, ckpt_name=None, start_from=None, stop_after=None, wandb_name=None, wandb_entity=None, wandb_project=None, lr=0.03, batch_size=32, label_perc=1.0, label_perc_by_class=1.0, joint=0, eval_future=False, validation=None, validation_mode='current', fitting_mode='epochs', early_stopping_patience=5, early_stopping_metric='loss', early_stopping_freq=1, early_stopping_epsilon=1e-06, n_epochs=5, n_iters=None, optimizer='sgd', optim_wd=0.0, optim_mom=0.0, optim_nesterov=False, drop_last=False, lr_scheduler='multisteplr', scheduler_mode='epoch', lr_milestones=[35, 45], sched_multistep_lr_gamma=0.1, noise_type='symmetric', noise_rate=0.0, disable_noisy_labels_cache=False, cache_path_noisy_labels='noisy_labels', conf_jobnum='734eea9a-b1bf-4056-9f75-e5cedc8326ff', conf_timestamp='2026-01-10 17:49:38.087460', conf_host='04dc83e5ff9c', conf_git_hash='d9141df2a3954ab461ad96da8cee30a4693fe2d0', minibatch_size=32, nowand=1)
begin task 1, dataset:seq-cifar100
100%|██████████| 169M/169M [00:14<00:00, 11.7MB/s]
target classes:  [81, 92, 17, 22, 21]
[INFO] 10-Jan-26 17:50:06 - Using 8 workers for the dataloader.
/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py:624: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 4, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(
target classes:  [81, 92, 17, 22, 21]
[INFO] 10-Jan-26 17:50:07 - Using 8 workers for the dataloader.
Applying BiLORA technique for current task:  0  at index:  0
Task 1 - Epoch 5: 100%|███████████████████████████████████████████████████████████████████████████████████████| 390/390 [04:59<00:00,  1.30it/s, loss_ce=0.000172, loss_nor=-0.0323, lr=0.03, ep/h=60.5]
Calculate distribution for task 1:  78%|███████▊  | 78/100 [00:14<00:04,  5.27it/s]
choose experts for evaluate: [1]
Evaluating Task 1:  20%|███████████████████▊                                                                               | 3/15 [00:02<00:08,  1.37it/s, acc_task_1=100]
Accuracy for 1 task(s): 	 [Class-IL]: 100.0 % 	 [Task-IL]: 0.0 %
	Raw accuracy values: Class-IL [100.0] | Task-IL [0]


Applying BiLORA technique for current task:  1  at index:  0
Applying BiLORA technique for current task:  1  at index:  0
Applying BiLORA technique for current task:  1  at index:  0
begin task 2, dataset:seq-cifar100
target classes:  [46, 79, 2, 40, 27]
[INFO] 10-Jan-26 17:55:26 - Using 8 workers for the dataloader.
target classes:  [46, 79, 2, 40, 27]
[INFO] 10-Jan-26 17:55:27 - Using 8 workers for the dataloader.
Choose params for task 2: 100%|██████████| 50/50 [00:09<00:00,  5.07it/s, distances=[0.59]]
load expert 1 parameters
Create new expert 2
Applying BiLORA technique for current task:  1  at index:  0
Task 2 - Epoch 5: 100%|███████████████████████████████████████████████████████████| 390/390 [05:38<00:00,  1.15it/s, loss_ce=0.0466, loss_kd=-0.965, loss_nor=-0.0307, loss_mi=3.73, lr=0.03, ep/h=53.9]
Calculate distribution for task 2:  78%|███████▊  | 78/100 [00:14<00:04,  5.29it/s]
Choose expert for evaluate: 100%|████████████████████████████████████████████████████████████████████████| 30/30 [00:06<00:00,  4.53it/s, task 2=2, distance=[2.98, 2.03]]
Evaluating:   0%|                                                                                                                                  | 0/30 [00:00<?, ?it/s]choose experts for evaluate: [1, 2]
Evaluating Task 1:  10%|█████████▊                                                                                        | 3/30 [00:01<00:15,  1.76it/s, acc_task_1=20.8]Applying BiLORA technique for current task:  2  at index:  0
Applying BiLORA technique for current task:  2  at index:  0
Applying BiLORA technique for current task:  2  at index:  0
Evaluating Task 2:  20%|███████████████████▌                                                                              | 6/30 [00:03<00:15,  1.58it/s, acc_task_2=87.5]
Accuracy for 2 task(s): 	 [Class-IL]: 54.17 % 	 [Task-IL]: 0.0 %
	Raw accuracy values: Class-IL [20.833333333333336, 87.5] | Task-IL [0, 0]


Applying BiLORA technique for current task:  2  at index:  1
Applying BiLORA technique for current task:  2  at index:  1
Applying BiLORA technique for current task:  2  at index:  1
begin task 3, dataset:seq-cifar100
target classes:  [2, 51, 44, 89, 80]
[INFO] 10-Jan-26 18:01:43 - Using 8 workers for the dataloader.
target classes:  [2, 51, 44, 89, 80]
[INFO] 10-Jan-26 18:01:44 - Using 8 workers for the dataloader.
Choose params for task 3: 100%|██████████| 50/50 [00:09<00:00,  5.04it/s, distances=[0.62, 0.64]]
load expert 1 parameters
Create new expert 3
Applying BiLORA technique for current task:  2  at index:  0
Task 3 - Epoch 5: 100%|███████████████████████████████████████████████████████████| 390/390 [05:38<00:00,  1.15it/s, loss_ce=0.0148, loss_kd=-1.48, loss_nor=-0.0315, loss_mi=0.092, lr=0.03, ep/h=53.6]
Calculate distribution for task 3:  78%|███████▊  | 78/100 [00:14<00:04,  5.28it/s]
Choose expert for evaluate: 100%|████████████████████████████████████████████████████████████████████| 45/45 [00:10<00:00,  4.47it/s, task 3=3, distance=[2.9, 2.26, 1.9]]
Evaluating:   0%|                                                                                                                                  | 0/45 [00:00<?, ?it/s]choose experts for evaluate: [1, 2, 3]
Evaluating Task 1:   7%|██████▌                                                                                           | 3/45 [00:01<00:24,  1.74it/s, acc_task_1=34.4]Applying BiLORA technique for current task:  3  at index:  0
Applying BiLORA technique for current task:  3  at index:  0
Applying BiLORA technique for current task:  3  at index:  0
Evaluating Task 2:  13%|█████████████▎                                                                                      | 6/45 [00:03<00:23,  1.68it/s, acc_task_2=25]Applying BiLORA technique for current task:  3  at index:  1
Applying BiLORA technique for current task:  3  at index:  1
Applying BiLORA technique for current task:  3  at index:  1
Evaluating Task 3:  20%|███████████████████▌                                                                              | 9/45 [00:05<00:23,  1.56it/s, acc_task_3=92.7]
Accuracy for 3 task(s): 	 [Class-IL]: 50.69 % 	 [Task-IL]: 0.0 %
## 	Raw accuracy values: Class-IL [34.375, 25.0, 92.70833333333334] | Task-IL [0, 0, 0]


Applying BiLORA technique for current task:  3  at index:  2
Applying BiLORA technique for current task:  3  at index:  2
Applying BiLORA technique for current task:  3  at index:  2
## System stats:
	Initial CPU memory usage: 2780.89 MB
	Average CPU memory usage: 3851.34 MB
	Final CPU memory usage: 4474.70 MB
	Max CPU memory usage: 4474.70 MB
	Initial GPU 0 memory usage: 1082.00 MB
	Average GPU 0 memory usage: 6188.00 MB
	Final GPU 0 memory usage: 6188.00 MB
	Max GPU 0 memory usage: 6188.00 MB
Logging results and arguments in ./data/results/class-il/seq-cifar100/LEAR/logs.pyd
# --apply_bilora_for local --bilora_mode separate --skip_task_0 1
[INFO CMD] Running:  python main_domain.py --list_datasets seq-cifar100,seq-cifar100,seq-cifar100 --dataset seq-cifar100 --ncls_per_task 5 --model LEAR --lr 0.03 --batch_size 32 --n_epochs 5 --num_workers 8 --backbone lear --transform_type weak --use_bilora 1 --apply_bilora_for local --bilora_mode separate --skip_task_0 1
[INFO] 10-Jan-26 18:08:08 - Running Mammoth! on 04dc83e5ff9c. (if you see this message more than once, you are probably importing something wrong)
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'repr' attribute with value False was provided to the `Field()` function, which has no effect in the context it was used. 'repr' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'frozen' attribute with value True was provided to the `Field()` function, which has no effect in the context it was used. 'frozen' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
[INFO] 10-Jan-26 18:08:11 - NumExpr defaulting to 4 threads.
/usr/local/lib/python3.11/dist-packages/deeplake/util/check_latest_version.py:32: UserWarning: A newer version of deeplake (4.4.4) is available. It's recommended that you update to the latest version using `pip install -U deeplake`.
  warnings.warn(
[WARNING] 10-Jan-26 18:08:12 - Trying to load default configuration for model LEAR but no configuration file found in None.
[WARNING] 10-Jan-26 18:08:12 - Default configuration file not found for dataset seq-cifar100. Using the defaults specified in the dataset class (if available).
[INFO] 10-Jan-26 18:08:12 - `lr_scheduler` set to multisteplr, overrides default from dataset.
[INFO] 10-Jan-26 18:08:12 - Using device cuda:0
Loading dataset: seq-cifar100
Loading dataset: seq-cifar100
Loading dataset: seq-cifar100
[INFO] 10-Jan-26 18:08:12 - `wandb_entity` and `wandb_project` not set. Disabling wandb.
[WARNING] 10-Jan-26 18:08:12 - Label noise is not available with multi-label datasets. If this is not multi-label, ignore this warning.
[INFO] 10-Jan-26 18:08:14 - Loading pretrained weights from Hugging Face hub (timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)
[INFO] 10-Jan-26 18:08:14 - [timm/vit_base_patch16_224.augreg2_in21k_ft_in1k] Safe alternative available for 'pytorch_model.bin' (as 'model.safetensors'). Loading weights using safetensors.
[INFO] 10-Jan-26 18:08:14 - Missing keys (head.weight, head.bias) discovered while loading pretrained weights. This is expected if model is being adapted.
[INFO] 10-Jan-26 18:08:15 - Loading pretrained weights from Hugging Face hub (timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)
[INFO] 10-Jan-26 18:08:16 - [timm/vit_base_patch16_224.augreg2_in21k_ft_in1k] Safe alternative available for 'pytorch_model.bin' (as 'model.safetensors'). Loading weights using safetensors.
[INFO] 10-Jan-26 18:08:16 - Missing keys (head.weight, head.bias) discovered while loading pretrained weights. This is expected if model is being adapted.
[INFO] 10-Jan-26 18:08:16 - Using backbone: lear

Using LEAR as backbone
Use BiLORA:  True
Apply BiLORA for:  local
Namespace(dataset='seq-cifar100', model='LEAR', backbone='lear', load_best_args=False, dataset_config=None, model_config='default', list_datasets='seq-cifar100,seq-cifar100,seq-cifar100', transform_type='weak', ncls_per_task=5, use_bilora=1, apply_bilora_for='local', skip_task_0=1, bilora_mode='separate', num_classes=100, seed=None, permute_classes=False, base_path='./data/', results_path='results/', device=device(type='cuda', index=0), notes=None, eval_epochs=None, non_verbose=False, disable_log=False, num_workers=8, enable_other_metrics=False, debug_mode=False, inference_only=False, code_optimization=0, distributed='no', savecheck=None, save_checkpoint_mode='safe', loadcheck=None, ckpt_name=None, start_from=None, stop_after=None, wandb_name=None, wandb_entity=None, wandb_project=None, lr=0.03, batch_size=32, label_perc=1.0, label_perc_by_class=1.0, joint=0, eval_future=False, validation=None, validation_mode='current', fitting_mode='epochs', early_stopping_patience=5, early_stopping_metric='loss', early_stopping_freq=1, early_stopping_epsilon=1e-06, n_epochs=5, n_iters=None, optimizer='sgd', optim_wd=0.0, optim_mom=0.0, optim_nesterov=False, drop_last=False, lr_scheduler='multisteplr', scheduler_mode='epoch', lr_milestones=[35, 45], sched_multistep_lr_gamma=0.1, noise_type='symmetric', noise_rate=0.0, disable_noisy_labels_cache=False, cache_path_noisy_labels='noisy_labels', conf_jobnum='feec348f-a22e-4ee0-a558-b5f143c83506', conf_timestamp='2026-01-10 18:08:12.596705', conf_host='04dc83e5ff9c', conf_git_hash='d9141df2a3954ab461ad96da8cee30a4693fe2d0', minibatch_size=32, nowand=1)
begin task 1, dataset:seq-cifar100
target classes:  [84, 20, 67, 40, 32]
[INFO] 10-Jan-26 18:08:18 - Using 8 workers for the dataloader.
/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py:624: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 4, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(
target classes:  [84, 20, 67, 40, 32]
[INFO] 10-Jan-26 18:08:19 - Using 8 workers for the dataloader.
Task 1 - Epoch 5: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 390/390 [03:46<00:00,  1.72it/s, loss_ce=0.0382, loss_nor=-0.0323, lr=0.03, ep/h=79.9]
Calculate distribution for task 1:  78%|███████▊  | 78/100 [00:14<00:04,  5.40it/s]
Evaluating:   0%|                                                                                                                                  | 0/15 [00:00<?, ?it/s]choose experts for evaluate: [1]
Evaluating Task 1:  20%|███████████████████▌                                                                              | 3/15 [00:01<00:07,  1.60it/s, acc_task_1=97.9]
Accuracy for 1 task(s): 	 [Class-IL]: 97.92 % 	 [Task-IL]: 0.0 %
	Raw accuracy values: Class-IL [97.91666666666666] | Task-IL [0]


Applying BiLORA technique for current task:  1  at index:  0
Applying BiLORA technique for current task:  1  at index:  0
Applying BiLORA technique for current task:  1  at index:  0
begin task 2, dataset:seq-cifar100
target classes:  [21, 89, 85, 13, 16]
[INFO] 10-Jan-26 18:12:24 - Using 8 workers for the dataloader.
target classes:  [21, 89, 85, 13, 16]
[INFO] 10-Jan-26 18:12:25 - Using 8 workers for the dataloader.
Choose params for task 2: 100%|██████████| 50/50 [00:09<00:00,  5.09it/s, distances=[1.02]]
load expert 1 parameters
Create new expert 2
Applying BiLORA technique for current task:  1  at index:  0
Task 2 - Epoch 5: 100%|█████████████████████████████████████████████████████████████| 390/390 [05:35<00:00,  1.16it/s, loss_ce=3.09, loss_kd=-0.406, loss_nor=-0.0321, loss_mi=3.95, lr=0.03, ep/h=54.4]
Calculate distribution for task 2:  78%|███████▊  | 78/100 [00:14<00:04,  5.29it/s]
Choose expert for evaluate: 100%|████████████████████████████████████████████████████████████████████████| 30/30 [00:06<00:00,  4.56it/s, task 2=2, distance=[4.01, 1.77]]
Evaluating:   0%|                                                                                                                                  | 0/30 [00:00<?, ?it/s]choose experts for evaluate: [2, 2]
Evaluating Task 1:  10%|██████████                                                                                           | 3/30 [00:01<00:15,  1.75it/s, acc_task_1=0]Applying BiLORA technique for current task:  2  at index:  1
Applying BiLORA technique for current task:  2  at index:  1
Applying BiLORA technique for current task:  2  at index:  1
Evaluating Task 2:  20%|███████████████████▌                                                                              | 6/30 [00:03<00:14,  1.60it/s, acc_task_2=47.9]
Accuracy for 2 task(s): 	 [Class-IL]: 23.96 % 	 [Task-IL]: 0.0 %
	Raw accuracy values: Class-IL [0.0, 47.91666666666667] | Task-IL [0, 0]


Applying BiLORA technique for current task:  2  at index:  1
Applying BiLORA technique for current task:  2  at index:  1
Applying BiLORA technique for current task:  2  at index:  1
begin task 3, dataset:seq-cifar100
target classes:  [24, 45, 56, 21, 60]
[INFO] 10-Jan-26 18:18:39 - Using 8 workers for the dataloader.
target classes:  [24, 45, 56, 21, 60]
[INFO] 10-Jan-26 18:18:40 - Using 8 workers for the dataloader.
Choose params for task 3: 100%|██████████| 50/50 [00:09<00:00,  5.05it/s, distances=[0.98, 0.64]]
load expert 2 parameters
Create new expert 3
Applying BiLORA technique for current task:  2  at index:  1
Task 3 - Epoch 5: 100%|███████████████████████████████████████████████████████████| 390/390 [05:38<00:00,  1.15it/s, loss_ce=0.0475, loss_kd=-2.3, loss_nor=-0.0313, loss_mi=0.0228, lr=0.03, ep/h=53.3]
Calculate distribution for task 3:  78%|███████▊  | 78/100 [00:14<00:04,  5.29it/s]
Choose expert for evaluate: 100%|██████████████████████████████████████████████████████████████████| 45/45 [00:10<00:00,  4.48it/s, task 3=3, distance=[3.33, 2.11, 1.93]]
Evaluating:   0%|                                                                                                                                  | 0/45 [00:00<?, ?it/s]choose experts for evaluate: [2, 2, 3]
Evaluating Task 1:   7%|██████▋                                                                                              | 3/45 [00:01<00:24,  1.71it/s, acc_task_1=0]Applying BiLORA technique for current task:  3  at index:  1
Applying BiLORA technique for current task:  3  at index:  1
Applying BiLORA technique for current task:  3  at index:  1
Evaluating Task 2:  13%|█████████████                                                                                     | 6/45 [00:03<00:23,  1.67it/s, acc_task_2=42.7]Applying BiLORA technique for current task:  3  at index:  1
Applying BiLORA technique for current task:  3  at index:  1
Applying BiLORA technique for current task:  3  at index:  1
Evaluating Task 3:  20%|███████████████████▌                                                                              | 9/45 [00:05<00:23,  1.56it/s, acc_task_3=95.8]
Accuracy for 3 task(s): 	 [Class-IL]: 46.18 % 	 [Task-IL]: 0.0 %
##	Raw accuracy values: Class-IL [0.0, 42.70833333333333, 95.83333333333334] | Task-IL [0, 0, 0]


Applying BiLORA technique for current task:  3  at index:  2
Applying BiLORA technique for current task:  3  at index:  2
Applying BiLORA technique for current task:  3  at index:  2
## System stats:
	Initial CPU memory usage: 2648.84 MB
	Average CPU memory usage: 3627.97 MB
	Final CPU memory usage: 4209.46 MB
	Max CPU memory usage: 4209.45 MB
	Initial GPU 0 memory usage: 1082.00 MB
	Average GPU 0 memory usage: 6214.00 MB
	Final GPU 0 memory usage: 6216.00 MB
	Max GPU 0 memory usage: 6214.00 MB
Logging results and arguments in ./data/results/class-il/seq-cifar100/LEAR/logs.pyd
# --apply_bilora_for local --bilora_mode aggregate --skip_task_0 0
[INFO CMD] Running:  python main_domain.py --list_datasets seq-cifar100,seq-cifar100,seq-cifar100 --dataset seq-cifar100 --ncls_per_task 5 --model LEAR --lr 0.03 --batch_size 32 --n_epochs 5 --num_workers 8 --backbone lear --transform_type weak --use_bilora 1 --apply_bilora_for local --bilora_mode aggregate --skip_task_0 0
[INFO] 10-Jan-26 18:25:03 - Running Mammoth! on 04dc83e5ff9c. (if you see this message more than once, you are probably importing something wrong)
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'repr' attribute with value False was provided to the `Field()` function, which has no effect in the context it was used. 'repr' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'frozen' attribute with value True was provided to the `Field()` function, which has no effect in the context it was used. 'frozen' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
[INFO] 10-Jan-26 18:25:06 - NumExpr defaulting to 4 threads.
/usr/local/lib/python3.11/dist-packages/deeplake/util/check_latest_version.py:32: UserWarning: A newer version of deeplake (4.4.4) is available. It's recommended that you update to the latest version using `pip install -U deeplake`.
  warnings.warn(
[WARNING] 10-Jan-26 18:25:07 - Trying to load default configuration for model LEAR but no configuration file found in None.
[WARNING] 10-Jan-26 18:25:07 - Default configuration file not found for dataset seq-cifar100. Using the defaults specified in the dataset class (if available).
[INFO] 10-Jan-26 18:25:07 - `lr_scheduler` set to multisteplr, overrides default from dataset.
[INFO] 10-Jan-26 18:25:07 - Using device cuda:0
Loading dataset: seq-cifar100
Loading dataset: seq-cifar100
Loading dataset: seq-cifar100
[INFO] 10-Jan-26 18:25:07 - `wandb_entity` and `wandb_project` not set. Disabling wandb.
[WARNING] 10-Jan-26 18:25:07 - Label noise is not available with multi-label datasets. If this is not multi-label, ignore this warning.
[INFO] 10-Jan-26 18:25:08 - Loading pretrained weights from Hugging Face hub (timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)
[INFO] 10-Jan-26 18:25:09 - [timm/vit_base_patch16_224.augreg2_in21k_ft_in1k] Safe alternative available for 'pytorch_model.bin' (as 'model.safetensors'). Loading weights using safetensors.
[INFO] 10-Jan-26 18:25:09 - Missing keys (head.weight, head.bias) discovered while loading pretrained weights. This is expected if model is being adapted.
[INFO] 10-Jan-26 18:25:10 - Loading pretrained weights from Hugging Face hub (timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)
[INFO] 10-Jan-26 18:25:10 - [timm/vit_base_patch16_224.augreg2_in21k_ft_in1k] Safe alternative available for 'pytorch_model.bin' (as 'model.safetensors'). Loading weights using safetensors.
[INFO] 10-Jan-26 18:25:10 - Missing keys (head.weight, head.bias) discovered while loading pretrained weights. This is expected if model is being adapted.
[INFO] 10-Jan-26 18:25:11 - Using backbone: lear

Using LEAR as backbone
Use BiLORA:  True
Apply BiLORA for:  local
Namespace(dataset='seq-cifar100', model='LEAR', backbone='lear', load_best_args=False, dataset_config=None, model_config='default', list_datasets='seq-cifar100,seq-cifar100,seq-cifar100', transform_type='weak', ncls_per_task=5, use_bilora=1, apply_bilora_for='local', skip_task_0=0, bilora_mode='aggregate', num_classes=100, seed=None, permute_classes=False, base_path='./data/', results_path='results/', device=device(type='cuda', index=0), notes=None, eval_epochs=None, non_verbose=False, disable_log=False, num_workers=8, enable_other_metrics=False, debug_mode=False, inference_only=False, code_optimization=0, distributed='no', savecheck=None, save_checkpoint_mode='safe', loadcheck=None, ckpt_name=None, start_from=None, stop_after=None, wandb_name=None, wandb_entity=None, wandb_project=None, lr=0.03, batch_size=32, label_perc=1.0, label_perc_by_class=1.0, joint=0, eval_future=False, validation=None, validation_mode='current', fitting_mode='epochs', early_stopping_patience=5, early_stopping_metric='loss', early_stopping_freq=1, early_stopping_epsilon=1e-06, n_epochs=5, n_iters=None, optimizer='sgd', optim_wd=0.0, optim_mom=0.0, optim_nesterov=False, drop_last=False, lr_scheduler='multisteplr', scheduler_mode='epoch', lr_milestones=[35, 45], sched_multistep_lr_gamma=0.1, noise_type='symmetric', noise_rate=0.0, disable_noisy_labels_cache=False, cache_path_noisy_labels='noisy_labels', conf_jobnum='9ea32f5c-07ca-4a4b-8340-963c2f75f35c', conf_timestamp='2026-01-10 18:25:07.501368', conf_host='04dc83e5ff9c', conf_git_hash='d9141df2a3954ab461ad96da8cee30a4693fe2d0', minibatch_size=32, nowand=1)
begin task 1, dataset:seq-cifar100
target classes:  [85, 73, 15, 22, 60]
[INFO] 10-Jan-26 18:25:13 - Using 8 workers for the dataloader.
/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py:624: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 4, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(
target classes:  [85, 73, 15, 22, 60]
[INFO] 10-Jan-26 18:25:14 - Using 8 workers for the dataloader.
Applying BiLORA technique for current task:  0  at index:  0
Task 1 - Epoch 5: 100%|████████████████████████████████████████████████████████████████████████████████████████| 390/390 [04:59<00:00,  1.30it/s, loss_ce=0.00572, loss_nor=-0.0323, lr=0.03, ep/h=60.4]
Calculate distribution for task 1:  78%|███████▊  | 78/100 [00:14<00:04,  5.27it/s]
Evaluating:   0%|                                                                                                                                  | 0/15 [00:00<?, ?it/s]choose experts for evaluate: [1]
Evaluating Task 1:  20%|███████████████████▊                                                                               | 3/15 [00:01<00:07,  1.55it/s, acc_task_1=100]
Accuracy for 1 task(s): 	 [Class-IL]: 100.0 % 	 [Task-IL]: 0.0 %
	Raw accuracy values: Class-IL [100.0] | Task-IL [0]


Applying BiLORA technique for current task:  1  at index:  0
Applying BiLORA technique for current task:  1  at index:  0
Applying BiLORA technique for current task:  1  at index:  0
begin task 2, dataset:seq-cifar100
target classes:  [99, 63, 87, 3, 18]
[INFO] 10-Jan-26 18:30:33 - Using 8 workers for the dataloader.
target classes:  [99, 63, 87, 3, 18]
[INFO] 10-Jan-26 18:30:34 - Using 8 workers for the dataloader.
Choose params for task 2: 100%|██████████| 50/50 [00:09<00:00,  5.02it/s, distances=[0.65]]
load expert 1 parameters
Create new expert 2
Applying BiLORA technique for current task:  1  at index:  0
Task 2 - Epoch 5: 100%|█████████████████████████████████████████████████████████| 390/390 [05:41<00:00,  1.14it/s, loss_ce=0.00494, loss_kd=-3.26, loss_nor=-0.0311, loss_mi=0.0927, lr=0.03, ep/h=53.1]
Calculate distribution for task 2:  78%|███████▊  | 78/100 [00:14<00:04,  5.21it/s]
Choose expert for evaluate: 100%|████████████████████████████████████████████████████████████████████████| 30/30 [00:06<00:00,  4.47it/s, task 2=2, distance=[1.99, 1.96]]
Evaluating:   0%|                                                                                                                                  | 0/30 [00:00<?, ?it/s]choose experts for evaluate: [1, 2]
Evaluating Task 1:  10%|█████████▉                                                                                         | 3/30 [00:01<00:16,  1.66it/s, acc_task_1=100]Applying BiLORA technique for current task:  2  at index:  0
Applying BiLORA technique for current task:  2  at index:  0
Applying BiLORA technique for current task:  2  at index:  0
Evaluating Task 2:  20%|███████████████████▌                                                                              | 6/30 [00:03<00:15,  1.53it/s, acc_task_2=97.9]
Accuracy for 2 task(s): 	 [Class-IL]: 98.96 % 	 [Task-IL]: 0.0 %
	Raw accuracy values: Class-IL [100.0, 97.91666666666666] | Task-IL [0, 0]


Applying BiLORA technique for current task:  2  at index:  1
Applying BiLORA technique for current task:  2  at index:  1
Applying BiLORA technique for current task:  2  at index:  1
begin task 3, dataset:seq-cifar100
target classes:  [7, 43, 57, 92, 97]
[INFO] 10-Jan-26 18:36:55 - Using 8 workers for the dataloader.
target classes:  [7, 43, 57, 92, 97]
[INFO] 10-Jan-26 18:36:55 - Using 8 workers for the dataloader.
Choose params for task 3: 100%|██████████| 50/50 [00:10<00:00,  4.90it/s, distances=[0.69, 1.44]]
load expert 1 parameters
Create new expert 3
Applying BiLORA technique for current task:  2  at index:  0
Task 3 - Epoch 5: 100%|█████████████████████████████████████████████████████████| 390/390 [05:46<00:00,  1.13it/s, loss_ce=0.0088, loss_kd=-0.218, loss_nor=-0.0284, loss_mi=0.0562, lr=0.03, ep/h=52.5]
Calculate distribution for task 3:  78%|███████▊  | 78/100 [00:15<00:04,  5.13it/s]
Choose expert for evaluate: 100%|██████████████████████████████████████████████████████████████████| 45/45 [00:10<00:00,  4.32it/s, task 3=3, distance=[2.24, 2.96, 2.16]]
Evaluating:   0%|                                                                                                                                  | 0/45 [00:00<?, ?it/s]choose experts for evaluate: [1, 1, 3]
Evaluating Task 1:   7%|██████▌                                                                                            | 3/45 [00:01<00:25,  1.66it/s, acc_task_1=100]Applying BiLORA technique for current task:  3  at index:  0
Applying BiLORA technique for current task:  3  at index:  0
Applying BiLORA technique for current task:  3  at index:  0
Evaluating Task 2:  13%|█████████████▍                                                                                       | 6/45 [00:03<00:24,  1.61it/s, acc_task_2=0]Applying BiLORA technique for current task:  3  at index:  0
Applying BiLORA technique for current task:  3  at index:  0
Applying BiLORA technique for current task:  3  at index:  0
Evaluating Task 3:  20%|███████████████████▌                                                                              | 9/45 [00:06<00:24,  1.50it/s, acc_task_3=97.9]
Accuracy for 3 task(s): 	 [Class-IL]: 65.97 % 	 [Task-IL]: 0.0 %
##	Raw accuracy values: Class-IL [100.0, 0.0, 97.91666666666666] | Task-IL [0, 0, 0]


Applying BiLORA technique for current task:  3  at index:  2
Applying BiLORA technique for current task:  3  at index:  2
Applying BiLORA technique for current task:  3  at index:  2
## System stats:
	Initial CPU memory usage: 2648.70 MB
	Average CPU memory usage: 3622.40 MB
	Final CPU memory usage: 4270.50 MB
	Max CPU memory usage: 4250.19 MB
	Initial GPU 0 memory usage: 1082.00 MB
	Average GPU 0 memory usage: 6190.00 MB
	Final GPU 0 memory usage: 6190.00 MB
	Max GPU 0 memory usage: 6190.00 MB
Logging results and arguments in ./data/results/class-il/seq-cifar100/LEAR/logs.pyd
# --apply_bilora_for local --bilora_mode aggregate --skip_task_0 1
[INFO CMD] Running:  python main_domain.py --list_datasets seq-cifar100,seq-cifar100,seq-cifar100 --dataset seq-cifar100 --ncls_per_task 5 --model LEAR --lr 0.03 --batch_size 32 --n_epochs 5 --num_workers 8 --backbone lear --transform_type weak --use_bilora 1 --apply_bilora_for local --bilora_mode aggregate --skip_task_0 1
[INFO] 10-Jan-26 18:43:27 - Running Mammoth! on 04dc83e5ff9c. (if you see this message more than once, you are probably importing something wrong)
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'repr' attribute with value False was provided to the `Field()` function, which has no effect in the context it was used. 'repr' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'frozen' attribute with value True was provided to the `Field()` function, which has no effect in the context it was used. 'frozen' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
[INFO] 10-Jan-26 18:43:31 - NumExpr defaulting to 4 threads.
/usr/local/lib/python3.11/dist-packages/deeplake/util/check_latest_version.py:32: UserWarning: A newer version of deeplake (4.4.4) is available. It's recommended that you update to the latest version using `pip install -U deeplake`.
  warnings.warn(
[WARNING] 10-Jan-26 18:43:32 - Trying to load default configuration for model LEAR but no configuration file found in None.
[WARNING] 10-Jan-26 18:43:32 - Default configuration file not found for dataset seq-cifar100. Using the defaults specified in the dataset class (if available).
[INFO] 10-Jan-26 18:43:32 - `lr_scheduler` set to multisteplr, overrides default from dataset.
[INFO] 10-Jan-26 18:43:32 - Using device cuda:0
Loading dataset: seq-cifar100
Loading dataset: seq-cifar100
Loading dataset: seq-cifar100
[INFO] 10-Jan-26 18:43:32 - `wandb_entity` and `wandb_project` not set. Disabling wandb.
[WARNING] 10-Jan-26 18:43:32 - Label noise is not available with multi-label datasets. If this is not multi-label, ignore this warning.
[INFO] 10-Jan-26 18:43:33 - Loading pretrained weights from Hugging Face hub (timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)
[INFO] 10-Jan-26 18:43:33 - [timm/vit_base_patch16_224.augreg2_in21k_ft_in1k] Safe alternative available for 'pytorch_model.bin' (as 'model.safetensors'). Loading weights using safetensors.
[INFO] 10-Jan-26 18:43:33 - Missing keys (head.weight, head.bias) discovered while loading pretrained weights. This is expected if model is being adapted.
[INFO] 10-Jan-26 18:43:35 - Loading pretrained weights from Hugging Face hub (timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)
[INFO] 10-Jan-26 18:43:35 - [timm/vit_base_patch16_224.augreg2_in21k_ft_in1k] Safe alternative available for 'pytorch_model.bin' (as 'model.safetensors'). Loading weights using safetensors.
[INFO] 10-Jan-26 18:43:35 - Missing keys (head.weight, head.bias) discovered while loading pretrained weights. This is expected if model is being adapted.
[INFO] 10-Jan-26 18:43:35 - Using backbone: lear

Using LEAR as backbone
Use BiLORA:  True
Apply BiLORA for:  local
Namespace(dataset='seq-cifar100', model='LEAR', backbone='lear', load_best_args=False, dataset_config=None, model_config='default', list_datasets='seq-cifar100,seq-cifar100,seq-cifar100', transform_type='weak', ncls_per_task=5, use_bilora=1, apply_bilora_for='local', skip_task_0=1, bilora_mode='aggregate', num_classes=100, seed=None, permute_classes=False, base_path='./data/', results_path='results/', device=device(type='cuda', index=0), notes=None, eval_epochs=None, non_verbose=False, disable_log=False, num_workers=8, enable_other_metrics=False, debug_mode=False, inference_only=False, code_optimization=0, distributed='no', savecheck=None, save_checkpoint_mode='safe', loadcheck=None, ckpt_name=None, start_from=None, stop_after=None, wandb_name=None, wandb_entity=None, wandb_project=None, lr=0.03, batch_size=32, label_perc=1.0, label_perc_by_class=1.0, joint=0, eval_future=False, validation=None, validation_mode='current', fitting_mode='epochs', early_stopping_patience=5, early_stopping_metric='loss', early_stopping_freq=1, early_stopping_epsilon=1e-06, n_epochs=5, n_iters=None, optimizer='sgd', optim_wd=0.0, optim_mom=0.0, optim_nesterov=False, drop_last=False, lr_scheduler='multisteplr', scheduler_mode='epoch', lr_milestones=[35, 45], sched_multistep_lr_gamma=0.1, noise_type='symmetric', noise_rate=0.0, disable_noisy_labels_cache=False, cache_path_noisy_labels='noisy_labels', conf_jobnum='0b4db1e7-e139-464c-8325-5d75ac79e7aa', conf_timestamp='2026-01-10 18:43:32.198409', conf_host='04dc83e5ff9c', conf_git_hash='d9141df2a3954ab461ad96da8cee30a4693fe2d0', minibatch_size=32, nowand=1)
begin task 1, dataset:seq-cifar100
target classes:  [32, 63, 6, 93, 57]
[INFO] 10-Jan-26 18:43:38 - Using 8 workers for the dataloader.
/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py:624: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 4, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(
target classes:  [32, 63, 6, 93, 57]
[INFO] 10-Jan-26 18:43:39 - Using 8 workers for the dataloader.
Task 1 - Epoch 5: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 390/390 [03:45<00:00,  1.73it/s, loss_ce=0.0025, loss_nor=-0.0323, lr=0.03, ep/h=80.6]
Calculate distribution for task 1:  78%|███████▊  | 78/100 [00:14<00:04,  5.43it/s]
Evaluating:   0%|                                                                                                                                  | 0/15 [00:00<?, ?it/s]choose experts for evaluate: [1]
Evaluating Task 1:  20%|███████████████████▌                                                                              | 3/15 [00:01<00:07,  1.58it/s, acc_task_1=95.8]
Accuracy for 1 task(s): 	 [Class-IL]: 95.83 % 	 [Task-IL]: 0.0 %
	Raw accuracy values: Class-IL [95.83333333333334] | Task-IL [0]


Applying BiLORA technique for current task:  1  at index:  0
Applying BiLORA technique for current task:  1  at index:  0
Applying BiLORA technique for current task:  1  at index:  0
begin task 2, dataset:seq-cifar100
target classes:  [92, 65, 66, 40, 46]
[INFO] 10-Jan-26 18:47:43 - Using 8 workers for the dataloader.
target classes:  [92, 65, 66, 40, 46]
[INFO] 10-Jan-26 18:47:44 - Using 8 workers for the dataloader.
Choose params for task 2: 100%|██████████| 50/50 [00:09<00:00,  5.01it/s, distances=[0.97]]
load expert 1 parameters
Create new expert 2
Applying BiLORA technique for current task:  1  at index:  0
Task 2 - Epoch 5: 100%|██████████████████████████████████████████████████████████████| 390/390 [05:38<00:00,  1.15it/s, loss_ce=1.82, loss_kd=-1.58, loss_nor=-0.0323, loss_mi=3.06, lr=0.03, ep/h=53.8]
Calculate distribution for task 2:  78%|███████▊  | 78/100 [00:15<00:04,  5.20it/s]
Choose expert for evaluate: 100%|█████████████████████████████████████████████████████████████████████████| 30/30 [00:06<00:00,  4.42it/s, task 2=2, distance=[3.7, 2.11]]
Evaluating:   0%|                                                                                                                                  | 0/30 [00:00<?, ?it/s]choose experts for evaluate: [2, 2]
Evaluating Task 1:  10%|██████████                                                                                           | 3/30 [00:01<00:16,  1.64it/s, acc_task_1=0]Applying BiLORA technique for current task:  2  at index:  1
Applying BiLORA technique for current task:  2  at index:  1
Applying BiLORA technique for current task:  2  at index:  1
Evaluating Task 2:  20%|████████████████████                                                                                | 6/30 [00:03<00:15,  1.51it/s, acc_task_2=76]
Accuracy for 2 task(s): 	 [Class-IL]: 38.02 % 	 [Task-IL]: 0.0 %
	Raw accuracy values: Class-IL [0.0, 76.04166666666666] | Task-IL [0, 0]


Applying BiLORA technique for current task:  2  at index:  1
Applying BiLORA technique for current task:  2  at index:  1
Applying BiLORA technique for current task:  2  at index:  1
begin task 3, dataset:seq-cifar100
target classes:  [91, 50, 1, 42, 74]
[INFO] 10-Jan-26 18:54:01 - Using 8 workers for the dataloader.
target classes:  [91, 50, 1, 42, 74]
[INFO] 10-Jan-26 18:54:02 - Using 8 workers for the dataloader.
Choose params for task 3: 100%|██████████| 50/50 [00:10<00:00,  4.90it/s, distances=[0.84, 0.75]]
load expert 2 parameters
Create new expert 3
Applying BiLORA technique for current task:  2  at index:  1
Task 3 - Epoch 5: 100%|████████████████████████████████████████████████████████████| 390/390 [05:45<00:00,  1.13it/s, loss_ce=0.747, loss_kd=-1.75, loss_nor=-0.0285, loss_mi=0.187, lr=0.03, ep/h=52.3]
Calculate distribution for task 3:  78%|███████▊  | 78/100 [00:15<00:04,  5.16it/s]
Choose expert for evaluate: 100%|██████████████████████████████████████████████████████████████████| 45/45 [00:10<00:00,  4.40it/s, task 3=3, distance=[6.76, 5.27, 2.06]]
Evaluating:   0%|                                                                                                                                  | 0/45 [00:00<?, ?it/s]choose experts for evaluate: [3, 3, 3]
Evaluating Task 1:   7%|██████▋                                                                                              | 3/45 [00:01<00:25,  1.64it/s, acc_task_1=0]Applying BiLORA technique for current task:  3  at index:  2
Applying BiLORA technique for current task:  3  at index:  2
Applying BiLORA technique for current task:  3  at index:  2
Evaluating Task 2:  13%|█████████████▍                                                                                       | 6/45 [00:03<00:24,  1.61it/s, acc_task_2=0]Applying BiLORA technique for current task:  3  at index:  2
Applying BiLORA technique for current task:  3  at index:  2
Applying BiLORA technique for current task:  3  at index:  2
Evaluating Task 3:  20%|███████████████████▌                                                                              | 9/45 [00:06<00:24,  1.48it/s, acc_task_3=90.6]
Accuracy for 3 task(s): 	 [Class-IL]: 30.21 % 	 [Task-IL]: 0.0 %
##	Raw accuracy values: Class-IL [0.0, 0.0, 90.625] | Task-IL [0, 0, 0]


Applying BiLORA technique for current task:  3  at index:  2
Applying BiLORA technique for current task:  3  at index:  2
Applying BiLORA technique for current task:  3  at index:  2
## System stats:
	Initial CPU memory usage: 2648.71 MB
	Average CPU memory usage: 3660.61 MB
	Final CPU memory usage: 4303.38 MB
	Max CPU memory usage: 4287.42 MB
	Initial GPU 0 memory usage: 1082.00 MB
	Average GPU 0 memory usage: 6214.00 MB
	Final GPU 0 memory usage: 6216.00 MB
	Max GPU 0 memory usage: 6214.00 MB
Logging results and arguments in ./data/results/class-il/seq-cifar100/LEAR/logs.pyd
# --apply_bilora_for both --bilora_mode separate --skip_task_0 0
[INFO CMD] Running:  python main_domain.py --list_datasets seq-cifar100,seq-cifar100,seq-cifar100 --dataset seq-cifar100 --ncls_per_task 5 --model LEAR --lr 0.03 --batch_size 32 --n_epochs 5 --num_workers 8 --backbone lear --transform_type weak --use_bilora 1 --apply_bilora_for both --bilora_mode separate --skip_task_0 0
[INFO] 10-Jan-26 19:00:33 - Running Mammoth! on 04dc83e5ff9c. (if you see this message more than once, you are probably importing something wrong)
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'repr' attribute with value False was provided to the `Field()` function, which has no effect in the context it was used. 'repr' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'frozen' attribute with value True was provided to the `Field()` function, which has no effect in the context it was used. 'frozen' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
[INFO] 10-Jan-26 19:00:36 - NumExpr defaulting to 4 threads.
/usr/local/lib/python3.11/dist-packages/deeplake/util/check_latest_version.py:32: UserWarning: A newer version of deeplake (4.4.4) is available. It's recommended that you update to the latest version using `pip install -U deeplake`.
  warnings.warn(
[WARNING] 10-Jan-26 19:00:37 - Trying to load default configuration for model LEAR but no configuration file found in None.
[WARNING] 10-Jan-26 19:00:37 - Default configuration file not found for dataset seq-cifar100. Using the defaults specified in the dataset class (if available).
[INFO] 10-Jan-26 19:00:37 - `lr_scheduler` set to multisteplr, overrides default from dataset.
[INFO] 10-Jan-26 19:00:37 - Using device cuda:0
Loading dataset: seq-cifar100
Loading dataset: seq-cifar100
Loading dataset: seq-cifar100
[INFO] 10-Jan-26 19:00:37 - `wandb_entity` and `wandb_project` not set. Disabling wandb.
[WARNING] 10-Jan-26 19:00:37 - Label noise is not available with multi-label datasets. If this is not multi-label, ignore this warning.
[INFO] 10-Jan-26 19:00:39 - Loading pretrained weights from Hugging Face hub (timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)
[INFO] 10-Jan-26 19:00:39 - [timm/vit_base_patch16_224.augreg2_in21k_ft_in1k] Safe alternative available for 'pytorch_model.bin' (as 'model.safetensors'). Loading weights using safetensors.
[INFO] 10-Jan-26 19:00:39 - Missing keys (head.weight, head.bias) discovered while loading pretrained weights. This is expected if model is being adapted.
[INFO] 10-Jan-26 19:00:41 - Loading pretrained weights from Hugging Face hub (timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)
[INFO] 10-Jan-26 19:00:41 - [timm/vit_base_patch16_224.augreg2_in21k_ft_in1k] Safe alternative available for 'pytorch_model.bin' (as 'model.safetensors'). Loading weights using safetensors.
[INFO] 10-Jan-26 19:00:41 - Missing keys (head.weight, head.bias) discovered while loading pretrained weights. This is expected if model is being adapted.
[INFO] 10-Jan-26 19:00:41 - Using backbone: lear

Using LEAR as backbone
Use BiLORA:  True
Apply BiLORA for:  both
Namespace(dataset='seq-cifar100', model='LEAR', backbone='lear', load_best_args=False, dataset_config=None, model_config='default', list_datasets='seq-cifar100,seq-cifar100,seq-cifar100', transform_type='weak', ncls_per_task=5, use_bilora=1, apply_bilora_for='both', skip_task_0=0, bilora_mode='separate', num_classes=100, seed=None, permute_classes=False, base_path='./data/', results_path='results/', device=device(type='cuda', index=0), notes=None, eval_epochs=None, non_verbose=False, disable_log=False, num_workers=8, enable_other_metrics=False, debug_mode=False, inference_only=False, code_optimization=0, distributed='no', savecheck=None, save_checkpoint_mode='safe', loadcheck=None, ckpt_name=None, start_from=None, stop_after=None, wandb_name=None, wandb_entity=None, wandb_project=None, lr=0.03, batch_size=32, label_perc=1.0, label_perc_by_class=1.0, joint=0, eval_future=False, validation=None, validation_mode='current', fitting_mode='epochs', early_stopping_patience=5, early_stopping_metric='loss', early_stopping_freq=1, early_stopping_epsilon=1e-06, n_epochs=5, n_iters=None, optimizer='sgd', optim_wd=0.0, optim_mom=0.0, optim_nesterov=False, drop_last=False, lr_scheduler='multisteplr', scheduler_mode='epoch', lr_milestones=[35, 45], sched_multistep_lr_gamma=0.1, noise_type='symmetric', noise_rate=0.0, disable_noisy_labels_cache=False, cache_path_noisy_labels='noisy_labels', conf_jobnum='8164b375-0a90-4781-b1af-8650ff9eccc3', conf_timestamp='2026-01-10 19:00:37.855908', conf_host='04dc83e5ff9c', conf_git_hash='d9141df2a3954ab461ad96da8cee30a4693fe2d0', minibatch_size=32, nowand=1)
begin task 1, dataset:seq-cifar100
target classes:  [86, 13, 36, 97, 28]
[INFO] 10-Jan-26 19:00:44 - Using 8 workers for the dataloader.
/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py:624: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 4, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(
target classes:  [86, 13, 36, 97, 28]
[INFO] 10-Jan-26 19:00:45 - Using 8 workers for the dataloader.
Applying BiLORA technique for current task:  0  at index:  0
Task 1 - Epoch 5: 100%|████████████████████████████████████████████████████████████████████████████████████████| 390/390 [06:12<00:00,  1.05it/s, loss_ce=0.00176, loss_nor=-0.0322, lr=0.03, ep/h=48.6]
Calculate distribution for task 1:  78%|███████▊  | 78/100 [00:14<00:04,  5.27it/s]
Evaluating:   0%|                                                                                                                                  | 0/15 [00:00<?, ?it/s]choose experts for evaluate: [1]
Evaluating Task 1:  20%|███████████████████▌                                                                              | 3/15 [00:02<00:08,  1.35it/s, acc_task_1=47.9]
Accuracy for 1 task(s): 	 [Class-IL]: 47.92 % 	 [Task-IL]: 0.0 %
	Raw accuracy values: Class-IL [47.91666666666667] | Task-IL [0]


Applying BiLORA technique for current task:  1  at index:  0
Applying BiLORA technique for current task:  1  at index:  0
Applying BiLORA technique for current task:  1  at index:  0
begin task 2, dataset:seq-cifar100
target classes:  [38, 61, 92, 65, 47]
[INFO] 10-Jan-26 19:07:17 - Using 8 workers for the dataloader.
target classes:  [38, 61, 92, 65, 47]
[INFO] 10-Jan-26 19:07:18 - Using 8 workers for the dataloader.
Choose params for task 2: 100%|██████████| 50/50 [00:09<00:00,  5.10it/s, distances=[0.84]]
load expert 1 parameters
Create new expert 2
Applying BiLORA technique for current task:  1  at index:  0
Task 2 - Epoch 5: 100%|███████████████████████████████████████████████████████████| 390/390 [07:16<00:00,  1.12s/it, loss_ce=0.108, loss_kd=-2.91, loss_nor=-0.0301, loss_mi=0.0731, lr=0.03, ep/h=41.3]
Calculate distribution for task 2:  78%|███████▊  | 78/100 [00:14<00:04,  5.29it/s]
Choose expert for evaluate: 100%|█████████████████████████████████████████████████████████████████████████| 30/30 [00:06<00:00,  4.53it/s, task 2=2, distance=[3.0, 1.84]]
Evaluating:   0%|                                                                                                                                  | 0/30 [00:00<?, ?it/s]choose experts for evaluate: [1, 2]
Evaluating Task 1:  10%|█████████▊                                                                                        | 3/30 [00:02<00:18,  1.43it/s, acc_task_1=18.8]Applying BiLORA technique for current task:  2  at index:  0
Applying BiLORA technique for current task:  2  at index:  0
Applying BiLORA technique for current task:  2  at index:  0
Evaluating Task 2:  20%|███████████████████▌                                                                              | 6/30 [00:04<00:18,  1.30it/s, acc_task_2=89.6]
Accuracy for 2 task(s): 	 [Class-IL]: 54.17 % 	 [Task-IL]: 0.0 %
	Raw accuracy values: Class-IL [18.75, 89.58333333333334] | Task-IL [0, 0]


Applying BiLORA technique for current task:  2  at index:  1
Applying BiLORA technique for current task:  2  at index:  1
Applying BiLORA technique for current task:  2  at index:  1
begin task 3, dataset:seq-cifar100
target classes:  [72, 99, 53, 11, 70]
[INFO] 10-Jan-26 19:15:13 - Using 8 workers for the dataloader.
target classes:  [72, 99, 53, 11, 70]
[INFO] 10-Jan-26 19:15:14 - Using 8 workers for the dataloader.
Choose params for task 3: 100%|██████████| 50/50 [00:09<00:00,  5.03it/s, distances=[0.94, 0.79]]
load expert 2 parameters
Create new expert 3
Applying BiLORA technique for current task:  2  at index:  1
Task 3 - Epoch 5: 100%|█████████████████████████████████████████████████████████| 390/390 [07:16<00:00,  1.12s/it, loss_ce=0.0602, loss_kd=-0.241, loss_nor=-0.0323, loss_mi=0.0044, lr=0.03, ep/h=41.4]
Calculate distribution for task 3:  78%|███████▊  | 78/100 [00:14<00:04,  5.28it/s]
Choose expert for evaluate: 100%|██████████████████████████████████████████████████████████████████| 45/45 [00:10<00:00,  4.41it/s, task 3=3, distance=[3.13, 2.43, 1.87]]
Evaluating:   0%|                                                                                                                                  | 0/45 [00:00<?, ?it/s]choose experts for evaluate: [1, 2, 3]
Evaluating Task 1:   7%|██████▌                                                                                           | 3/45 [00:02<00:30,  1.38it/s, acc_task_1=16.7]Applying BiLORA technique for current task:  3  at index:  0
Applying BiLORA technique for current task:  3  at index:  0
Applying BiLORA technique for current task:  3  at index:  0
Evaluating Task 2:  13%|█████████████                                                                                     | 6/45 [00:05<00:33,  1.16it/s, acc_task_2=82.3]Applying BiLORA technique for current task:  3  at index:  1
Applying BiLORA technique for current task:  3  at index:  1
Applying BiLORA technique for current task:  3  at index:  1
Evaluating Task 3:  20%|███████████████████▌                                                                              | 9/45 [00:07<00:30,  1.19it/s, acc_task_3=80.2]
Accuracy for 3 task(s): 	 [Class-IL]: 59.72 % 	 [Task-IL]: 0.0 %
##	Raw accuracy values: Class-IL [16.666666666666664, 82.29166666666666, 80.20833333333334] | Task-IL [0, 0, 0]


Applying BiLORA technique for current task:  3  at index:  2
Applying BiLORA technique for current task:  3  at index:  2
Applying BiLORA technique for current task:  3  at index:  2
## System stats:
	Initial CPU memory usage: 2649.64 MB
	Average CPU memory usage: 3789.96 MB
	Final CPU memory usage: 4528.64 MB
	Max CPU memory usage: 4528.64 MB
	Initial GPU 0 memory usage: 1084.00 MB
	Average GPU 0 memory usage: 8934.00 MB
	Final GPU 0 memory usage: 8934.00 MB
	Max GPU 0 memory usage: 8934.00 MB
Logging results and arguments in ./data/results/class-il/seq-cifar100/LEAR/logs.pyd
# --apply_bilora_for both --bilora_mode separate --skip_task_0 1
[INFO CMD] Running:  python main_domain.py --list_datasets seq-cifar100,seq-cifar100,seq-cifar100 --dataset seq-cifar100 --ncls_per_task 5 --model LEAR --lr 0.03 --batch_size 32 --n_epochs 5 --num_workers 8 --backbone lear --transform_type weak --use_bilora 1 --apply_bilora_for both --bilora_mode separate --skip_task_0 1
[INFO] 10-Jan-26 19:23:17 - Running Mammoth! on 04dc83e5ff9c. (if you see this message more than once, you are probably importing something wrong)
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'repr' attribute with value False was provided to the `Field()` function, which has no effect in the context it was used. 'repr' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'frozen' attribute with value True was provided to the `Field()` function, which has no effect in the context it was used. 'frozen' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
[INFO] 10-Jan-26 19:23:20 - NumExpr defaulting to 4 threads.
/usr/local/lib/python3.11/dist-packages/deeplake/util/check_latest_version.py:32: UserWarning: A newer version of deeplake (4.4.4) is available. It's recommended that you update to the latest version using `pip install -U deeplake`.
  warnings.warn(
[WARNING] 10-Jan-26 19:23:21 - Trying to load default configuration for model LEAR but no configuration file found in None.
[WARNING] 10-Jan-26 19:23:21 - Default configuration file not found for dataset seq-cifar100. Using the defaults specified in the dataset class (if available).
[INFO] 10-Jan-26 19:23:21 - `lr_scheduler` set to multisteplr, overrides default from dataset.
[INFO] 10-Jan-26 19:23:22 - Using device cuda:0
Loading dataset: seq-cifar100
Loading dataset: seq-cifar100
Loading dataset: seq-cifar100
[INFO] 10-Jan-26 19:23:22 - `wandb_entity` and `wandb_project` not set. Disabling wandb.
[WARNING] 10-Jan-26 19:23:22 - Label noise is not available with multi-label datasets. If this is not multi-label, ignore this warning.
[INFO] 10-Jan-26 19:23:23 - Loading pretrained weights from Hugging Face hub (timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)
[INFO] 10-Jan-26 19:23:23 - [timm/vit_base_patch16_224.augreg2_in21k_ft_in1k] Safe alternative available for 'pytorch_model.bin' (as 'model.safetensors'). Loading weights using safetensors.
[INFO] 10-Jan-26 19:23:23 - Missing keys (head.weight, head.bias) discovered while loading pretrained weights. This is expected if model is being adapted.
[INFO] 10-Jan-26 19:23:25 - Loading pretrained weights from Hugging Face hub (timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)
[INFO] 10-Jan-26 19:23:25 - [timm/vit_base_patch16_224.augreg2_in21k_ft_in1k] Safe alternative available for 'pytorch_model.bin' (as 'model.safetensors'). Loading weights using safetensors.
[INFO] 10-Jan-26 19:23:25 - Missing keys (head.weight, head.bias) discovered while loading pretrained weights. This is expected if model is being adapted.
[INFO] 10-Jan-26 19:23:25 - Using backbone: lear

Using LEAR as backbone
Use BiLORA:  True
Apply BiLORA for:  both
Namespace(dataset='seq-cifar100', model='LEAR', backbone='lear', load_best_args=False, dataset_config=None, model_config='default', list_datasets='seq-cifar100,seq-cifar100,seq-cifar100', transform_type='weak', ncls_per_task=5, use_bilora=1, apply_bilora_for='both', skip_task_0=1, bilora_mode='separate', num_classes=100, seed=None, permute_classes=False, base_path='./data/', results_path='results/', device=device(type='cuda', index=0), notes=None, eval_epochs=None, non_verbose=False, disable_log=False, num_workers=8, enable_other_metrics=False, debug_mode=False, inference_only=False, code_optimization=0, distributed='no', savecheck=None, save_checkpoint_mode='safe', loadcheck=None, ckpt_name=None, start_from=None, stop_after=None, wandb_name=None, wandb_entity=None, wandb_project=None, lr=0.03, batch_size=32, label_perc=1.0, label_perc_by_class=1.0, joint=0, eval_future=False, validation=None, validation_mode='current', fitting_mode='epochs', early_stopping_patience=5, early_stopping_metric='loss', early_stopping_freq=1, early_stopping_epsilon=1e-06, n_epochs=5, n_iters=None, optimizer='sgd', optim_wd=0.0, optim_mom=0.0, optim_nesterov=False, drop_last=False, lr_scheduler='multisteplr', scheduler_mode='epoch', lr_milestones=[35, 45], sched_multistep_lr_gamma=0.1, noise_type='symmetric', noise_rate=0.0, disable_noisy_labels_cache=False, cache_path_noisy_labels='noisy_labels', conf_jobnum='bfb4ce0f-62bd-4b74-8eaa-72606238c24c', conf_timestamp='2026-01-10 19:23:21.994565', conf_host='04dc83e5ff9c', conf_git_hash='d9141df2a3954ab461ad96da8cee30a4693fe2d0', minibatch_size=32, nowand=1)
begin task 1, dataset:seq-cifar100
target classes:  [81, 58, 70, 83, 60]
[INFO] 10-Jan-26 19:23:28 - Using 8 workers for the dataloader.
/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py:624: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 4, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(
target classes:  [81, 58, 70, 83, 60]
[INFO] 10-Jan-26 19:23:29 - Using 8 workers for the dataloader.
Task 1 - Epoch 5: 100%|██████████████████████████████████████████████████████████████████████████████████████████| 390/390 [03:45<00:00,  1.73it/s, loss_ce=0.00037, loss_nor=-0.0323, lr=0.03, ep/h=81]
Calculate distribution for task 1:  78%|███████▊  | 78/100 [00:14<00:04,  5.43it/s]
Evaluating:   0%|                                                                                                                                  | 0/15 [00:00<?, ?it/s]choose experts for evaluate: [1]
Evaluating Task 1:  20%|████████████████████                                                                                | 3/15 [00:02<00:09,  1.31it/s, acc_task_1=49]
Accuracy for 1 task(s): 	 [Class-IL]: 48.96 % 	 [Task-IL]: 0.0 %
	Raw accuracy values: Class-IL [48.95833333333333] | Task-IL [0]


Applying BiLORA technique for current task:  1  at index:  0
Applying BiLORA technique for current task:  1  at index:  0
Applying BiLORA technique for current task:  1  at index:  0
begin task 2, dataset:seq-cifar100
target classes:  [82, 13, 34, 10, 5]
[INFO] 10-Jan-26 19:27:34 - Using 8 workers for the dataloader.
target classes:  [82, 13, 34, 10, 5]
[INFO] 10-Jan-26 19:27:35 - Using 8 workers for the dataloader.
Choose params for task 2: 100%|██████████| 50/50 [00:09<00:00,  5.10it/s, distances=[1.04]]
load expert 1 parameters
Create new expert 2
Applying BiLORA technique for current task:  1  at index:  0
Task 2 - Epoch 5: 100%|███████████████████████████████████████████████████████████| 390/390 [07:09<00:00,  1.10s/it, loss_ce=0.115, loss_kd=-2.94, loss_nor=-0.00586, loss_mi=0.026, lr=0.03, ep/h=42.2]
Calculate distribution for task 2:  78%|███████▊  | 78/100 [00:14<00:04,  5.29it/s]
Choose expert for evaluate: 100%|████████████████████████████████████████████████████████████████████████| 30/30 [00:06<00:00,  4.55it/s, task 2=2, distance=[4.85, 1.77]]
Evaluating:   0%|                                                                                                                                  | 0/30 [00:00<?, ?it/s]choose experts for evaluate: [2, 2]
Evaluating Task 1:  10%|██████████                                                                                           | 3/30 [00:02<00:18,  1.45it/s, acc_task_1=0]Applying BiLORA technique for current task:  2  at index:  1
Applying BiLORA technique for current task:  2  at index:  1
Applying BiLORA technique for current task:  2  at index:  1
Evaluating Task 2:  20%|███████████████████▌                                                                              | 6/30 [00:04<00:18,  1.30it/s, acc_task_2=94.8]
Accuracy for 2 task(s): 	 [Class-IL]: 47.4 % 	 [Task-IL]: 0.0 %
	Raw accuracy values: Class-IL [0.0, 94.79166666666666] | Task-IL [0, 0]


Applying BiLORA technique for current task:  2  at index:  1
Applying BiLORA technique for current task:  2  at index:  1
Applying BiLORA technique for current task:  2  at index:  1
begin task 3, dataset:seq-cifar100
target classes:  [46, 73, 26, 69, 77]
[INFO] 10-Jan-26 19:35:23 - Using 8 workers for the dataloader.
target classes:  [46, 73, 26, 69, 77]
[INFO] 10-Jan-26 19:35:25 - Using 8 workers for the dataloader.
Choose params for task 3: 100%|██████████| 50/50 [00:09<00:00,  5.02it/s, distances=[1.05, 0.54]]
load expert 2 parameters
Create new expert 3
Applying BiLORA technique for current task:  2  at index:  1
Task 3 - Epoch 5: 100%|███████████████████████████████████████████████████████████| 390/390 [07:16<00:00,  1.12s/it, loss_ce=0.421, loss_kd=-0.45, loss_nor=-0.0309, loss_mi=0.0147, lr=0.03, ep/h=41.6]
Calculate distribution for task 3:  78%|███████▊  | 78/100 [00:14<00:04,  5.29it/s]
Choose expert for evaluate: 100%|███████████████████████████████████████████████████████████████████| 45/45 [00:10<00:00,  4.47it/s, task 3=3, distance=[3.4, 1.95, 1.71]]
Evaluating:   0%|                                                                                                                                  | 0/45 [00:00<?, ?it/s]choose experts for evaluate: [2, 2, 3]
Evaluating Task 1:   7%|██████▋                                                                                              | 3/45 [00:02<00:29,  1.43it/s, acc_task_1=0]Applying BiLORA technique for current task:  3  at index:  1
Applying BiLORA technique for current task:  3  at index:  1
Applying BiLORA technique for current task:  3  at index:  1
Evaluating Task 2:  13%|█████████████                                                                                     | 6/45 [00:04<00:28,  1.37it/s, acc_task_2=94.8]Applying BiLORA technique for current task:  3  at index:  1
Applying BiLORA technique for current task:  3  at index:  1
Applying BiLORA technique for current task:  3  at index:  1
Evaluating Task 3:  20%|████████████████████                                                                                | 9/45 [00:07<00:29,  1.23it/s, acc_task_3=74]
Accuracy for 3 task(s): 	 [Class-IL]: 56.25 % 	 [Task-IL]: 0.0 %
##	Raw accuracy values: Class-IL [0.0, 94.79166666666666, 73.95833333333334] | Task-IL [0, 0, 0]


Applying BiLORA technique for current task:  3  at index:  2
Applying BiLORA technique for current task:  3  at index:  2
Applying BiLORA technique for current task:  3  at index:  2
## System stats:
	Initial CPU memory usage: 2647.53 MB
	Average CPU memory usage: 3709.07 MB
	Final CPU memory usage: 4417.75 MB
	Max CPU memory usage: 4399.53 MB
	Initial GPU 0 memory usage: 1084.00 MB
	Average GPU 0 memory usage: 8888.00 MB
	Final GPU 0 memory usage: 8888.00 MB
	Max GPU 0 memory usage: 8888.00 MB
Logging results and arguments in ./data/results/class-il/seq-cifar100/LEAR/logs.pyd
# --apply_bilora_for both --bilora_mode aggregate --skip_task_0 0
[INFO CMD] Running:  python main_domain.py --list_datasets seq-cifar100,seq-cifar100,seq-cifar100 --dataset seq-cifar100 --ncls_per_task 5 --model LEAR --lr 0.03 --batch_size 32 --n_epochs 5 --num_workers 8 --backbone lear --transform_type weak --use_bilora 1 --apply_bilora_for both --bilora_mode aggregate --skip_task_0 0
[INFO] 10-Jan-26 19:43:27 - Running Mammoth! on 04dc83e5ff9c. (if you see this message more than once, you are probably importing something wrong)
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'repr' attribute with value False was provided to the `Field()` function, which has no effect in the context it was used. 'repr' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'frozen' attribute with value True was provided to the `Field()` function, which has no effect in the context it was used. 'frozen' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
[INFO] 10-Jan-26 19:43:30 - NumExpr defaulting to 4 threads.
/usr/local/lib/python3.11/dist-packages/deeplake/util/check_latest_version.py:32: UserWarning: A newer version of deeplake (4.4.4) is available. It's recommended that you update to the latest version using `pip install -U deeplake`.
  warnings.warn(
[WARNING] 10-Jan-26 19:43:31 - Trying to load default configuration for model LEAR but no configuration file found in None.
[WARNING] 10-Jan-26 19:43:31 - Default configuration file not found for dataset seq-cifar100. Using the defaults specified in the dataset class (if available).
[INFO] 10-Jan-26 19:43:31 - `lr_scheduler` set to multisteplr, overrides default from dataset.
[INFO] 10-Jan-26 19:43:31 - Using device cuda:0
Loading dataset: seq-cifar100
Loading dataset: seq-cifar100
Loading dataset: seq-cifar100
[INFO] 10-Jan-26 19:43:31 - `wandb_entity` and `wandb_project` not set. Disabling wandb.
[WARNING] 10-Jan-26 19:43:31 - Label noise is not available with multi-label datasets. If this is not multi-label, ignore this warning.
[INFO] 10-Jan-26 19:43:33 - Loading pretrained weights from Hugging Face hub (timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)
[INFO] 10-Jan-26 19:43:33 - [timm/vit_base_patch16_224.augreg2_in21k_ft_in1k] Safe alternative available for 'pytorch_model.bin' (as 'model.safetensors'). Loading weights using safetensors.
[INFO] 10-Jan-26 19:43:33 - Missing keys (head.weight, head.bias) discovered while loading pretrained weights. This is expected if model is being adapted.
[INFO] 10-Jan-26 19:43:35 - Loading pretrained weights from Hugging Face hub (timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)
[INFO] 10-Jan-26 19:43:35 - [timm/vit_base_patch16_224.augreg2_in21k_ft_in1k] Safe alternative available for 'pytorch_model.bin' (as 'model.safetensors'). Loading weights using safetensors.
[INFO] 10-Jan-26 19:43:35 - Missing keys (head.weight, head.bias) discovered while loading pretrained weights. This is expected if model is being adapted.
[INFO] 10-Jan-26 19:43:35 - Using backbone: lear

Using LEAR as backbone
Use BiLORA:  True
Apply BiLORA for:  both
Namespace(dataset='seq-cifar100', model='LEAR', backbone='lear', load_best_args=False, dataset_config=None, model_config='default', list_datasets='seq-cifar100,seq-cifar100,seq-cifar100', transform_type='weak', ncls_per_task=5, use_bilora=1, apply_bilora_for='both', skip_task_0=0, bilora_mode='aggregate', num_classes=100, seed=None, permute_classes=False, base_path='./data/', results_path='results/', device=device(type='cuda', index=0), notes=None, eval_epochs=None, non_verbose=False, disable_log=False, num_workers=8, enable_other_metrics=False, debug_mode=False, inference_only=False, code_optimization=0, distributed='no', savecheck=None, save_checkpoint_mode='safe', loadcheck=None, ckpt_name=None, start_from=None, stop_after=None, wandb_name=None, wandb_entity=None, wandb_project=None, lr=0.03, batch_size=32, label_perc=1.0, label_perc_by_class=1.0, joint=0, eval_future=False, validation=None, validation_mode='current', fitting_mode='epochs', early_stopping_patience=5, early_stopping_metric='loss', early_stopping_freq=1, early_stopping_epsilon=1e-06, n_epochs=5, n_iters=None, optimizer='sgd', optim_wd=0.0, optim_mom=0.0, optim_nesterov=False, drop_last=False, lr_scheduler='multisteplr', scheduler_mode='epoch', lr_milestones=[35, 45], sched_multistep_lr_gamma=0.1, noise_type='symmetric', noise_rate=0.0, disable_noisy_labels_cache=False, cache_path_noisy_labels='noisy_labels', conf_jobnum='18b56e7c-22e5-46f9-bd98-99513a079d07', conf_timestamp='2026-01-10 19:43:31.871537', conf_host='04dc83e5ff9c', conf_git_hash='d9141df2a3954ab461ad96da8cee30a4693fe2d0', minibatch_size=32, nowand=1)
begin task 1, dataset:seq-cifar100
target classes:  [50, 22, 6, 67, 19]
[INFO] 10-Jan-26 19:43:38 - Using 8 workers for the dataloader.
/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py:624: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 4, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(
target classes:  [50, 22, 6, 67, 19]
[INFO] 10-Jan-26 19:43:39 - Using 8 workers for the dataloader.
Applying BiLORA technique for current task:  0  at index:  0
Task 1 - Epoch 5: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 390/390 [06:12<00:00,  1.05it/s, loss_ce=0.0081, loss_nor=-0.0323, lr=0.03, ep/h=48.7]
Calculate distribution for task 1:  78%|███████▊  | 78/100 [00:14<00:04,  5.26it/s]
choose experts for evaluate: [1]
Evaluating Task 1:  20%|███████████████████▌                                                                              | 3/15 [00:02<00:09,  1.31it/s, acc_task_1=56.2]
Accuracy for 1 task(s): 	 [Class-IL]: 56.25 % 	 [Task-IL]: 0.0 %
	Raw accuracy values: Class-IL [56.25] | Task-IL [0]


Applying BiLORA technique for current task:  1  at index:  0
Applying BiLORA technique for current task:  1  at index:  0
Applying BiLORA technique for current task:  1  at index:  0
begin task 2, dataset:seq-cifar100
target classes:  [89, 54, 34, 83, 33]
[INFO] 10-Jan-26 19:50:11 - Using 8 workers for the dataloader.
target classes:  [89, 54, 34, 83, 33]
[INFO] 10-Jan-26 19:50:12 - Using 8 workers for the dataloader.
Choose params for task 2: 100%|██████████| 50/50 [00:09<00:00,  5.05it/s, distances=[0.73]]
load expert 1 parameters
Create new expert 2
Applying BiLORA technique for current task:  1  at index:  0
Task 2 - Epoch 5: 100%|█████████████████████████████████████████████████████████████| 390/390 [07:21<00:00,  1.13s/it, loss_ce=2.84, loss_kd=-0.21, loss_nor=-0.0738, loss_mi=0.439, lr=0.03, ep/h=40.8]
Calculate distribution for task 2:  78%|███████▊  | 78/100 [00:14<00:04,  5.22it/s]
Choose expert for evaluate: 100%|████████████████████████████████████████████████████████████████████████| 30/30 [00:06<00:00,  4.46it/s, task 2=1, distance=[4.83, 5.43]]
Evaluating:   0%|                                                                                                                                  | 0/30 [00:00<?, ?it/s]choose experts for evaluate: [1, 1]
Evaluating Task 1:  10%|█████████▊                                                                                        | 3/30 [00:02<00:19,  1.41it/s, acc_task_1=4.17]Applying BiLORA technique for current task:  2  at index:  0
Applying BiLORA technique for current task:  2  at index:  0
Applying BiLORA technique for current task:  2  at index:  0
Evaluating Task 2:  20%|████████████████████▏                                                                                | 6/30 [00:04<00:18,  1.27it/s, acc_task_2=0]
Accuracy for 2 task(s): 	 [Class-IL]: 2.08 % 	 [Task-IL]: 0.0 %
	Raw accuracy values: Class-IL [4.166666666666666, 0.0] | Task-IL [0, 0]


Applying BiLORA technique for current task:  2  at index:  0
Applying BiLORA technique for current task:  2  at index:  0
Applying BiLORA technique for current task:  2  at index:  0
begin task 3, dataset:seq-cifar100
target classes:  [82, 10, 87, 69, 8]
[INFO] 10-Jan-26 19:58:13 - Using 8 workers for the dataloader.
target classes:  [82, 10, 87, 69, 8]
[INFO] 10-Jan-26 19:58:14 - Using 8 workers for the dataloader.
Choose params for task 3: 100%|██████████| 50/50 [00:10<00:00,  4.93it/s, distances=[0.66, 7.25]]
load expert 1 parameters
Create new expert 3
Applying BiLORA technique for current task:  2  at index:  0
Task 3 - Epoch 5: 100%|████████████████████████████████████████████████████████| 390/390 [07:28<00:00,  1.15s/it, loss_ce=0.00488, loss_kd=-0.456, loss_nor=-0.0317, loss_mi=0.0228, lr=0.03, ep/h=40.4]
Calculate distribution for task 3:  78%|███████▊  | 78/100 [00:15<00:04,  5.18it/s]
Choose expert for evaluate: 100%|█████████████████████████████████████████████████████████████████| 45/45 [00:10<00:00,  4.33it/s, task 3=3, distance=[2.51, 31.27, 1.77]]
Evaluating:   0%|                                                                                                                                  | 0/45 [00:00<?, ?it/s]choose experts for evaluate: [1, 1, 3]
Evaluating Task 1:   7%|██████▋                                                                                              | 3/45 [00:02<00:32,  1.30it/s, acc_task_1=0]Applying BiLORA technique for current task:  3  at index:  0
Applying BiLORA technique for current task:  3  at index:  0
Applying BiLORA technique for current task:  3  at index:  0
Evaluating Task 2:  13%|█████████████▍                                                                                       | 6/45 [00:05<00:30,  1.26it/s, acc_task_2=0]Applying BiLORA technique for current task:  3  at index:  0
Applying BiLORA technique for current task:  3  at index:  0
Applying BiLORA technique for current task:  3  at index:  0
Evaluating Task 3:  20%|███████████████████▌                                                                              | 9/45 [00:07<00:31,  1.14it/s, acc_task_3=77.1]
Accuracy for 3 task(s): 	 [Class-IL]: 25.69 % 	 [Task-IL]: 0.0 %
##	Raw accuracy values: Class-IL [0.0, 0.0, 77.08333333333334] | Task-IL [0, 0, 0]


Applying BiLORA technique for current task:  3  at index:  2
Applying BiLORA technique for current task:  3  at index:  2
Applying BiLORA technique for current task:  3  at index:  2
## System stats:
	Initial CPU memory usage: 2648.54 MB
	Average CPU memory usage: 3803.34 MB
	Final CPU memory usage: 4575.53 MB
	Max CPU memory usage: 4573.12 MB
	Initial GPU 0 memory usage: 1084.00 MB
	Average GPU 0 memory usage: 8914.00 MB
	Final GPU 0 memory usage: 8914.00 MB
	Max GPU 0 memory usage: 8914.00 MB
Logging results and arguments in ./data/results/class-il/seq-cifar100/LEAR/logs.pyd
# --apply_bilora_for both --bilora_mode aggregate --skip_task_0 1
[INFO CMD] Running:  python main_domain.py --list_datasets seq-cifar100,seq-cifar100,seq-cifar100 --dataset seq-cifar100 --ncls_per_task 5 --model LEAR --lr 0.03 --batch_size 32 --n_epochs 5 --num_workers 8 --backbone lear --transform_type weak --use_bilora 1 --apply_bilora_for both --bilora_mode aggregate --skip_task_0 1
[INFO] 10-Jan-26 20:06:30 - Running Mammoth! on 04dc83e5ff9c. (if you see this message more than once, you are probably importing something wrong)
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'repr' attribute with value False was provided to the `Field()` function, which has no effect in the context it was used. 'repr' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
/usr/local/lib/python3.11/dist-packages/pydantic/_internal/_generate_schema.py:2249: UnsupportedFieldAttributeWarning: The 'frozen' attribute with value True was provided to the `Field()` function, which has no effect in the context it was used. 'frozen' is field-specific metadata, and can only be attached to a model field using `Annotated` metadata or by assignment. This may have happened because an `Annotated` type alias using the `type` statement was used, or if the `Field()` function was attached to a single member of a union type.
  warnings.warn(
[INFO] 10-Jan-26 20:06:33 - NumExpr defaulting to 4 threads.
/usr/local/lib/python3.11/dist-packages/deeplake/util/check_latest_version.py:32: UserWarning: A newer version of deeplake (4.4.4) is available. It's recommended that you update to the latest version using `pip install -U deeplake`.
  warnings.warn(
[WARNING] 10-Jan-26 20:06:34 - Trying to load default configuration for model LEAR but no configuration file found in None.
[WARNING] 10-Jan-26 20:06:34 - Default configuration file not found for dataset seq-cifar100. Using the defaults specified in the dataset class (if available).
[INFO] 10-Jan-26 20:06:34 - `lr_scheduler` set to multisteplr, overrides default from dataset.
[INFO] 10-Jan-26 20:06:34 - Using device cuda:0
Loading dataset: seq-cifar100
Loading dataset: seq-cifar100
Loading dataset: seq-cifar100
[INFO] 10-Jan-26 20:06:34 - `wandb_entity` and `wandb_project` not set. Disabling wandb.
[WARNING] 10-Jan-26 20:06:34 - Label noise is not available with multi-label datasets. If this is not multi-label, ignore this warning.
[INFO] 10-Jan-26 20:06:36 - Loading pretrained weights from Hugging Face hub (timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)
[INFO] 10-Jan-26 20:06:36 - [timm/vit_base_patch16_224.augreg2_in21k_ft_in1k] Safe alternative available for 'pytorch_model.bin' (as 'model.safetensors'). Loading weights using safetensors.
[INFO] 10-Jan-26 20:06:36 - Missing keys (head.weight, head.bias) discovered while loading pretrained weights. This is expected if model is being adapted.
[INFO] 10-Jan-26 20:06:37 - Loading pretrained weights from Hugging Face hub (timm/vit_base_patch16_224.augreg2_in21k_ft_in1k)
[INFO] 10-Jan-26 20:06:38 - [timm/vit_base_patch16_224.augreg2_in21k_ft_in1k] Safe alternative available for 'pytorch_model.bin' (as 'model.safetensors'). Loading weights using safetensors.
[INFO] 10-Jan-26 20:06:38 - Missing keys (head.weight, head.bias) discovered while loading pretrained weights. This is expected if model is being adapted.
[INFO] 10-Jan-26 20:06:38 - Using backbone: lear

Using LEAR as backbone
Use BiLORA:  True
Apply BiLORA for:  both
Namespace(dataset='seq-cifar100', model='LEAR', backbone='lear', load_best_args=False, dataset_config=None, model_config='default', list_datasets='seq-cifar100,seq-cifar100,seq-cifar100', transform_type='weak', ncls_per_task=5, use_bilora=1, apply_bilora_for='both', skip_task_0=1, bilora_mode='aggregate', num_classes=100, seed=None, permute_classes=False, base_path='./data/', results_path='results/', device=device(type='cuda', index=0), notes=None, eval_epochs=None, non_verbose=False, disable_log=False, num_workers=8, enable_other_metrics=False, debug_mode=False, inference_only=False, code_optimization=0, distributed='no', savecheck=None, save_checkpoint_mode='safe', loadcheck=None, ckpt_name=None, start_from=None, stop_after=None, wandb_name=None, wandb_entity=None, wandb_project=None, lr=0.03, batch_size=32, label_perc=1.0, label_perc_by_class=1.0, joint=0, eval_future=False, validation=None, validation_mode='current', fitting_mode='epochs', early_stopping_patience=5, early_stopping_metric='loss', early_stopping_freq=1, early_stopping_epsilon=1e-06, n_epochs=5, n_iters=None, optimizer='sgd', optim_wd=0.0, optim_mom=0.0, optim_nesterov=False, drop_last=False, lr_scheduler='multisteplr', scheduler_mode='epoch', lr_milestones=[35, 45], sched_multistep_lr_gamma=0.1, noise_type='symmetric', noise_rate=0.0, disable_noisy_labels_cache=False, cache_path_noisy_labels='noisy_labels', conf_jobnum='a8b48a41-b3e9-4226-b834-3bdc802b837b', conf_timestamp='2026-01-10 20:06:34.781020', conf_host='04dc83e5ff9c', conf_git_hash='d9141df2a3954ab461ad96da8cee30a4693fe2d0', minibatch_size=32, nowand=1)
begin task 1, dataset:seq-cifar100
target classes:  [42, 16, 5, 32, 86]
[INFO] 10-Jan-26 20:06:41 - Using 8 workers for the dataloader.
/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py:624: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 4, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  warnings.warn(
target classes:  [42, 16, 5, 32, 86]
[INFO] 10-Jan-26 20:06:42 - Using 8 workers for the dataloader.
Task 1 - Epoch 5: 100%|█████████████████████████████████████████████████████████████████████████████████████████| 390/390 [03:45<00:00,  1.73it/s, loss_ce=0.000514, loss_nor=-0.0322, lr=0.03, ep/h=80]
Calculate distribution for task 1:  78%|███████▊  | 78/100 [00:14<00:04,  5.42it/s]
Evaluating:   0%|                                                                                                                                  | 0/15 [00:00<?, ?it/s]choose experts for evaluate: [1]
Evaluating Task 1:  20%|███████████████████▌                                                                              | 3/15 [00:02<00:09,  1.28it/s, acc_task_1=46.9]
Accuracy for 1 task(s): 	 [Class-IL]: 46.88 % 	 [Task-IL]: 0.0 %
	Raw accuracy values: Class-IL [46.875] | Task-IL [0]


Applying BiLORA technique for current task:  1  at index:  0
Applying BiLORA technique for current task:  1  at index:  0
Applying BiLORA technique for current task:  1  at index:  0
begin task 2, dataset:seq-cifar100
target classes:  [83, 33, 54, 43, 55]
[INFO] 10-Jan-26 20:10:46 - Using 8 workers for the dataloader.
target classes:  [83, 33, 54, 43, 55]
[INFO] 10-Jan-26 20:10:47 - Using 8 workers for the dataloader.
Choose params for task 2: 100%|██████████| 50/50 [00:09<00:00,  5.07it/s, distances=[0.95]]
load expert 1 parameters
Create new expert 2
Applying BiLORA technique for current task:  1  at index:  0
Task 2 - Epoch 5: 100%|████████████████████████████████████████████████████████████| 390/390 [07:15<00:00,  1.12s/it, loss_ce=0.163, loss_kd=-1.41, loss_nor=-0.0323, loss_mi=0.683, lr=0.03, ep/h=41.5]
Calculate distribution for task 2:  78%|███████▊  | 78/100 [00:14<00:04,  5.22it/s]
Choose expert for evaluate: 100%|████████████████████████████████████████████████████████████████████████| 30/30 [00:06<00:00,  4.48it/s, task 2=2, distance=[3.87, 1.89]]
Evaluating:   0%|                                                                                                                                  | 0/30 [00:00<?, ?it/s]choose experts for evaluate: [2, 2]
Evaluating Task 1:  10%|██████████                                                                                           | 3/30 [00:02<00:19,  1.36it/s, acc_task_1=0]Applying BiLORA technique for current task:  2  at index:  1
Applying BiLORA technique for current task:  2  at index:  1
Applying BiLORA technique for current task:  2  at index:  1
Evaluating Task 2:  20%|███████████████████▌                                                                              | 6/30 [00:04<00:19,  1.25it/s, acc_task_2=94.8]
Accuracy for 2 task(s): 	 [Class-IL]: 47.4 % 	 [Task-IL]: 0.0 %
	Raw accuracy values: Class-IL [0.0, 94.79166666666666] | Task-IL [0, 0]


Applying BiLORA technique for current task:  2  at index:  1
Applying BiLORA technique for current task:  2  at index:  1
Applying BiLORA technique for current task:  2  at index:  1
begin task 3, dataset:seq-cifar100
target classes:  [42, 7, 56, 46, 55]
[INFO] 10-Jan-26 20:18:42 - Using 8 workers for the dataloader.
target classes:  [42, 7, 56, 46, 55]
[INFO] 10-Jan-26 20:18:43 - Using 8 workers for the dataloader.
Choose params for task 3: 100%|██████████| 50/50 [00:10<00:00,  4.92it/s, distances=[0.99, 0.67]]
load expert 2 parameters
Create new expert 3
Applying BiLORA technique for current task:  2  at index:  1
Task 3 - Epoch 5: 100%|████████████████████████████████████████████████████████████| 390/390 [07:29<00:00,  1.15s/it, loss_ce=0.0458, loss_kd=-1.6, loss_nor=-0.0208, loss_mi=0.104, lr=0.03, ep/h=40.3]
Calculate distribution for task 3:  78%|███████▊  | 78/100 [00:15<00:04,  5.16it/s]
Choose expert for evaluate: 100%|███████████████████████████████████████████████████████████████████| 45/45 [00:10<00:00,  4.30it/s, task 3=3, distance=[3.0, 2.41, 1.82]]
Evaluating:   0%|                                                                                                                                  | 0/45 [00:00<?, ?it/s]choose experts for evaluate: [3, 2, 3]
Evaluating Task 1:   7%|██████▌                                                                                           | 3/45 [00:02<00:31,  1.33it/s, acc_task_1=17.7]Applying BiLORA technique for current task:  3  at index:  2
Applying BiLORA technique for current task:  3  at index:  2
Applying BiLORA technique for current task:  3  at index:  2
Evaluating Task 2:  13%|█████████████                                                                                     | 6/45 [00:04<00:30,  1.27it/s, acc_task_2=66.7]Applying BiLORA technique for current task:  3  at index:  1
Applying BiLORA technique for current task:  3  at index:  1
Applying BiLORA technique for current task:  3  at index:  1
Evaluating Task 3:  20%|███████████████████▌                                                                              | 9/45 [00:07<00:31,  1.14it/s, acc_task_3=87.5]
Accuracy for 3 task(s): 	 [Class-IL]: 57.29 % 	 [Task-IL]: 0.0 %
##	Raw accuracy values: Class-IL [17.708333333333336, 66.66666666666666, 87.5] | Task-IL [0, 0, 0]


Applying BiLORA technique for current task:  3  at index:  2
Applying BiLORA technique for current task:  3  at index:  2
Applying BiLORA technique for current task:  3  at index:  2
## System stats:
	Initial CPU memory usage: 2648.73 MB
	Average CPU memory usage: 3777.44 MB
	Final CPU memory usage: 4571.80 MB
	Max CPU memory usage: 4562.64 MB
	Initial GPU 0 memory usage: 1084.00 MB
	Average GPU 0 memory usage: 8888.00 MB
	Final GPU 0 memory usage: 8888.00 MB
	Max GPU 0 memory usage: 8888.00 MB
Logging results and arguments in ./data/results/class-il/seq-cifar100/LEAR/logs.pyd