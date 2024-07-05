from lightning_sdk import Studio, Machine

# specify the teamspace name correctly
user_name = "gallegi"
teamspace_name = "PracticalML"

# initialize the studio with the correct teamspace
studio = Studio(teamspace=teamspace_name, user=user_name)

# use the jobs plugin
studio.install_plugin('jobs')
job_plugin = studio.installed_plugins['jobs']

variable = [0.05, 0.01]
config_file = "src/config/cls_vm_charades_s224_f8_exp0.yaml"
params = ""
params += f" -c {config_file}"

params += f" --layer_norm"
layers = "512 256 256"
params += f" --layers {layers}"
params += f" --init_lr {1e-4}"

num_cpus = 4  # Change this to the number of CPUs you want to use

for val in variable:
    params += f" --dropout {val}"
    cmd = f'cd PracticalML_2024 && python train_cls_head.py {params}'
    job_name = f'exp_[{layers}]_Dr{val}'
    job_plugin.run(cmd, machine=Machine.L4, cpus=num_cpus, name=job_name)
    # Machine.CPU

# GRID RUN
# do a sweep over learning rates
# learning_rates = [1e-4, 1e-3, 1e-2]
# batch_sizes = [32, 64, 128]

# a grid search combines all params
# grid_search_params = [(lr, bs) for lr in learning_rates for bs in batch_sizes]

# # start all jobs on an A10G GPU with names containing an index
# for index, (lr, bs) in enumerate(grid_search_params):
#     cmd = f'python finetune_sweep.py --lr {lr} --batch_size {bs} --max_steps {100}'
#     job_name = f'run-2-exp-{index}'
#     job_plugin.run(cmd, machine=Machine.L4, name=job_name)

