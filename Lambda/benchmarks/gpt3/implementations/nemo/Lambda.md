# Running mlcommon GPT3 benchmark on Lambda 1-Click Clusters

## Install SLURM/Enroot/Pyxis/Docker registry
These are pre-installed by Lambda engineers. 

You should be able to submit SLURM job from the head node
```
ubuntu@head1:~$ sinfo
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
batch*       up   infinite      2   idle worker[1-2]
```

A docker registry should be running on the head node
```
ubuntu@head1:~$ docker ps
CONTAINER ID   IMAGE          COMMAND                  CREATED      STATUS       PORTS     NAMES
7ac6b15b8518   registry:2.8   "/entrypoint.sh /etc…"   5 days ago   Up 3 hours             deepops-registry
``` 

### Docker Configuration

On ALL nodes, remove the need of `sudo` for running docker
```
export USERNAME=$(whoami)
sudo groupadd docker
sudo usermod -aG docker $USERNAME
newgrp docker
```

On ALL nodes, make sure Docker daemon is configured to allow insecure registries. To do so, add the following to `/etc/docker/daemon.json` (create the file if it doesn't exist). Notice `<head-node-hostname>` should exists in `etc/hosts` of all nodes.  

```
{
 "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
  "insecure-registries": ["<head-node-hostname>"]
}
```

Run `sudo systemctl restart docker` on all nodes after the change.

### Enroot Setting
On worker nodes, check if the following variables are set correctly in `/etc/enroot/enroot.conf` for all workers nodes:
```
ENROOT_MOUNT_HOME     no  # Otherwise enroot-mount: failed to mount: /home/ubuntu/ml-1cc/mnt at /tmp/enroot-data/user-0/pyxis_76.0/mnt: Permission denied
ENROOT_ALLOW_HTTP     yes # Otherwise slurmstepd: error: pyxis:     [INFO] Querying registry for permission grant
```

On worker nodes, update the file system permission for enroot-related folders: 
```
sudo chmod 1777 /tmp/enroot-data
sudo chmod 1777 /run/enroot
```

## Buid Docker Container

```
# Build the container and push to local registry
# Currently head node will crash during docker build, so better use a worker node to build the image and push to the head node registery
export HEADNODE_HOSTNAME=ml-64-head-001
docker build --build-arg CACHEBUST=$(date +%s) -t $HEADNODE_HOSTNAME:5000/local/mlperf-nvidia-gpt3:latest .
docker push $HEADNODE_HOSTNAME:5000/local/mlperf-nvidia-gpt3:latest

# Verify if the image has been pushed to the registry
curl http://$HEADNODE_HOSTNAME:5000/v2/_catalog
```

## Prepare dataset

Follow this [instruction](https://github.com/mlcommons/training/blob/master/large_language_model/megatron-lm/README.md#s3-artifacts-download). Just download and unzip.

From a worker node:
```
mkdir -p ~/ml-1cc/data/mlperf/gpt3 && \
cd ~/ml-1cc/data/mlperf/gpt3

curl https://rclone.org/install.sh | sudo bash

rclone config create mlc-training s3 provider=Cloudflare access_key_id=76ea42eadb867e854061a1806220ee1e secret_access_key=a53625c4d45e3ca8ac0df8a353ea3a41ffc3292aa25259addd8b7dc5a6ce2936 endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com

cd /home/ubuntu/ml-1cc/data/mlperf/gpt3 && mkdir c4 && cd c4 && \
rclone copy mlc-training:mlcommons-training-wg-public/gpt3/megatron-lm/dataset_c4_spm.tar ./ -P
tar -xvf dataset_c4_spm.tar

cd /home/ubuntu/ml-1cc/data/mlperf/gpt3 && mkdir checkpoint_megatron_fp32 && cd checkpoint_megatron_fp32 && \
rclone copy mlc-training:mlcommons-training-wg-public/gpt3/megatron-lm/checkpoint_megatron_fp32.tar ./ -P

# Follow guide here
# https://github.com/mlcommons/training/blob/master/large_language_model/megatron-lm/README.md#bf16-training
cd /home/ubuntu/ml-1cc/data/mlperf/gpt3 && mkdir checkpoint_nemo_bf16 && cd checkpoint_nemo_bf16 && \
rclone copy mlc-training:mlcommons-training-wg-public/gpt3/megatron-lm/checkpoint_nemo_bf16.tar ./ -P

tar -xvf checkpoint_nemo_bf16.tar

# https://github.com/mlcommons/training_results_v4.0/tree/6fe954378efe4a1a1bd665550b18ff8f26018def/NVIDIA/benchmarks/gpt3/implementations/nemo#model-checkpoint
# "The postprocessing step can be skipped - the gpt3/megatron-lm/checkpoint_nemo_bf16.tar is already NeMo-compatible after unpacking."
# python ./scripts/json_to_torch.py -i ./scripts/common_bf16.json -o /home/ubuntu/ml-1cc/data/mlperf/gpt3/ckpt4000-consumed_samples=0/common.pt

```

## Run training

```
# 8x nodes (will OOM)
export HEADNODE_HOSTNAME=calvin-training-head-003 && \
source ./config_1cc_8x8x32x8x4_mbs2.sh && \
sbatch -N8 --gres=gpu:8 run_1cc_8x_test_4.sub

# 16x nodes (minimal num_gpus to get things to work)
export HEADNODE_HOSTNAME=calvin-training-head-003 && \
source ./config_1cc_16x8x128x8x4_mbs2.sh && \
sbatch -N16 --ntasks-per-node=8 --gres=gpu:8 run_1cc.sub
```

You should see training finished with log like this
```
  0: ENDING TIMING RUN AT 2024-08-28 10:31:22 AM
  0: RESULT,large language model,29046,20724,root,2024-08-28 04:45:58 AM
```

# Troubleshoot

1. __srun: unrecognized option '--container-env=MASTER_PORT,MASTER_ADDR,NCCL_SHARP_GROUP_SIZE_THRESH'__

Change `--container-env=MASTER_PORT,MASTER_ADDR,NCCL_SHARP_GROUP_SIZE_THRESH` to `--export=ALL,MASTER_PORT=${MASTER_PORT},MASTER_ADDR=${MASTER_ADDR},NCCL_SHARP_GROUP_SIZE_THRESH=${NCCL_SHARP_GROUP_SIZE_THRESH}`


2. __ImportError: cannot import name 'ModelFilter' from 'huggingface_hub'__

```
pip install huggingface-hub==0.22.0
```