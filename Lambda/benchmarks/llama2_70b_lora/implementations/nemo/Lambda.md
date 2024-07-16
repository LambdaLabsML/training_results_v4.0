# Running mlcommon BERT benchmark on Lambda 1-Click Clusters

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
export HEADNODE_HOSTNAME=ml-512-head-001
docker build --build-arg CACHEBUST=$(date +%s) -t $HEADNODE_HOSTNAME:5000/local/mlperf-nvidia-llama2_70b_lora:latest .
docker push $HEADNODE_HOSTNAME:5000/local/mlperf-nvidia-llama2_70b_lora:latest

# Verify if the image has been pushed to the registry
curl http://$HEADNODE_HOSTNAME:5000/v2/_catalog
```

## Prepare dataset

```
# see if scripts are in the right place
docker run -it $HEADNODE_HOSTNAME:5000/local/mlperf-nvidia-llama2_70b_lora:latest
```

```
export HEADNODE_HOSTNAME=$(hostname)
export DATAPATH=/home/ubuntu/ml-1cc/data/mlperf/llama2_70b_lora/data
export MODELPATH=/home/ubuntu/ml-1cc/data/mlperf/llama2_70b_lora/ckpt
sudo mkdir -p $DATAPATH
sudo chmod -R 777 $DATAPATH
sudo mkdir -p $MODELPATH
sudo chmod -R 777 $MODELPATH
sbatch -N1 --ntasks-per-node=1 --export=HEADNODE_HOSTNAME=$HEADNODE_HOSTNAME,DATAPATH=$DATAPATH,MODELPATH=$MODELPATH dataset.sub
```

## Run training

```
# Single node
export HEADNODE_HOSTNAME=$(hostname) && \
source configs/config_1cc_1x8x4xtp4pp1cp1.sh && \
sbatch -N1 --ntasks-per-node=8 --gres=gpu:8 run_1cc.sub

# 2x nodes
export HEADNODE_HOSTNAME=$(hostname) && \
source configs/config_1cc_2x8x4xtp4pp1cp2.sh && \
sbatch -N2 --ntasks-per-node=8 --gres=gpu:8 run_1cc.sub

# 4x nodes
export HEADNODE_HOSTNAME=$(hostname) && \
source configs/config_1cc_4x8x4xtp4pp1cp2.sh && \
sbatch -N4 --ntasks-per-node=8 --gres=gpu:8 run_1cc.sub
```

You should see training finished with log like this
```
0: :::MLLOG {"namespace": "", "time_ms": 1721026606541, "event_type": "INTERVAL_START", "key": "block_start", "value": null, "metadata": {"file": "/workspace/ft-llm/custom_callbacks.py", "lineno": 129, "samples_count": 2688}}
0: :::MLLOG {"namespace": "", "time_ms": 1721026805975, "event_type": "POINT_IN_TIME", "key": "tracked_stats", "value": {"throughput": 1.9257918929116467}, "metadata": {"file": "/workspace/ft-llm/custom_callbacks.py", "lineno": 161, "step": 3072}}
0: :::MLLOG {"namespace": "", "time_ms": 1721026805975, "event_type": "INTERVAL_END", "key": "block_stop", "value": null, "metadata": {"file": "/workspace/ft-llm/custom_callbacks.py", "lineno": 112, "samples_count": 3072}}
0: :::MLLOG {"namespace": "", "time_ms": 1721026805975, "event_type": "INTERVAL_START", "key": "eval_start", "value": null, "metadata": {"file": "/workspace/ft-llm/custom_callbacks.py", "lineno": 117, "samples_count": 3072}}
0: :::MLLOG {"namespace": "", "time_ms": 1721026850352, "event_type": "POINT_IN_TIME", "key": "eval_accuracy", "value": 0.9234074354171753, "metadata": {"file": "/workspace/ft-llm/custom_callbacks.py", "lineno": 181, "samples_count": 3072}}
0: :::MLLOG {"namespace": "", "time_ms": 1721026850352, "event_type": "INTERVAL_END", "key": "eval_stop", "value": null, "metadata": {"file": "/workspace/ft-llm/custom_callbacks.py", "lineno": 186, "samples_count": 3072}}
0: :::MLLOG {"namespace": "", "time_ms": 1721026850352, "event_type": "INTERVAL_END", "key": "run_stop", "value": 0.9234074354171753, "metadata": {"file": "/workspace/ft-llm/custom_callbacks.py", "lineno": 195, "samples_count": 3072, "status": "success"}}
```


# Troubleshoot

1. __ImportError: cannot import name 'ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST'__

```
ImportError: cannot import name 'ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST' from 'transformers' (/usr/local/lib/python3.10/dist-packages/transformers/__init__.py)
```
Discussions:[NVIDIA/NeMo/issues/9318](https://github.com/NVIDIA/NeMo/issues/9318#issuecomment-2176619464)
Solution: `RUN pip install transformers==4.40.2` in Dockerfile. 


2. __Error: ucx send failed: Destination is unreachable__
Use [Supermicro config](https://github.com/mlcommons/training_results_v4.0/tree/main/Supermicro/benchmarks/llama2_70b_lora/implementations/AS-8125GS-TNHR_8_H100-SXM-80GB)

`config_common_1cc.sh`: 
```
export NCCL_CFG_PATH="conf/nccl/custom_communicator_cta.yaml"
export TP_COMM_OVERLAP=False
export MC_TP_OVERLAP_AG=True
export MC_TP_OVERLAP_RS=True
export MC_TP_OVERLAP_RS_DGRAD=False
export CUBLAS_FORCE_XMMA_KERNEL_INIT=DEVICE
export NVTE_RS_STRIDED_ATOMIC=2
export LORA_A2A=1
```

Also in experiment configs e.g. `config_1cc_1x8x4xtp4pp1cp1.sh`:
```
export TP_COMM_OVERLAP=False
```