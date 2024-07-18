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
docker build --build-arg CACHEBUST=$(date +%s) -t $HEADNODE_HOSTNAME:5000/local/mlperf-nvidia-stable_diffusion-pyt:latest .
docker push $HEADNODE_HOSTNAME:5000/local/mlperf-nvidia-stable_diffusion-pyt:latest

# Verify if the image has been pushed to the registry
curl http://$HEADNODE_HOSTNAME:5000/v2/_catalog
```

## Prepare dataset

```
export HEADNODE_HOSTNAME=$(hostname)
export DATAPATH=/home/ubuntu/ml-1cc/data/mlperf/stable_diffusion
sudo mkdir -p $DATAPATH
sudo chmod -R 777 $DATAPATH

sbatch  --export=HEADNODE_HOSTNAME=$HEADNODE_HOSTNAME,DATAPATH=$DATAPATH dataset.sub
```

It took a couple of days to get all data prepared, with ~36 hours spent on downloading the LAION400m dataset. Once finished, the following folders should be created: 
```
ubuntu@ml-512-head-002:~/ml-1cc/data/mlperf$ tree stable_diffusion/ -L 2
stable_diffusion/
├── clip
│   ├── models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K
│   ├── open_clip_config.json
│   └── open_clip_pytorch_model.bin
├── coco2014
│   ├── val2014_30k.tsv
│   └── val2014_30k_stats.npz
├── inception
│   └── pt_inception-2015-12-05-6726825d.pth
├── laion-400m
│   ├── webdataset-moments-filtered
│   └── webdataset-moments-filtered-encoded
└── sd
    └── 512-base-ema.ckpt
```


## Run training

```
# Single node
export HEADNODE_HOSTNAME=$(hostname) && \
source ./config_1cc_01x08x64.sh && \
sbatch -N1 --ntasks-per-node=8 --gres=gpu:8 run_1cc_nccl.sub

# 2x nodes



# 4x nodes

```

You should see training finished with log like this
```
 
```



# Troubleshoot


1. __libnuma: Warning: cpu argument 91-103,195-207 is out of range__

Miss in run_1cc.sub ?
```
#SBATCH --exclusive
#SBATCH --mem=0
```