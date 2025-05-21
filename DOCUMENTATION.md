# Repository Documentation

This document provides an overview of the files within this repository.

## Root Directory

### `.gitignore`
Specifies intentionally untracked files that Git should ignore, such as `*.ipynb_checkpoints`.

### `README.md`
## Maxtext on Vertex A3 Mega

A3 Mega on Vertex AI is in early access release. 

MaxText is a high performance, highly scalable, open-source LLM written in pure Python/Jax and targeting Google Cloud TPUs and GPUs for training and inference. 


MaxText achieves high MFUs (Model Flop Utilization) and scales from single host to very large clusters while staying simple and "optimization-free".



Maxtext additionally provided reference implementations for popular Open Source models like:
- Llama 2 and 3
- Mistral and Mixtral
- Gemma
- GPT 

These reference implementations support pre-training and full fine tuning. The key value proposition of using maxtext  for pre-training or full fine tuning is:
- Very High Performance
- Open Code Base
- Easy to understand 

MaxText aims to be a launching off point for ambitious LLM projects both in research and production. We encourage users to start by experimenting with MaxText out of the box and then fork and modify MaxText to meet their needs.


In this repo we have training examples for launching following model test runs:
- LLama2 7B
- LLama2 70

### Note

This is simple example for running LLama2-7b. You can use the setup to run the same with different models in [Maxtext](https://github.com/google/maxtext/tree/maxtext-a3plus-release/MaxText/configs/models).

### Setup
On A3 Mega it achieves as high median MFU of 55% on 2 Node A3 Mega.

```
git clone https://github.com/shivajid/a3_mega_benchmarking.git
cd a3_mega_benchmarking.git

```
To submit an A3 Mega job on Vertex you need:
- Job json file, describe the job artifact
- A Script to submit the job

We have the following file for the job json. This has all the NCCL environment specific to A3 Mega set up.
- working_config_ported_supercomputer.json 

A script that submits this job to right project and region. In the below script we are submitting to the
- Project: google.com:vertex-training-dlexamples
- Region: us-centra1

```
curl -X POST \
     -H "Authorization: Bearer $(gcloud auth print-access-token)" \
     -H "Content-Type: application/json; charset=utf-8" \
     -d @working_config_ported_supercomputer.json \
     "https://us-central1-aiplatform.googleapis.com/v1/projects/google.com:vertex-training-dlexamples/locations/us-central1/customJobs"
```

The above code is encapsulated in the file:
- working_config_ported_supercomputer.sh


### Additional Information

Maxtext is written in JAX. It needs the following to form a cluster:
- JAX_COORDINATOR_ADDRESS: IP Address of the master node
- JAX_COORDINATOR_PORT: 2222 for Vertex
- GPUS_PER_NODE: 8 For A3 Mega on GCP
- NODE_RANK: A value 0 to N-1 (Depending where this is being run)
- NNODES: Number of nodes in the cluster

Vertex provides environment variable about the MASTER ADDRESS, PORT and NODE RANK through its environment variables. The GPUS_PER_NODE and NNODES are set by the user and can be set in the environment variable by the user submitting the job in the job json file.

Since the original code is from maxtext, A3 branch. To make it easy to work with the testing we are using a pre-built docker image. The docker image is uploaded in the docker hub under:
```
aurius/maxtext-fastrak:06-11-2024
```

This source code used is https://github.com/google/maxtext/tree/maxtext-a3plus-release

To support these changes we need to dynamically update the "gpu_multi_process_run.sh" in the docker image. In the job json "working_config_ported_supercomputer.json" file command we perform the following. The current location of the file is in "gs://snap-maxtext-output/vertex_config/05". Ensure that you have a GCS Bucket where you can store as for the region where Vertex Training job is submitted.


```
"command": ["bash","-c", "cd /deps && gsutil cp gs://snap-maxtext-output/vertex_config/05/gpu_multi_process_run.sh /deps && bash gpu_multi_process_run.sh"]
```

### XLA Flags

Following are the XLA Flags used. It is present in the working_config_ported_supercomputer.json file. These are well tuned for A3 Mega on GCP.

```
--xla_dump_hlo_pass_re=.* --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_triton_gemm=false --xla_gpu_graph_level=0 --xla_gpu_enable_highest_priority_async_stream=true --xla_gpu_all_reduce_combine_threshold_bytes=536870912 --xla_gpu_all_gather_combine_threshold_bytes=134217728 --xla_gpu_reduce_scatter_combine_threshold_bytes=67108864 --xla_gpu_enable_pipelined_all_gather=true --xla_gpu_enable_pipelined_reduce_scatter=true --xla_gpu_enable_pipelined_all_reduce=true --xla_gpu_enable_while_loop_double_buffering=true --xla_gpu_enable_triton_softmax_fusion=false --xla_gpu_enable_all_gather_combine_by_dim=false --xla_gpu_enable_reduce_scatter_combine_by_dim=false --xla_disable_hlo_passes=rematerialization
```

### Command

Following is the maxtext training command:

```
python  MaxText/train.py MaxText/configs/base.yml base_output_directory=gs://snap-maxtext-output/ dataset_path=gs://snap-maxtext-dataset/   attention=cudnn_flash_te  use_iota_embed=true scan_layers=false  dcn_data_parallelism=2 ici_fsdp_parallelism=8 per_device_batch_size=4  max_target_length=4096 remat_policy=minimal_flash logits_dot_in_fp32=false  tokenizer_path=assets/tokenizer.llama2 run_name=llama2_finetune_4vm-401 steps=400 async_checkpointing=false model_name=llama2-7b checkpoint_period=200 enable_checkpointing=True hardware=gpu"
```
### Dataset
Note in the training command above is using C4 dataset. This is about 800 GB of text. The dataset is downloaded using the script is in maxtext repo:

```
bash download_dataset.sh {GCS_LOGS_BUCKET} {GCS_DATSET_BUCKET}
```

### Sharding

The default job json is setup to run on 2 nodes. 1 master and 1 worker.

We set FSDP across the GPUs in the A3 Mega: ici_fsdp_parallelism=8 

We setup Data parallelism across the nodes: dcn_data_parallelism

```
dcn_data_parallelism=2 
ici_fsdp_parallelism=8 
per_device_batch_size=4
```

### run_name

Please update the following with every run

### model_name

Maxtext has specific strings for each model. 
- LLama2 7b
- LLama2 70b

## Adding more Nodes to the Cluster

The default configuration is with 2 worker, 1 master and 1 worker:

- Update the worker count:
Change the replca count to number of workers you want. Update for the worker and not for the master:
```
 "replicaCount": "1",
```

To support the above configuration. Update the following values in the working_config_ported_supercomputer.json file:

- NNNODES: Change this is the env varible section of the 
- In the training COMMAND string, update the following to a value same as number of nodes in the Cluster:
    - dcn_data_parallelism
    - num_slices


## Submit a job
Before you submit a job ensure to edit working_config_ported_supercomputer.json" file to confuigure the 
- dataset: Download and prepare the dataset 
- Stage the gpu_multi_process_run.sh and ensure that it is reachable
- Update the run name
- Update the sharding

To submit the job run the following:
```
bash working_config_ported_supercomputer.sh
```


## Contact

Please reach out to shivajid@google.com for any questions.

### `gpu_multi_process_run.sh`
This script is designed to run on multiple GPUs in a distributed environment. It sets up various NCCL and JAX environment variables for communication and coordination between nodes. It resolves the JAX coordinator IP address and executes a command specified by the `COMMAND` environment variable. It includes configurations for GPUDirect TCPX and FasTrak.

### `notest.txt`
Contains notes for configuring MaxText runs, specifically:
- Adjust `NNODES`, `num_slices`, and `dcn_data_parallelism` to match the number of A3 machines.
- Update `run_name` for each MaxText execution.

### `working_config_ported_supercomputer.json`
A JSON configuration file for a Vertex AI custom job, likely for training Llama2-7B. It defines:
- `displayName`: "shivaji_llama2_7b_vertex_gcs_v01_05"
- Machine specifications (a3-megagpu-8g with NVIDIA_H100_MEGA_80GB GPUs).
- Docker image: `aurius/maxtext-fastrak:06-11-2024`.
- Numerous environment variables for NCCL, XLA, JAX, and other distributed training parameters.
- The command to run MaxText training: `python MaxText/train.py MaxText/configs/base.yml ...`
- Specifies configurations for 2 nodes (1 master, 1 worker).

### `working_config_ported_supercomputer.sh`
A shell script that uses `curl` to submit a custom job to Vertex AI. It sends the configuration defined in `working_config_ported_supercomputer.json` to the Vertex AI API endpoint for the `us-central1` region and project `google.com:vertex-training-dlexamples`.

## `LLama2-70B/` Directory

### `LLama2-70B/LLama2-70b-4vm.json`
A JSON configuration file for a Vertex AI custom job, specifically for Llama2-70B model training using 4 VMs. Key details:
- `displayName`: "shivaji_llama2_70b_vertex_8VM_syn_v01" (Note: displayName seems to indicate 8VM, which might be a mismatch with filename or an intended configuration).
- Machine specifications: `a3-megagpu-8g` with 8 `NVIDIA_H100_MEGA_80GB` accelerators per replica.
- Defines two worker pools:
    - Worker pool 0: 1 replica (likely the master).
    - Worker pool 1: 7 replicas. (Filename suggests 4 VMs in total, this configuration implies 8 VMs: 1 master + 7 workers. This might be an error or a specific setup).
- Docker image: `aurius/maxtext-fastrak:06-11-2024`.
- Extensive environment variables for NCCL, XLA, JAX, and specific command for training Llama2-70B with synthetic data: `python MaxText/train.py MaxText/configs/base.yml ... model_name=llama2-70b ... num_slices=8`. The `NNODES` is set to 8 for the first worker pool and 4 for the second, which is inconsistent. The command specifies `num_slices=8`.

### `LLama2-70B/LLama2-70b-8vm.json`
A JSON configuration file for a Vertex AI custom job, for Llama2-70B model training using 8 VMs. Key details:
- `displayName`: "shivaji_llama2_70b_vertex_8VM_syn_v01".
- Machine specifications: `a3-megagpu-8g` with 8 `NVIDIA_H100_MEGA_80GB` accelerators per replica.
- Defines two worker pools:
    - Worker pool 0: 1 replica (master).
    - Worker pool 1: 7 replicas (workers).
- Docker image: `aurius/maxtext-fastrak:06-11-2024`.
- Extensive environment variables for NCCL, XLA, JAX.
- The command for training Llama2-70B with synthetic data: `python MaxText/train.py MaxText/configs/base.yml ... model_name=llama2-70b ... num_slices=8`. `NNODES` is set to 8 for the first worker pool and 4 for the second, which is inconsistent. The command specifies `num_slices=8` and `per_device_batch_size=6`.

### `LLama2-70B/gpu_multi_process_run.sh`
This script is designed to run on multiple GPUs in a distributed environment, likely within the Docker container specified in the JSON job configurations. It sets up various NCCL and JAX environment variables (e.g., `NNODES`, `NODE_RANK`, `JAX_COORDINATOR_ADDRESS`, `GPUS_PER_NODE`) essential for multi-node/multi-GPU communication and coordination.
It includes:
- Functions to set NCCL configurations specific to `GPUDirect tcpx` or `fastrak`.
- Logic to resolve the coordinator IP address using `nslookup`.
- Execution of a main `COMMAND` passed as an environment variable.
- Error handling and process monitoring.

### `LLama2-70B/llama2-70b-4vm.sh`
A shell script that uses `curl` to submit a custom job to Vertex AI. It sends the configuration defined in `LLama2-70b-4vm.json` to the Vertex AI API endpoint for the `us-east4` region and project `google.com:vertex-training-dlexamples`. This script initiates a training job for Llama2-70B, presumably on 4 VMs.

### `LLama2-70B/llama2-70b-8vm.sh`
A shell script that uses `curl` to submit a custom job to Vertex AI. It sends the configuration defined in `LLama2-70b-8vm.json` to the Vertex AI API endpoint for the `us-east4` region and project `google.com:vertex-training-dlexamples`. This script initiates a training job for Llama2-70B, presumably on 8 VMs.

## `LLama2-7B/` Directory

### `LLama2-7B/gpu_multi_process_run.sh`
This script is identical in content and purpose to `LLama2-70B/gpu_multi_process_run.sh`. It's a general script for setting up and running distributed JAX processes on multiple GPUs, configuring NCCL and other environment variables for optimal performance on A3 Mega instances.

### `LLama2-7B/working_config_ported_supercomputer.json`
A JSON configuration file for a Vertex AI custom job, tailored for Llama2-7B model training.
- `displayName`: "shivaji_llama2_7b_vertex_gcs_v01_05".
- Machine specifications: `a3-megagpu-8g` with 8 `NVIDIA_H100_MEGA_80GB` accelerators per replica.
- Defines two worker pools, each with 1 replica, suggesting a 2-node setup (`NNODES=2`).
- Docker image: `aurius/maxtext-fastrak:06-11-2024`.
- Extensive environment variables for NCCL, XLA, JAX.
- The command for training Llama2-7B: `python MaxText/train.py MaxText/configs/base.yml ... model_name=llama2-7b ... dcn_data_parallelism=2 ici_fsdp_parallelism=8`.

### `LLama2-7B/working_config_ported_supercomputer.sh`
A shell script that uses `curl` to submit a custom job to Vertex AI. It sends the configuration defined in `LLama2-7B/working_config_ported_supercomputer.json` to the Vertex AI API endpoint for the `us-central1` region and project `google.com:vertex-training-dlexamples`. This script is used to launch Llama2-7B training jobs.

## `cluster_test/` Directory

### `cluster_test/Dockerfile`
Defines instructions to build a Docker image based on `ghcr.io/nvidia/jax:base`.
It:
- Creates a `/deps` directory and sets it as the working directory.
- Copies project files into the container.
- Installs Python packages, including JAX with CUDA 12 support (from `constraints_gpu.txt`) and `datasets`.

### `cluster_test/README.md`
## Vertex Cluster Test

This is a simple code base to test a cluster in JAX. You can read about [JAX Multihost training](https://jax.readthedocs.io/en/latest/multi_process.html) here.

Vertex Distributed training provides cluster information in terms of [CLUSTER SPEC](https://cloud.google.com/vertex-ai/docs/training/distributed-training) json Object.

In this example we will parse the Vertex Cluster Spec, get the details for

- JAX_COORDINATOR_ADDRESS
- JAX_COORDINATOR_PORT
- NNODES
- NODE_RANK

This forms the cluster and runs some basic sharding tests on Vertex.



## Build the docker file

In this file I build and push a local artifact Google Cloud repository gcr.io/google.com/vertex-training-dlexamples/snap-perf-repo/. Please edit this file to the Artifact Repository for your project. 

```
bash buid_push.sh
```

### `cluster_test/build_push.sh`
A shell script to build a Docker image using the `Dockerfile` in the current directory and push it to Google Container Registry.
- Image name: `gcr.io/google.com/vertex-training-dlexamples/snap-perf-repo/vertex_dist_example_v6:sharded` (and also `:latest` implicitly by some interpretations, though only `:sharded` is explicitly tagged here).

### `cluster_test/cluster.py`
A Python script designed to run in a Vertex AI distributed training environment. It:
- Reads environment variables `TF_CONFIG` and `CLUSTER_SPEC` to understand the cluster topology.
- Parses `CLUSTER_SPEC` to extract coordinator address, port, number of nodes (`NNODES`), and the current node's rank (`NODE_RANK`).
- Sets JAX environment variables (`JAX_COORDINATOR_ADDRESS`, `JAX_COORDINATOR_PORT`, `NNODES`, `NODE_RANK`).
- Resolves the coordinator's IP address.
- Initializes JAX distributed system using `jax.distributed.initialize()`.
- Prints the global JAX devices available.
This script helps in setting up JAX for multi-host/multi-process execution on Vertex AI.

### `cluster_test/constraints_gpu.txt`
A text file listing Python packages and their specific versions. This is used as a constraints file for `pip install` to ensure a reproducible Python environment, particularly for GPU-enabled JAX and TensorFlow development. It includes packages like `jax`, `jaxlib`, `tensorflow`, `flax`, `orbax-checkpoint`, `nvidia-*` libraries, and various Google Cloud client libraries.

### `cluster_test/job_submission-us-central1.sh`
A shell script that uses `curl` to submit a custom job to Vertex AI. It sends the configuration defined in `payload.json` (presumably located in the same directory) to the Vertex AI API endpoint for the `us-central1` region and project `google.com:vertex-training-dlexamples`.

### `cluster_test/payload.json`
A JSON configuration file for a Vertex AI custom job, likely for testing distributed setups.
- `displayName`: "vertex_sharded_distributed".
- Machine specifications: `a3-megagpu-8g` with 8 `NVIDIA_H100_MEGA_80GB` accelerators per replica.
- Defines two worker pools:
    - Worker pool 0 (master): 1 replica.
    - Worker pool 1 (workers): 3 replicas.
    Totaling 4 nodes (`NNODES=4` is set as an environment variable).
- Docker image: `gcr.io/google.com/vertex-training-dlexamples/snap-perf-repo/vertex_dist_example:latest`.
- Sets environment variables `NCCL_LIB_DIR` and `NNODES`.

### `cluster_test/sharding.py`
A Python script that demonstrates JAX sharding capabilities in a distributed environment. It:
- Sets up JAX distributed environment similarly to `cluster.py` by parsing `CLUSTER_SPEC`.
- Creates NumPy arrays (`A`, `B`).
- Defines a JAX mesh and `NamedSharding` with `PartitionSpec("myaxis")` to shard data across devices along one axis.
- Performs sharding using `jax.device_put`, `jax.make_array_from_process_local_data`, and `jax.make_array_from_callback`.
- Prints the shape of the addressable data on the local device for sharded arrays.
- Demonstrates unsharding by putting a sharded array back to a fully replicated layout using `PartitionSpec(None)`.
This script is used to test and verify data sharding across multiple devices/hosts in a JAX cluster.
