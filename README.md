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

### Setup
On A3 Mega it achieves as high as 55% MFU on 2 Node A3 Mega.

```
git clone https://github.com/shivajid/a3_mega_benchmarking.git
cd a3_mega_benchmarking.git

```
To submit an A3 Mega job on Vertex you need:
- Job json file, describe the job artifact
- A Scrip to submit the job

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
### Submit a job
To submit the job run the following:
```
bash working_config_ported_supercomputer.sh
```

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
#### Dataset
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
  
## Contact

Please reach out to shivajid@google.com for any questions.








