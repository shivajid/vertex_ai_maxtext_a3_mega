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

On A3 Mega it achieves as high as 55% MFU on 2 Node A3 Mega.

```
git clone https://github.com/shivajid/a3_mega_benchmarking.git

```

```
cd a3_mega_benchmarking.git

```
To submit an A3 Mega job on Vertex you need:
- Job json file, describe the job artifact
- A Scrip to submit the job








