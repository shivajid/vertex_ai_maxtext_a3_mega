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