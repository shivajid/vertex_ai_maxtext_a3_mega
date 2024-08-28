import json
import os
import socket
import jax


tf_config_str = os.environ.get('TF_CONFIG')
nnodes = os.environ.get('NNODES')
print("NNODES_FROM_ENV", nnodes)

'''
if (tf_config_str != None) :
    print(tf_config_str)
    tf_config_dict  = json.loads(tf_config_str)
    print(json.dumps(tf_config_dict, indent=2))
else:
    print( "tf_config_str is empty")
''' 

def get_coordinator_ip_address():
  """Get coordinator IP Address with retries"""
  coordinator_address = ""
  coordinator_ip_address = ""
  if os.environ.get("JAX_COORDINATOR_ADDRESS") is not None:
    coordinator_address = os.environ.get("JAX_COORDINATOR_ADDRESS")
    coordinator_found = False
    lookup_attempt = 1
    max_coordinator_lookups = 50
    while not coordinator_found and lookup_attempt <= max_coordinator_lookups:
      try:
        coordinator_ip_address = socket.gethostbyname(coordinator_address)
        coordinator_found = True
      except socket.gaierror:
        max_logging.log(
            f"Failed to recognize coordinator address {coordinator_address} on attempt {lookup_attempt}, retrying..."
        )
        lookup_attempt += 1
        time.sleep(5)
  print(f"Coordinator IP address: {coordinator_ip_address}")
  return coordinator_ip_address



cluster_spec = os.environ.get('CLUSTER_SPEC')
if (cluster_spec != None) :
    print(cluster_spec)
    cluster_spec_json  = json.loads(cluster_spec)
    print(json.dumps(cluster_spec_json, indent=2))
    print("task/type:", cluster_spec_json['task']['type'])
    print("task/index:", cluster_spec_json['task']['index'])
    coordinator= cluster_spec_json['cluster']['workerpool0'][0].split(":")[0]
    print("JAX_COORDINATOR_ADDRESS",coordinator)
    print("JAX_COORDINATOR_PORT", cluster_spec_json['cluster']['workerpool0'][0].split(":")[1])
    print("NNODES",4)
    node_type= cluster_spec_json['task']['type']
    if( node_type == "workerpool0"):
        NODE_RANK = cluster_spec_json['task']['index']
    else:
        NODE_RANK = int(cluster_spec_json['task']['index']) + 1
    print("NODE_RANK",NODE_RANK) 
    
    
    
    os.environ["JAX_COORDINATOR_ADDRESS"] = coordinator #cluster_spec_json['cluster']['workerpool0'][0].split(":")[0]
    os.environ["JAX_COORDINATOR_PORT"] = cluster_spec_json['cluster']['workerpool0'][0].split(":")[1]
    os.environ["NNODES"] = "4"
    if( node_type == "workerpool0"):
        NODE_RANK = cluster_spec_json['task']['index']
    else:
        NODE_RANK = int(cluster_spec_json['task']['index']) + 1
    os.environ["NODE_RANK"] = str(NODE_RANK)
    
    
    ip_addr = get_coordinator_ip_address()
    
    print("IP Address", ip_addr)
    os.environ["JAX_COORDINATOR_IP"] = ip_addr
    if os.environ.get("JAX_COORDINATOR_IP") is not None:
     coordinator_ip = str(os.getenv("JAX_COORDINATOR_IP"))
     coordinator_port = str(os.getenv("JAX_COORDINATOR_PORT"))
     jax.distributed.initialize(
        coordinator_address=f"{coordinator_ip}:{coordinator_port}",
        num_processes=int(os.getenv("NNODES")),
        process_id=int(os.getenv("NODE_RANK")),
      )
     print(f"JAX global devices: {jax.devices()}")
    else:
      print("JAX_COORDINATOR_IP NOT FOUND!!!")

    


else:
    print( "cluster_spec is empty")
