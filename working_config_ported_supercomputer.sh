curl -X POST \
     -H "Authorization: Bearer $(gcloud auth print-access-token)" \
     -H "Content-Type: application/json; charset=utf-8" \
     -d @working_config_ported_supercomputer.json \
     "https://us-east4-aiplatform.googleapis.com/v1/projects/google.com:vertex-training-dlexamples/locations/us-east4/customJobs"
