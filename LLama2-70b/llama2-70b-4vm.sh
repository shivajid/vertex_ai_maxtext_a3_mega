curl -X POST \
     -H "Authorization: Bearer $(gcloud auth print-access-token)" \
     -H "Content-Type: application/json; charset=utf-8" \
     -d @LLama2-70b-4vm.json \
     "https://us-east4-aiplatform.googleapis.com/v1/projects/google.com:vertex-training-dlexamples/locations/us-east4/customJobs"
