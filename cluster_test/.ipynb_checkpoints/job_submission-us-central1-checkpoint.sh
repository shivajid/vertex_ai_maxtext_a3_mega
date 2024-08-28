curl -X POST \
     -H "Authorization: Bearer $(gcloud auth print-access-token)" \
     -H "Content-Type: application/json; charset=utf-8" \
     -d @payload.json \
     "https://us-central1-aiplatform.googleapis.com/v1/projects/google.com:vertex-training-dlexamples/locations/us-central1/customJobs"
