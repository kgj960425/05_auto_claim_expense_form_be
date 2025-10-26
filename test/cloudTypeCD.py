# from pymupdf4llm import PyMuPDF4LLM

# 클라우드 타입 배포 api 
# $ curl -d '{"project":"myproject", "app": "myapp", "stage":"main"}' \
#        -H "Content-Type: application/json" \
#        -H "Authorization: Bearer <apikey>" \
#        -X POST https://api.cloudtype.io/webhooks/deploy
# $ curl -X GET https://api.cloudtype.io/webhooks/deploy?token=<apikey>&project=<projectname>&stage=<stagename>&app=<appname>