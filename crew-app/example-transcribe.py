import boto3
import time

REGION = "us-east-1"

BUCKET = "fiap-pos-teste-20-19"
KEY = "vitima-01.mp3"

transcribe = boto3.client('transcribe', region_name=REGION)

s3_config = boto3.session.Session().client(
    's3', 
    region_name=REGION,
    config=boto3.session.Config(s3={'addressing_style': 'virtual'})
)

job_name = f"job_{int(time.time())}"
s3_path = f"s3://{BUCKET}/{KEY}"

job_params = {
    'TranscriptionJobName': job_name,
    'Media': {'MediaFileUri': s3_path},
    'MediaFormat': 'mp3',
    'LanguageCode': 'pt-BR',
    'Settings': {
        'ShowAlternatives': True,
        'MaxAlternatives': 2,
    },
    'JobExecutionSettings': {
        'DataAccessRoleArn': 'arn:aws:iam::517171444774:role/TranscribeDataAccessRole',
    }
}

print(job_params)

response = transcribe.start_transcription_job(**job_params)

print(response)