# ============================================================================
# S3 BUCKETS
# ============================================================================

output "audio_bucket_name" {
  description = "Nome do bucket S3 para áudios"
  value       = aws_s3_bucket.audio_bucket.id
}

output "audio_bucket_arn" {
  description = "ARN do bucket S3 para áudios"
  value       = aws_s3_bucket.audio_bucket.arn
}

output "sagemaker_bucket_name" {
  description = "Nome do bucket S3 para dados do SageMaker (se criado)"
  value       = var.create_sagemaker_bucket ? aws_s3_bucket.sagemaker_data_bucket[0].id : null
}

output "sagemaker_bucket_arn" {
  description = "ARN do bucket S3 para dados do SageMaker (se criado)"
  value       = var.create_sagemaker_bucket ? aws_s3_bucket.sagemaker_data_bucket[0].arn : null
}

# ============================================================================
# IAM ROLES
# ============================================================================

output "sagemaker_role_arn" {
  description = "ARN da IAM Role para SageMaker"
  value       = aws_iam_role.sagemaker_role.arn
}

output "sagemaker_role_name" {
  description = "Nome da IAM Role para SageMaker"
  value       = aws_iam_role.sagemaker_role.name
}

output "transcribe_role_arn" {
  description = "ARN da IAM Role para Transcribe"
  value       = aws_iam_role.transcribe_data_access_role.arn
}

output "transcribe_role_name" {
  description = "Nome da IAM Role para Transcribe"
  value       = aws_iam_role.transcribe_data_access_role.name
}

# ============================================================================
# IAM USER
# ============================================================================

output "iam_user_name" {
  description = "Nome do usuário IAM criado para execução local"
  value       = aws_iam_user.local_user.name
}

output "access_key_id" {
  description = "Access Key ID do usuário (sensível)"
  value       = aws_iam_access_key.local_user_key.id
  sensitive   = false
}

output "secret_access_key" {
  description = "Secret Access Key do usuário (sensível - salve com segurança!)"
  value       = aws_iam_access_key.local_user_key.secret
  sensitive   = true
}

# ============================================================================
# CLOUDWATCH
# ============================================================================

output "sagemaker_log_group" {
  description = "Nome do CloudWatch Log Group para SageMaker"
  value       = aws_cloudwatch_log_group.sagemaker_logs.name
}

output "transcribe_log_group" {
  description = "Nome do CloudWatch Log Group para Transcribe"
  value       = aws_cloudwatch_log_group.transcribe_logs.name
}

# ============================================================================
# COMANDOS ÚTEIS
# ============================================================================

output "aws_cli_configure_command" {
  description = "Comando para configurar AWS CLI com as credenciais do usuário"
  value       = "aws configure set aws_access_key_id ${aws_iam_access_key.local_user_key.id} && aws configure set aws_secret_access_key ${aws_iam_access_key.local_user_key.secret} && aws configure set region ${var.aws_region}"
  sensitive   = true
}
