variable "aws_region" {
  description = "Região AWS para provisionamento dos recursos"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Nome do projeto (usado como prefixo para recursos)"
  type        = string
  default     = "maternal-health-system"
}

variable "audio_bucket_name" {
  description = "Nome do bucket S3 para áudios (deve ser único globalmente)"
  type        = string
}

variable "create_sagemaker_bucket" {
  description = "Se deve criar um bucket específico para dados do SageMaker (se false, usa bucket padrão)"
  type        = bool
  default     = false
}

variable "sagemaker_bucket_name" {
  description = "Nome do bucket S3 para dados de treinamento do SageMaker (usado apenas se create_sagemaker_bucket = true)"
  type        = string
  default     = ""
}

variable "sagemaker_additional_buckets" {
  description = "Lista adicional de ARNs de buckets S3 que o SageMaker pode acessar"
  type        = list(string)
  default     = []
}

variable "transcribe_role_name" {
  description = "Nome da IAM Role para Transcribe"
  type        = string
  default     = "TranscribeDataAccess"
}

variable "log_retention_days" {
  description = "Dias de retenção dos logs no CloudWatch"
  type        = number
  default     = 7
}

