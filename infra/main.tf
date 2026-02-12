terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Data source para obter informações da conta AWS
data "aws_caller_identity" "current" {}

# ============================================================================
# S3 BUCKETS
# ============================================================================

# Bucket para armazenar áudios do crew-app
resource "aws_s3_bucket" "audio_bucket" {
  bucket = var.audio_bucket_name
}

# Configuração de versionamento do bucket de áudio
resource "aws_s3_bucket_versioning" "audio_bucket_versioning" {
  bucket = aws_s3_bucket.audio_bucket.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Criptografia do bucket de áudio
resource "aws_s3_bucket_server_side_encryption_configuration" "audio_bucket_encryption" {
  bucket = aws_s3_bucket.audio_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Bloqueia acesso público do bucket de áudio
resource "aws_s3_bucket_public_access_block" "audio_bucket_block" {
  bucket = aws_s3_bucket.audio_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Bucket para dados de treinamento do SageMaker (opcional, pode usar bucket padrão)
resource "aws_s3_bucket" "sagemaker_data_bucket" {
  count  = var.create_sagemaker_bucket ? 1 : 0
  bucket = var.sagemaker_bucket_name
}

# Configuração de versionamento do bucket SageMaker
resource "aws_s3_bucket_versioning" "sagemaker_bucket_versioning" {
  count  = var.create_sagemaker_bucket ? 1 : 0
  bucket = aws_s3_bucket.sagemaker_data_bucket[0].id

  versioning_configuration {
    status = "Enabled"
  }
}

# Criptografia do bucket SageMaker
resource "aws_s3_bucket_server_side_encryption_configuration" "sagemaker_bucket_encryption" {
  count  = var.create_sagemaker_bucket ? 1 : 0
  bucket = aws_s3_bucket.sagemaker_data_bucket[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Bloqueia acesso público do bucket SageMaker
resource "aws_s3_bucket_public_access_block" "sagemaker_bucket_block" {
  count  = var.create_sagemaker_bucket ? 1 : 0
  bucket = aws_s3_bucket.sagemaker_data_bucket[0].id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ============================================================================
# IAM ROLES
# ============================================================================

# Role para SageMaker
resource "aws_iam_role" "sagemaker_role" {
  name = "${var.project_name}-sagemaker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })
}

# Policy para SageMaker acessar S3
resource "aws_iam_policy" "sagemaker_s3_policy" {
  name        = "${var.project_name}-sagemaker-s3-policy"
  description = "Permite SageMaker acessar buckets S3"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = concat(
          [
            aws_s3_bucket.audio_bucket.arn,
            "${aws_s3_bucket.audio_bucket.arn}/*"
          ],
          var.create_sagemaker_bucket ? [
            aws_s3_bucket.sagemaker_data_bucket[0].arn,
            "${aws_s3_bucket.sagemaker_data_bucket[0].arn}/*"
          ] : [],
          var.sagemaker_additional_buckets
        )
      }
    ]
  })
}

# Policy para SageMaker (operações básicas)
resource "aws_iam_policy" "sagemaker_policy" {
  name        = "${var.project_name}-sagemaker-policy"
  description = "Permite operações do SageMaker"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:StopTrainingJob",
          "sagemaker:CreateModel",
          "sagemaker:DescribeModel",
          "sagemaker:DeleteModel",
          "sagemaker:CreateEndpointConfig",
          "sagemaker:DescribeEndpointConfig",
          "sagemaker:DeleteEndpointConfig",
          "sagemaker:CreateEndpoint",
          "sagemaker:DescribeEndpoint",
          "sagemaker:DeleteEndpoint",
          "sagemaker:UpdateEndpoint",
          "sagemaker:InvokeEndpoint",
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams",
          "logs:GetLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      }
    ]
  })
}

# Anexar policies à role SageMaker
resource "aws_iam_role_policy_attachment" "sagemaker_s3_attachment" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = aws_iam_policy.sagemaker_s3_policy.arn
}

resource "aws_iam_role_policy_attachment" "sagemaker_attachment" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = aws_iam_policy.sagemaker_policy.arn
}

# Role para Transcribe (TranscribeDataAccessRole)
resource "aws_iam_role" "transcribe_data_access_role" {
  name = var.transcribe_role_name

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "transcribe.amazonaws.com"
        }
      }
    ]
  })
}

# Policy para Transcribe acessar S3 (anexada à role)
resource "aws_iam_policy" "transcribe_s3_policy" {
  name        = "${var.project_name}-transcribe-s3-policy"
  description = "Permite Transcribe acessar bucket S3"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.audio_bucket.arn,
          "${aws_s3_bucket.audio_bucket.arn}/*"
        ]
      }
    ]
  })
}

# Anexar policy à role Transcribe
resource "aws_iam_role_policy_attachment" "transcribe_s3_attachment" {
  role       = aws_iam_role.transcribe_data_access_role.name
  policy_arn = aws_iam_policy.transcribe_s3_policy.arn
}

# Bucket Policy para permitir que o Transcribe acesse o bucket usando a role
resource "aws_s3_bucket_policy" "audio_bucket_transcribe_policy" {
  bucket = aws_s3_bucket.audio_bucket.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowTranscribeService"
        Effect = "Allow"
        Principal = {
          Service = "transcribe.amazonaws.com"
        }
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = "${aws_s3_bucket.audio_bucket.arn}/*"
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = data.aws_caller_identity.current.account_id
          }
          ArnLike = {
            "aws:SourceArn" = "arn:aws:transcribe:${var.aws_region}:${data.aws_caller_identity.current.account_id}:transcription-job/*"
          }
        }
      },
      {
        Sid    = "AllowTranscribeServiceListBucket"
        Effect = "Allow"
        Principal = {
          Service = "transcribe.amazonaws.com"
        }
        Action = [
          "s3:ListBucket"
        ]
        Resource = aws_s3_bucket.audio_bucket.arn
        Condition = {
          StringEquals = {
            "aws:SourceAccount" = data.aws_caller_identity.current.account_id
          }
          ArnLike = {
            "aws:SourceArn" = "arn:aws:transcribe:${var.aws_region}:${data.aws_caller_identity.current.account_id}:transcription-job/*"
          }
        }
      }
    ]
  })
}

# ============================================================================
# IAM USER PARA EXECUÇÃO LOCAL
# ============================================================================

# IAM User para executar localmente
resource "aws_iam_user" "local_user" {
  name = "${var.project_name}-local-user"
}

# Access Key para o usuário
resource "aws_iam_access_key" "local_user_key" {
  user = aws_iam_user.local_user.name
}

# Policy para o usuário acessar S3
resource "aws_iam_policy" "local_user_s3_policy" {
  name        = "${var.project_name}-local-user-s3-policy"
  description = "Permite usuario local acessar buckets S3"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = concat(
          [
            aws_s3_bucket.audio_bucket.arn,
            "${aws_s3_bucket.audio_bucket.arn}/*"
          ],
          var.create_sagemaker_bucket ? [
            aws_s3_bucket.sagemaker_data_bucket[0].arn,
            "${aws_s3_bucket.sagemaker_data_bucket[0].arn}/*"
          ] : [],
          var.sagemaker_additional_buckets
        )
      }
    ]
  })
}

resource "aws_iam_user_policy_attachment" "local_user_s3_attachment" {
  user       = aws_iam_user.local_user.name
  policy_arn = aws_iam_policy.local_user_s3_policy.arn
}

# Policy para o usuário usar SageMaker
resource "aws_iam_policy" "local_user_sagemaker_policy" {
  name        = "${var.project_name}-local-user-sagemaker-policy"
  description = "Permite usuario local usar SageMaker"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sagemaker:CreateTrainingJob",
          "sagemaker:DescribeTrainingJob",
          "sagemaker:StopTrainingJob",
          "sagemaker:CreateModel",
          "sagemaker:DescribeModel",
          "sagemaker:DeleteModel",
          "sagemaker:CreateEndpointConfig",
          "sagemaker:DescribeEndpointConfig",
          "sagemaker:DeleteEndpointConfig",
          "sagemaker:CreateEndpoint",
          "sagemaker:DescribeEndpoint",
          "sagemaker:DeleteEndpoint",
          "sagemaker:UpdateEndpoint",
          "sagemaker:InvokeEndpoint",
          "sagemaker-runtime:InvokeEndpoint",
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "iam:PassRole"
        ]
        Resource = aws_iam_role.sagemaker_role.arn
        Condition = {
          StringEquals = {
            "iam:PassedToService" = "sagemaker.amazonaws.com"
          }
        }
      }
    ]
  })
}

resource "aws_iam_user_policy_attachment" "local_user_sagemaker_attachment" {
  user       = aws_iam_user.local_user.name
  policy_arn = aws_iam_policy.local_user_sagemaker_policy.arn
}

# Policy para o usuário usar Transcribe
resource "aws_iam_policy" "local_user_transcribe_policy" {
  name        = "${var.project_name}-local-user-transcribe-policy"
  description = "Permite usuario local usar Transcribe"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "transcribe:StartTranscriptionJob",
          "transcribe:GetTranscriptionJob",
          "transcribe:ListTranscriptionJobs",
          "transcribe:DeleteTranscriptionJob"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "iam:PassRole"
        ]
        Resource = aws_iam_role.transcribe_data_access_role.arn
        Condition = {
          StringEquals = {
            "iam:PassedToService" = "transcribe.amazonaws.com"
          }
        }
      }
    ]
  })
}

resource "aws_iam_user_policy_attachment" "local_user_transcribe_attachment" {
  user       = aws_iam_user.local_user.name
  policy_arn = aws_iam_policy.local_user_transcribe_policy.arn
}

# Policy para o usuário acessar CloudWatch Logs
resource "aws_iam_policy" "local_user_logs_policy" {
  name        = "${var.project_name}-local-user-logs-policy"
  description = "Permite usuario local acessar CloudWatch Logs"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams",
          "logs:GetLogEvents"
        ]
        Resource = [
          "arn:aws:logs:*:*:log-group:/aws/sagemaker/*",
          "arn:aws:logs:*:*:log-group:/aws/transcribe/*"
        ]
      }
    ]
  })
}

resource "aws_iam_user_policy_attachment" "local_user_logs_attachment" {
  user       = aws_iam_user.local_user.name
  policy_arn = aws_iam_policy.local_user_logs_policy.arn
}

# Policy para o usuário usar Comprehend Medical
resource "aws_iam_policy" "local_user_comprehend_medical_policy" {
  name        = "${var.project_name}-local-user-comprehend-medical-policy"
  description = "Permite usuario local usar Comprehend Medical"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "comprehendmedical:DetectEntities",
          "comprehendmedical:DetectPHI",
          "comprehendmedical:InferICD10CM",
          "comprehendmedical:InferRxNorm",
          "comprehendmedical:InferSNOMEDCT",
          "comprehendmedical:DetectEntitiesV2"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_user_policy_attachment" "local_user_comprehend_medical_attachment" {
  user       = aws_iam_user.local_user.name
  policy_arn = aws_iam_policy.local_user_comprehend_medical_policy.arn
}

# Policy para o usuário usar Textract
resource "aws_iam_policy" "local_user_textract_policy" {
  name        = "${var.project_name}-local-user-textract-policy"
  description = "Permite usuario local usar Textract"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "textract:DetectDocumentText"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_user_policy_attachment" "local_user_textract_attachment" {
  user       = aws_iam_user.local_user.name
  policy_arn = aws_iam_policy.local_user_textract_policy.arn
}

resource "aws_iam_policy" "local_user_ecr_policy" {
  name        = "${var.project_name}-local-user-ecr-policy"
  description = "Permite operacoes de ECR para usuario local"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow",
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ],
        "Resource" : "*"
      }
    ]
  })
}

resource "aws_iam_user_policy_attachment" "local_user_ecr_attachment" {
  user       = aws_iam_user.local_user.name
  policy_arn = aws_iam_policy.local_user_ecr_policy.arn
}

# ============================================================================
# CLOUDWATCH LOG GROUPS
# ============================================================================

# CloudWatch Log Group para SageMaker
resource "aws_cloudwatch_log_group" "sagemaker_logs" {
  name              = "/aws/sagemaker/${var.project_name}"
  retention_in_days = var.log_retention_days
}

# CloudWatch Log Group para Transcribe
resource "aws_cloudwatch_log_group" "transcribe_logs" {
  name              = "/aws/transcribe/${var.project_name}"
  retention_in_days = var.log_retention_days
}

