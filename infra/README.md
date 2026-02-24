# Infra — Terraform AWS

Provisiona os recursos AWS necessários para o sistema de saúde materna.

## Recursos Criados

### S3 Buckets

- **Bucket de áudio** — armazena arquivos para transcrição
  - Versionamento, criptografia AES256, acesso público bloqueado
  - Política de acesso para o AWS Transcribe

- **Bucket SageMaker** (opcional) — armazena dados de treinamento e artefatos do modelo

### IAM Roles

- **SageMaker Role** — permissões para criar e gerenciar jobs, modelos e endpoints
- **Transcribe Role** — acesso ao bucket S3 de áudio

### IAM User

Usuário para execução local com acesso a:
S3, SageMaker, Transcribe, Comprehend Medical, Textract, Bedrock, CloudWatch Logs, ECR

### CloudWatch Log Groups

- Logs do SageMaker
- Logs do Transcribe

---

## Pré-requisitos

- Terraform >= 1.0
- AWS CLI configurado com credenciais administrativas

---

## Uso

### 1. Configurar variáveis

```bash
cp terraform.tfvars.example terraform.tfvars
```

Edite `terraform.tfvars`:

```hcl
aws_region              = "us-east-1"
project_name            = "maternal-health-system"
audio_bucket_name       = "seu-bucket-audio-unico"      # único globalmente
create_sagemaker_bucket = true
sagemaker_bucket_name   = "seu-bucket-sagemaker-unico"  # único globalmente
transcribe_role_name    = "TranscribeDataAccess"
log_retention_days      = 7
```

**Atenção**: nomes de buckets S3 são únicos globalmente na AWS.

### 2. Inicializar e aplicar

```bash
terraform init
terraform plan
terraform apply
```

### 3. Obter credenciais do usuário IAM

```bash
terraform output access_key_id
terraform output -raw secret_access_key
```

Salve essas credenciais — elas não serão exibidas novamente.

### 4. (Opcional) Configurar AWS CLI

```bash
terraform output -raw aws_cli_configure_command
```

---

## Variáveis

| Variável | Descrição | Padrão | Obrigatório |
|---|---|---|---|
| `aws_region` | Região AWS | `us-east-1` | Não |
| `project_name` | Prefixo dos recursos | `maternal-health-system` | Não |
| `audio_bucket_name` | Nome do bucket de áudio | — | Sim |
| `create_sagemaker_bucket` | Criar bucket para SageMaker | `false` | Não |
| `sagemaker_bucket_name` | Nome do bucket SageMaker | — | Se `create_sagemaker_bucket = true` |
| `sagemaker_additional_buckets` | ARNs de buckets adicionais | `[]` | Não |
| `transcribe_role_name` | Nome da role do Transcribe | `TranscribeDataAccess` | Não |
| `log_retention_days` | Retenção de logs (dias) | `7` | Não |

---

## Outputs

```bash
terraform output audio_bucket_name
terraform output sagemaker_bucket_name
terraform output sagemaker_role_arn
terraform output transcribe_role_arn
terraform output iam_user_name
terraform output access_key_id
terraform output sagemaker_log_group
terraform output transcribe_log_group
```

---

## Estrutura

```
infra/
├── main.tf                   # Recursos principais
├── variables.tf              # Definição de variáveis
├── outputs.tf                # Outputs
├── terraform.tfvars          # Valores (não commitar)
└── terraform.tfvars.example  # Exemplo de configuração
```

---

## Segurança

- Criptografia AES256 em todos os buckets
- Versionamento habilitado
- Acesso público bloqueado
- Políticas IAM com princípio de menor privilégio
- Retenção de logs configurável

**Não commitar:** `terraform.tfvars`, `terraform.tfstate`, `.terraform/`

---

## Destruir infraestrutura

```bash
terraform destroy
```

Isso remove todos os recursos, incluindo buckets S3 e seu conteúdo.

---

## Troubleshooting

**Bucket já existe** — nomes de buckets são únicos globalmente; escolha um nome diferente.

**Permissões insuficientes** — as credenciais precisam de acesso administrativo para criar IAM roles e buckets.

**Rate limiting** — aguarde alguns minutos e tente novamente.

---

## Referências

- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [AWS S3 Security Best Practices](https://docs.aws.amazon.com/AmazonS3/latest/userguide/security-best-practices.html)
- [AWS IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
