# 🏗️ Infraestrutura AWS - Terraform

Infraestrutura como Código (IaC) para provisionar todos os recursos AWS necessários para o sistema de análise de saúde materna.

## 📋 Descrição

Este módulo Terraform provisiona automaticamente:

- **Buckets S3**: Armazenamento de áudios e dados de treinamento do SageMaker
- **IAM Roles**: Permissões para SageMaker, Transcribe e outros serviços
- **IAM User**: Usuário para execução local com permissões adequadas
- **CloudWatch Log Groups**: Logs centralizados para monitoramento
- **Políticas de Segurança**: Criptografia, versionamento e bloqueio de acesso público

## 🏗️ Recursos Provisionados

### S3 Buckets

- **Bucket de Áudio**: Armazena arquivos de áudio para transcrição
  - Versionamento habilitado
  - Criptografia AES256
  - Acesso público bloqueado
  - Política para permitir acesso do Transcribe

- **Bucket SageMaker** (opcional): Armazena dados de treinamento
  - Configurações de segurança similares ao bucket de áudio

### IAM Roles

- **SageMaker Role**: Permissões para criar e gerenciar modelos, endpoints e jobs de treinamento
- **Transcribe Role**: Permissões para acessar o bucket S3 de áudio

### IAM User

- **Local User**: Usuário para execução local com permissões para:
  - Acessar buckets S3
  - Usar SageMaker (criar jobs, endpoints, invocar modelos)
  - Usar Transcribe
  - Usar Comprehend Medical
  - Usar Textract
  - Usar Bedrock
  - Acessar CloudWatch Logs
  - Acessar ECR

### CloudWatch Log Groups

- Logs do SageMaker
- Logs do Transcribe

## 📦 Pré-requisitos

- **Terraform** >= 1.0
- **AWS CLI** configurado com credenciais administrativas
- **Conta AWS** com permissões para criar recursos

## 🚀 Instalação e Uso

### 1. Configurar Variáveis

Copie o arquivo de exemplo e ajuste os valores:

```bash
cp terraform.tfvars.example terraform.tfvars
```

Edite `terraform.tfvars` com seus valores:

```hcl
aws_region = "us-east-1"
project_name = "maternal-health-system"
audio_bucket_name = "seu-bucket-audio-unico"  # DEVE SER ÚNICO GLOBALMENTE
create_sagemaker_bucket = true
sagemaker_bucket_name = "seu-bucket-sagemaker-unico"  # DEVE SER ÚNICO GLOBALMENTE
transcribe_role_name = "TranscribeDataAccess"
log_retention_days = 7
```

**⚠️ IMPORTANTE**: Os nomes dos buckets S3 devem ser **únicos globalmente** na AWS.

### 2. Inicializar Terraform

```bash
terraform init
```

### 3. Revisar o Plano

```bash
terraform plan
```

Este comando mostra todos os recursos que serão criados.

### 4. Aplicar a Infraestrutura

```bash
terraform apply
```

Confirme digitando `yes` quando solicitado.

### 5. Obter Credenciais do Usuário Local

Após o `terraform apply`, você pode obter as credenciais:

```bash
# Access Key ID
terraform output access_key_id

# Secret Access Key (sensível)
terraform output -raw secret_access_key
```

**⚠️ IMPORTANTE**: Salve essas credenciais com segurança! Elas não serão exibidas novamente.

### 6. Configurar AWS CLI (Opcional)

```bash
aws configure set aws_access_key_id $(terraform output -raw access_key_id)
aws configure set aws_secret_access_key $(terraform output -raw secret_access_key)
aws configure set region $(terraform output -raw aws_region)
```

Ou use o comando fornecido:

```bash
terraform output -raw aws_cli_configure_command
```

## 📊 Outputs Disponíveis

Após aplicar o Terraform, você pode consultar os outputs:

```bash
# Buckets
terraform output audio_bucket_name
terraform output sagemaker_bucket_name

# IAM Roles
terraform output sagemaker_role_arn
terraform output transcribe_role_arn

# IAM User
terraform output iam_user_name
terraform output access_key_id

# CloudWatch
terraform output sagemaker_log_group
terraform output transcribe_log_group
```

## 🔧 Variáveis Configuráveis

| Variável | Descrição | Padrão | Obrigatório |
|----------|-----------|--------|-------------|
| `aws_region` | Região AWS | `us-east-1` | Não |
| `project_name` | Nome do projeto (prefixo) | `maternal-health-system` | Não |
| `audio_bucket_name` | Nome do bucket S3 para áudios | - | **Sim** |
| `create_sagemaker_bucket` | Criar bucket para SageMaker | `false` | Não |
| `sagemaker_bucket_name` | Nome do bucket SageMaker | - | Se `create_sagemaker_bucket = true` |
| `sagemaker_additional_buckets` | Lista de ARNs de buckets adicionais | `[]` | Não |
| `transcribe_role_name` | Nome da role para Transcribe | `TranscribeDataAccess` | Não |
| `log_retention_days` | Dias de retenção de logs | `7` | Não |

## 🗑️ Destruir Infraestrutura

Para remover todos os recursos criados:

```bash
terraform destroy
```

**⚠️ ATENÇÃO**: Isso irá deletar todos os recursos, incluindo buckets S3 e seus conteúdos!

## 📁 Estrutura do Projeto

```
infra/
├── main.tf              # Recursos principais
├── variables.tf         # Definição de variáveis
├── outputs.tf          # Outputs do Terraform
├── terraform.tfvars     # Valores das variáveis (não commitar!)
├── terraform.tfvars.example  # Exemplo de configuração
└── README.md           # Este arquivo
```

## 🔒 Segurança

### Boas Práticas Implementadas

- ✅ Criptografia AES256 em todos os buckets
- ✅ Versionamento habilitado nos buckets
- ✅ Acesso público bloqueado
- ✅ Políticas IAM com princípio de menor privilégio
- ✅ Logs centralizados no CloudWatch
- ✅ Retenção configurável de logs

### Arquivos Sensíveis

- `terraform.tfvars`: Contém valores sensíveis, não commitar!
- `terraform.tfstate`: Contém estado sensível, não commitar!
- `.terraform/`: Cache do Terraform, não commitar!

Certifique-se de que o `.gitignore` está configurado corretamente.

## 🐛 Troubleshooting

### Erro: Bucket já existe
- Os nomes de buckets S3 devem ser únicos globalmente
- Escolha um nome diferente ou delete o bucket existente

### Erro: Permissões insuficientes
- Verifique se suas credenciais AWS têm permissões administrativas
- Confirme que pode criar IAM roles, buckets S3, etc.

### Erro: Rate limiting
- Alguns recursos podem ter limites de criação
- Aguarde alguns minutos e tente novamente

## 📝 Notas Importantes

1. **Nomes de Buckets**: Devem ser únicos globalmente na AWS
2. **IAM Roles**: Podem levar alguns minutos para propagar
3. **Custos**: Monitore os custos na AWS, especialmente para SageMaker endpoints
4. **Estado do Terraform**: Mantenha o arquivo `terraform.tfstate` seguro e faça backup

## 🔗 Links Úteis

- [Documentação do Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [AWS S3 Best Practices](https://docs.aws.amazon.com/AmazonS3/latest/userguide/security-best-practices.html)
- [AWS IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)

## 📝 Sobre o Projeto

Este projeto faz parte do sistema de saúde materna desenvolvido para o trabalho de pós-graduação.
