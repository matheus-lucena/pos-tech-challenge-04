import argparse
import os
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Tenta carregar dotenv apenas se estiver local
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def model_fn(model_dir):
    """Função obrigatória para o SageMaker carregar o modelo na inferência"""
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return model

if __name__ == '__main__':
    print(">>> Iniciando script de treinamento...")

    # 1. Parse de argumentos
    parser = argparse.ArgumentParser()

    # Hiperparâmetros
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--max_depth', type=int, default=5)

    # --- CORREÇÃO 1: Adicionar argumentos de diretórios do SageMaker ---
    # O SageMaker injeta essas variáveis de ambiente automaticamente
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))

    args = parser.parse_args()

    # 2. Carregar Dados
    # --- CORREÇÃO 2: Usar o caminho completo do arquivo ---
    # args.train aponta para /opt/ml/input/data/train
    print(f"Buscando dados em: {args.train}")
    
    csv_path = os.path.join(args.train, "maternal_health_risk.csv")
    
    if not os.path.exists(csv_path):
        # Fallback de segurança para listar o que tem na pasta (ajuda no debug)
        print(f"ERRO: Arquivo não encontrado em {csv_path}")
        print(f"Conteúdo da pasta {args.train}: {os.listdir(args.train)}")
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Dataset carregado. Shape: {df.shape}")

    # 3. Pré-processamento
    df['RiskLevel'] = df['RiskLevel'].apply(lambda x: 1 if str(x).lower().strip() == 'high risk' else 0)
    
    X = df.drop('RiskLevel', axis=1)
    y = df['RiskLevel']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Treinamento
    print(f"Treinando com: n_estimators={args.n_estimators}, max_depth={args.max_depth}")
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        objective='binary:logistic',
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)

    # 5. Validação
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f">>> Acurácia na validação: {acc:.4f}")
    print(classification_report(y_val, preds))

    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print(f">>> Modelo salvo com sucesso em: {model_path}")