import os
import pandas as pd
import numpy as np
import wandb   # pip install wandb    
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

import parameters

def run_train():

    api_key = os.environ["WANDB_API_KEY"]
    wandb.login(key=api_key)

    # 1. WandB 시작 (실험 이름에 모델명/파라미터 포함)
    cfg = parameters.MODEL_CONFIG
    m_type = cfg['model_type']
    p = cfg['params']

    # 실험 이름 생성
    exp_name = (
        f"{m_type}_"
        f"n{p['n_estimators']}_"
        f"lr{p['learning_rate']}_"
        f"d{p['max_depth']}"
    )

    wandb.init(
        project=parameters.WANDB_PROJECT,
        name=exp_name, 
        config=cfg,
    )

    # 2. 데이터 로드 (전처리된 파일)
    df = pd.read_csv(os.path.join(parameters.SAVE_DIR, "movies_cleaned.csv"))
    
    # 3. 학습/테스트 데이터 분리
    # (주의: 장르 인코딩 등 수치화된 컬럼만 X에 넣어야 함)
    features = ['budget', 'revenue', 'runtime', 'release_year', 'vote_count', 'main_genre_encoded']
    X = df[features]
    y = df['vote_average']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. 모델 선택
    cfg = parameters.MODEL_CONFIG
    if cfg["model_type"] == "xgboost":
        model = XGBRegressor(**cfg["params"])
    elif cfg["model_type"] == "lgbm":
        model = LGBMRegressor(**cfg["params"])
    else:
        model = RandomForestRegressor(**cfg["params"])

    # 5. 학습 및 예측
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # 6. 평가 (RMSE)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    # 7. WandB에 결과 기록
    wandb.log({"test_rmse": rmse})
    print(f"[{cfg['model_type']}] 실험 완료! RMSE: {rmse:.4f}")

    # 8. 모델 저장
    model_path = os.path.join(parameters.SAVE_DIR, "latest_model.pkl")
    joblib.dump(model, model_path)
    
    wandb.finish()

if __name__ == "__main__":
    run_train()