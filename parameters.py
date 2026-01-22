# 실험용 파라미터 설정

WANDB_PROJECT = "movie-rating-predictor"
SAVE_DIR = "./data-prepare"

MODEL_CONFIG = {
    "model_type": "rf",  # "rf", "xgboost", "lgbm" 중 선택
    "params": {
        "n_estimators": 200,      # 나무의 개수
        "learning_rate": 0.05,    # 학습률
        "max_depth": 6,           # 나무 깊이
        "subsample": 0.8,         # 데이터 샘플링 비율
        "random_state": 42
    }
}