import os
from scripts.collect import crawl_tmdb_data  # 수집 함수
from scripts.preprocess import preprocess_tmdb_data  # 전처리 함수
from scripts.train import run_train  # 학습 및 WandB 기록 함수

def run_pipeline():
    print("=== MLOps 파이프라인 시작 ===")

    # Step 1: 데이터 수집 (Raw Data)
    print("\n[Step 1] 데이터 수집 중...")
    try:
        # context={} 는 Airflow와 호환성을 위해 넣어둔 인자입니다.
        crawl_tmdb_data(ds="manual")   
    except Exception as e:
        print(f"수집 단계 오류: {e}")
        return

    # Step 2: 데이터 전처리 (Cleansing)
    print("\n[Step 2] 데이터 전처리 중...")
    try:
        preprocess_tmdb_data()
    except Exception as e:
        print(f"전처리 단계 오류: {e}")
        return

    # Step 3: 모델 학습 및 실험 기록 (Train & WandB)
    print("\n[Step 3] 모델 학습 및 WandB 기록 중...")
    try:
        run_train()
    except Exception as e:
        print(f"학습 단계 오류: {e}")
        return

    print("\n=== 전체 파이프라인 실행 완료 ===")

if __name__ == "__main__":
    run_pipeline()