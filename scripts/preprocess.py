import os
import pandas as pd
import numpy as np
import ast
import joblib  # 인코더 저장을 위해 추가
from sklearn.preprocessing import LabelEncoder

def preprocess_tmdb_data(input_filename="movies_manual.csv", output_filename="movies_cleaned.csv"):
    save_dir = "./data-prepare"
    input_path = os.path.join(save_dir, input_filename)
    output_path = os.path.join(save_dir, output_filename)

    if not os.path.exists(input_path):
        print(f"오류: 수집된 파일({input_path})이 없습니다. 데이터 수집을 먼저 실행하세요.")
        return

    df = pd.read_csv(input_path)
    print(f"전처리 시작 - 원본 데이터 크기: {df.shape}")

    cols = ['id', 'title', 'budget', 'revenue', 'runtime', 'genres', 'release_date', 'vote_count', 'vote_average']
    # 혹시 모를 오류 방지를 위해 실제로 존재하는 컬럼만 선택
    available_cols = [c for c in cols if c in df.columns]
    df = df[available_cols]
    
    # 결측치 처리
    df = df.dropna(subset=['runtime', 'release_date','budget', 'revenue'])
    df = df[df['vote_count'] > 5] # 미니프로젝트용으로 기준을 낮춤

    # 장르(genres) 처리
    def extract_genre(genre_str):
        try:
            # 문자열 형태의 리스트['{...}']를 실제 리스트로 변환
            import ast
            genres = ast.literal_eval(genre_str)
            if isinstance(genres, list) and len(genres) > 0:
                return genres[0]['name']
            return 'Unknown'
        except:
            return 'Unknown'

    df['main_genre'] = df['genres'].apply(extract_genre)

    # 장르 인코딩 (Label Encoding)
    print("장르 인코딩 진행 중...")
    le = LabelEncoder()
    df['main_genre_encoded'] = le.fit_transform(df['main_genre'].astype(str))

    # 나중에 쓸 수 있도록 인코더 저장 (중요!)
    encoder_path = os.path.join(save_dir, "genre_encoder.pkl")
    joblib.dump(le, encoder_path)

    # 날짜 데이터 처리 (연도 추출)
    df['release_year'] = pd.to_datetime(df['release_date']).dt.year

    # 불필요한 원본 컬럼 삭제
    df_clean = df.drop(['genres', 'release_date','main_genre'], axis=1)

    # 결과 저장 (상대 경로 폴더 확인 후 저장)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    df_clean.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"전처리 완료! 저장 위치: {output_path}")
    print(f"최종 변수(Feature) 리스트: {df_clean.columns.tolist()}")
    print(f"인코더 저장 위치: {encoder_path}")
    return df_clean

if __name__ == "__main__":
    preprocess_tmdb_data()