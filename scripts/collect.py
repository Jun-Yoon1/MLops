import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("TMDB_API_KEY")
base_url = os.getenv("TMDB_BASE_URL")

def get_movie_details(movie_id, api_key, base_url):
    response = requests.get(
        f"{base_url}/movie/{movie_id}",
        params={"api_key": api_key}
    )
    response.raise_for_status()
    return response.json()

def crawl_tmdb_data(**context): # context: 어떤 인자든 받을 수 있게
    # 저장할 폴더가 없으면 생성 (에러 방지)
    save_dir = "./data-prepare"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"폴더 생성됨: {save_dir}")

    # 인자(ds) 처리: 
    # main.py에서 ds="manual"을 주면 그 값을 쓰고, Airflow에서는 실행 날짜가 자동으로 들어옴
    ds = context.get("ds", "manual")

    print("데이터 수집 시작...")
    movies = []
    for page in range(1, 5+1):
        response = requests.get(
            f"{base_url}/movie/popular",
            params={"api_key": api_key, "page": page}
        )
        response.raise_for_status()
        movies.extend(response.json()["results"])

    detailed_movies = []
    print("상세 정보 수집 중...")
    for movie in movies: 
        detail = get_movie_details(movie["id"], api_key, base_url)
        detailed_movies.append(detail)

    # CSV 저장 (파일명에 ds활용)
    df = pd.DataFrame(detailed_movies)
    file_path = os.path.join(save_dir, f"movies_{ds}.csv")
    df.to_csv(file_path, index=False, encoding='utf-8-sig') 
    
    print(f"파일 저장 완료: {file_path}")
    return len(detailed_movies)

# 실제로 함수를 실행시키는 문장
if __name__ == "__main__":
    try:
        count = crawl_tmdb_data()
        print(f"성공: {count}개의 데이터를 가져왔습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")