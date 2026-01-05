import os
from datetime import datetime

import boto3
import pandas as pd

class TMDBPreprocessor:
    def __init__(self, movies):
        self._movies = movies if isinstance(movies, list) else []
        self._bucket_name = os.getenv('S3_BUCKET_NAME')

    def run(self, mode, target_date=None):
        if not self._movies:
            print(f'{mode} 모드: 처리할 데이터가 없습니다. 전처리를 중단합니다.')
            return False

        df = pd.DataFrame(self._movies)
        print(f'원본 데이터 건수: {len(df)}')
        # print(df.columns)

        # 추론 모드일 때 평점 관련 컬럼 제거
        if mode == 'inference':
            drop_inf_cols = ['vote_average', 'vote_count']
            df = df.drop(columns=[c for c in drop_inf_cols if c in df.columns])
            print('추론 모드: 평점 데이터(vote_average, vote_count) 제거')

        # 불필요한 컬럼 삭제 (포스터 경로는 원본 json에 남아있음)
        drop_cols = ['adult', 'backdrop_path', 'poster_path', 'video', 'original_title']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # 데이터 정제 (줄거리)
        df['overview'] = df['overview'].str.strip()
        df['overview'] = df['overview'].replace('', None)
        # df = df.dropna(subset=['overview'])

        ## Feature Engineering (release_year, main_genre) ##
        # 1) 개봉일에서 연도만 추출
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df = df.dropna(subset=['release_date']) #변환 실패한 행만 제거됨
        df['release_year'] = df['release_date'].dt.year.astype(int)

        # 2) 장르 ID 리스트에서 첫 번째 대표 장르만 추출 (모델 입력 단순화)
        df['main_genre'] = df['genre_ids'].apply(
            lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 0
        )

        genre_zero = len(df[df['main_genre'] == 0])
        if genre_zero > 0:
            print(f'장르 미정(0) 데이터 {genre_zero}건을 제거합니다.')
            df = df[df['main_genre'] != 0]

        if df.empty:
            print(f'{mode} 모드: 정제 후 남은 데이터가 없습니다.')
            return False

        # 정제 완료된 데이터 확인
        print(f'최종 정제 데이터 건수: {len(df)}')
        # print(df[['title', 'release_year', 'main_genre', 'vote_average']].head())

        # S3에 preprocess 경로로 적재 (.csv)
        base_date = target_date if target_date else datetime.now().strftime('%Y%m%d')

        if mode == 'train':
            pre_filename = f'preprocess_train_{base_date}.csv'
            s3_filename = f'preprocess/train/train_refined_{base_date}.csv'
        else:
            pre_filename = f'preprocess_inf_{base_date}.csv'
            s3_filename = f'preprocess/inference/inf_refined_{base_date}.csv'

        try:
            df.to_csv(pre_filename, index=False, encoding='utf-8-sig')

            s3 = boto3.client('s3')
            s3.upload_file(pre_filename, self._bucket_name, s3_filename)
            print(f'정제 데이터 S3 적재 완료! ({s3_filename})')
        except Exception as e:
            print(f'정제 데이터 S3 적재 실패: {e}')
        finally:
            # 로컬 임시 파일 삭제
            if os.path.exists(pre_filename):
                os.remove(pre_filename)
