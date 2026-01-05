import json
import os
import time
from datetime import datetime, timedelta

import boto3
import requests

class TMDBCollector:
    def __init__(self, language='ko-KR', request_interval_seconds=0.4):
        self._base_url = os.environ.get('TMDB_BASE_URL')
        self._api_key = os.environ.get('TMDB_API_KEY')
        self._language = language
        self._request_interval_seconds = request_interval_seconds
        self._bucket_name = os.getenv('S3_BUCKET_NAME')

        # 일간 데이터 수집량을 필요시 유연하게 변경하기 위해 상수화
        self.TRAIN_PAGE_SIZE = 500 # 매일 10,000건 (페이지당 20건 * 500페이지)
        self.INF_PAST_MONTHS = 12  # 추론용은 과거 1년 ~ 미래 6개월
        self.INF_FUTURE_MONTHS = 6

    # TMDB API 호출 (discover api)
    def _fetch_from_tmdb(self, endpoint, params):
        full_params = {
            'api_key': self._api_key,
            'language': self._language,
            **params
        }

        target_url = f'{self._base_url}{endpoint}'
        # print(f'Target URL: {target_url}')

        try:
            response = requests.get(f'{target_url}', params=full_params)
            if response.status_code == 200:
                return response.json().get('results', [])
            else:
                print(f'[API 호출 실패] Status: {response.status_code} | Reason: {response.reason}')
                print(f'{response.text}')
                return []
        except Exception as e:
            print(f'요청 중 예외 발생: {e}')

        return []

    # [학습용] 인기 영화를 페이지 단위로 수집
    def get_train_data(self, offset, target_date=None):
        # 10년 단위 계산
        base_date = datetime.strptime(target_date, '%Y%m%d') if target_date else datetime.now()
        current_year = base_date.year

        # 0이면 2015~2024, 1이면 2005~2014
        end_year = current_year - 1 - (offset * 10)
        start_year = end_year - 9

        train_start_date = f'{start_year}-01-01'
        train_end_date = f'{end_year}-12-31'

        movies = []
        print(f'학습용 데이터 수집 시작 (기간: {train_start_date} ~ {train_end_date})')

        for page in range(1, self.TRAIN_PAGE_SIZE + 1):
            params = {
                'sort_by': 'popularity.desc',
                'primary_release_date.gte': train_start_date,
                'primary_release_date.lte': train_end_date,
                'vote_count.gte': 20, #최소 투표 수 조건 추가
                'include_adult': 'false', #성인물 차단
                'page': page
            }
            res = self._fetch_from_tmdb('/discover/movie', params)
            if not res: break

            movies.extend(res)
            time.sleep(self._request_interval_seconds)

        return movies

    # [추론용] 3개월 전부터 미래 3개월 (개봉예정작) 사이 신규 영화만 수집
    def get_inference_data(self, target_date=None):
        # target_date 기준으로 시작/종료일 계산
        base_date = datetime.strptime(target_date, '%Y%m%d') if target_date else datetime.now()

        inf_start_date = (base_date - timedelta(days=30 * self.INF_PAST_MONTHS)).strftime('%Y-%m-%d')
        inf_end_date = (base_date + timedelta(days=30 * self.INF_FUTURE_MONTHS)).strftime('%Y-%m-%d')

        movies = []
        MAX_INF_PAGES = 500
        print(f'추론용 신규 데이터 수집 시작 (기간: {inf_start_date} ~ {inf_end_date})')

        for page in range(1, MAX_INF_PAGES + 1):
            params = {
                'primary_release_date.gte': inf_start_date,
                'primary_release_date.lte': inf_end_date,
                'sort_by': 'primary_release_date.asc', # 날짜순
                'include_adult': 'false',
                'page': page
            }
            res = self._fetch_from_tmdb('/discover/movie', params)
            if not res: break

            movies.extend(res)
            time.sleep(self._request_interval_seconds)

        return movies

    def save_movies_to_s3(self, movies, mode, target_date=None):
        # Data Versioning (날짜별 파일 생성)
        base_date = target_date if target_date else datetime.now().strftime('%Y%m%d')

        if mode == 'train':
            local_filename = f'raw_train_{base_date}.json'
            s3_filename = f'raw/train/train_{base_date}.json'
        else:
            local_filename = f'raw_inf_{base_date}.json'
            s3_filename = f'raw/inference/inf_{base_date}.json'

        try:
            with open(local_filename, 'w', encoding='utf-8') as f:
                json.dump(movies, f, ensure_ascii=False, indent=4)

            s3 = boto3.client('s3')
            s3.upload_file(local_filename, self._bucket_name, s3_filename)
            print(f'{mode} 원본 데이터 S3 적재 완료: ({s3_filename})')
        except Exception as e:
            print(f'{mode} 원본 데이터 S3 적재 실패: {e}')
        finally:
            # 로컬 임시 파일 삭제
            if os.path.exists(local_filename):
                os.remove(local_filename)
