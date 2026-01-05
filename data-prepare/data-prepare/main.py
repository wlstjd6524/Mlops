import argparse
from datetime import datetime

from dotenv import load_dotenv

from collector import TMDBCollector
from preprocessor import TMDBPreprocessor

load_dotenv()

def run_pipeline(mode, offset=0, target_date=None):
    if not target_date:
        target_date = datetime.now().strftime('%Y%m%d')

    print(f'현재 실행 모드: {mode}')
    tmdb_collector = TMDBCollector()

    if mode == 'train':
        print(f'학습용 데이터 증분 수집 시작 (offset: {offset})')
        result = tmdb_collector.get_train_data(offset=offset, target_date=target_date)
    else:
        print(f'추론용 신규 데이터 수집 시작')
        result = tmdb_collector.get_inference_data(target_date=target_date)

    if result:
        tmdb_collector.save_movies_to_s3(result, mode, target_date=target_date)

        tmdb_preprocessor = TMDBPreprocessor(result)
        tmdb_preprocessor.run(mode, target_date)
    else:
        print('수집된 데이터가 없습니다.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TMDB Pipeline Main Runner')
    parser.add_argument('--mode', type=str, default='train', help='Set mode: train or inference')

    # Airflow로부터 매일 계산된 값을 받기 위한 인자 추가
    parser.add_argument('--offset', type=int, default=0, help='Year offset for train mode')
    parser.add_argument('--target_date', type=str)

    args = parser.parse_args()
    run_pipeline(mode=args.mode, offset=args.offset, target_date=args.target_date)
