import io
import os

import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

def get_latest_refined_data(mode='train'):
    # S3에서 가장 최근 정제된 데이터를 읽어와서 DataFrame으로 반환
    bucket_name = os.getenv('S3_BUCKET_NAME')
    s3 = boto3.client('s3')

    # S3에서 preprocess/ 폴더 내의 파일 목록 가져오기
    prefix = f'preprocess/{mode}/'
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    if 'Contents' not in response or len(response['Contents']) == 0:
        print(f'S3의 {prefix} 경로에 정제된 데이터가 없습니다.')
        return None

    # 가장 최근 파일 찾기 (마지막 파일이 최신)
    files = sorted(response['Contents'], key=lambda x: x['LastModified'])
    latest_file = files[-1]['Key']
    print(f'S3에서 최신 데이터를 가져오는 중: {latest_file}')

    # 파일 읽기
    obj = s3.get_object(Bucket=bucket_name, Key=latest_file)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))

    return df

if __name__ == '__main__':
    # S3 데이터 로드 테스트
    train_data = get_latest_refined_data(mode='train') #훈련용데이터 가져오기
    if train_data is not None:
        print(train_data.head(2))

    inference_data = get_latest_refined_data(mode='inference') #추론용데이터 가져오기
    if inference_data is not None:
        print(inference_data.head(2))
