[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/xoFPmgXs)
## 👨‍🏫 프로젝트 소개
## TMDB 데이터를 활용한 영화 평점 예측 서비스 & MLOps Flow 이해하기 <br> <br>
## Team
<table>
  <tr>
    <td> <div align=center> 👑 </div> </td>
    <td> <div align=center> 🙍 </div> </td>
    <td> <div align=center> 🙍 </div> </td>
    <td> <div align=center> 🙍 </div> </td>
  </tr>
  <tr>
    <td> <div align=center> <b>이진성</b> </div> </td>
    <td> <div align=center> <b>고아연</b> </div> </td>
    <td> <div align=center> <b>윤소정</b> </div> </td>
    <td> <div align=center> <b>한혜숙</b> </div> </td>
  </tr>
  <tr>
    <td> <div align=center> <img width="1024" height="1024" alt="Image" src="https://github.com/user-attachments/assets/0630c3b4-e9e8-40e9-8909-650aa91ca71e" /> </td> <!--이진성-->
    <td> <div align=center> <img width="1024" height="1024" alt="Image" src="https://github.com/user-attachments/assets/1f42b6d4-7d84-453b-ac51-e5439f420303" /> </td> <!--고아연-->
    <td> <div align=center> <img width="1024" height="1024" alt="Image" src="https://github.com/user-attachments/assets/749df542-8d79-4512-abe5-c98abb13b31f" /> </td> <!--윤소정-->
    <td> <div align=center> <img width="1024" height="1024" alt="Image" src="https://github.com/user-attachments/assets/f0ffe120-6645-4ff5-8de0-329a624906b5" /> </td> <!--한혜숙-->
  </tr>
  <tr>
    <td> <div align=center> <a href="https://github.com/wlstjd6524"> <img alt="Github" src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/> </a> </div> </td>
    <td> <div align=center> <a href="https://github.com/ayoun88"> <img alt="Github" src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/> </a> </div> </td>
    <td> <div align=center> <a href="https://github.com/Lucia128"> <img alt="Github" src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/> </a> </div> </td>
    <td> <div align=center> <a href="https://github.com/lxlfoo"> <img alt="Github" src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/> </a> </div> </td>
    </tr>
</table>

## 💻 개발환경 및 도구
- Python 3.10.0
- Pytonn Library
  - Numpy  
  - Pandas
  - Scikit-learn
  - LightGBM
  - joblib
- Amazon S3 (Simple Storage Service)
- Amazon Web Service (AWS) EC2
- Apache Airflow
- Docker
- FastAPI
- Uvicorn
- Pydantic
- Jinja2
- dotenv
- Linux File System & Linux Automated System
  - systemd
  - curl
  - scp


## 📏 프로젝트 목적
MLOps PipeLine 구조와 TMDB 데이터를 활용한 영화 평점 예측 시스템 프로젝트 입니다.

TMDB (The Movies DataBase) 데이터를 기반으로 앞으로 개봉할 새로운 영화에 대한 평점을 예측하여 반환하여 사용자에게 해당 결과를 제공하는 것을 목표로 설정했습니다.

## 📁 프로젝트 구조
```text
mlops/
├─ airflow_pipeline/
│  ├─ airflow/
│  │  ├─ dags/
│  │  │  └─ model_train_dag.py
│  │  │     
│  │  │
│  │  ├─ logs/
│  │  │  ├─ dag_id=model_training/
│  │  │  │  └─ run_id=manual__2025-12-30T08_03_08.073557+00_00/
│  │  │  │     
│  │  │  │
│  │  │  ├─ dag_processor_manager/
│  │  │  │  └─ dag_processor_manager.txt
│  │  │  │
│  │  │  └─ scheduler/
│  │  │     ├─ 2025-12-30/
│  │  │     │  └─ model_train_dag.py
│  │  │     └─ latest/
│  │  │        └─ model_train_dag.py
│  │  │
│  │  ├─ plugins/           
│  │  ├─ src/                
│  │  ├─ .env
│  │  ├─ .env.common
│  │  ├─ Dockerfile
│  │  ├─ entrypoint
│  │  ├─ env.common.template
│  │  └─ requirements-airflow.txt
│  │
│  ├─ train/
│  │  ├─ src/
│  │  │  └─ model/
│  │  │     ├─ data_loader.py
│  │  │     ├─ features.py
│  │  │     ├─ train_model.py
│  │  │     ├─ evaluate.py
│  │  │     ├─ save.py
│  │  │     ├─ train_orchestration.py
│  │  │     ├─ utils.py
│  │  │     ├─ main.py
│  │  │     └─ __init__.py
│  │  │    
│  │  │
│  │  ├─ Dockerfile
│  │  └─ requirements-train.txt
│  │
│  └─ webserver_config.py
│    
│
├─ data-prepare/
│  ├─ dags/
│  │  └─ collector_dag.py
│  │  │  
│  │  │
│  ├─ data_prepare/
│  │  ├─ collector.py
│  │  ├─ preprocessor.py
│  │  ├─ load_test.py
│  │  └─ main.py
│  │
│  ├─ Dockerfile
│  ├─ requirements.txt
│  └─ .env.template
│
├─ modeling/
│  ├─ src/
│  │  └─ ...                
│  │
│  ├─ Dockerfile
│  ├─ docker-compose.yml
│  ├─ requirements.txt
│  ├─ run_experiment.py
│  ├─ .dockerignore
│  └─ .gitignore
│
└─ serving/
   ├─ app/
   │  ├─ main.py
   │  ├─ model_service.py
   │  ├─ s3_io.py
   │  ├─ schemas.py
   │  └─ templates/
   │     └─ results.html
   │     
   │
   ├─ models/
   │  ├─ metadata.json
   │  ├─ metrics.json
   │  └─ model_bundle.joblib
   │  
   │
   ├─ Dockerfile
   ├─ docker-compose.yml
   ├─ requirements.txt
   └─ .env
```

## 🔨 프로젝트 시스템 아키텍처
<img width="1536" height="1024" alt="Image" src="https://github.com/user-attachments/assets/01c0a0d2-8cec-4e8f-9ed1-d5da526fd0c6" />

## ✍ 구현 기능
### 0. EC2 서버 셋팅
해당 리눅스 Ubuntu 기반 서버는 향후 팀원들이 구현한 PipeLine 이 로컬에서 도는걸로 국한되는게 아니라 서버환경에서도 병렬적으로 작업했던 모든 PipeLine 이 돌아가는지 확인하기 위한 서버를 셋팅하였습니다.
각 파이프라인 마다 인스턴스를 하나씩 잡고 해당 환경에 맞는 Linux System 을 이용하여 기본 환경만 구축을 진행하였습니다.

EC2 서버의 보안그룹에는 프로젝트의 다른 팀원들의 접근성을 위해 IP Section 은 모든 IP 가 참여할 수 있게 열어놨으며, 이 문제점은 보안에 굉장히 취약하므로 향후 화이트리스트를 적용하여 보안을 강화해야할 의무가 있습니다.
또한 Airflow, FastAPI 의 접근을 위해 8000, 8080 TCP Port 를 추가하였습니다.

다음으로 Airflow 의 스케줄링을 위해 인스턴스가 켜지자마자 docker file 이 자동으로 Build 할 수 있도록 자동화 System 까지 추가로 진행하였습니다.

### 1. 데이터 자동 수집 과 전처리
TMDB Discover API 를 이용해 학습데이터와 추론데이터를 매일 수집합니다.

데이터 누출 방지를 위해 학습데이터와 추론데이터는 완전 분리되어 다른 기간을 수집하며, 매일 새로운 데이터로 모델을 학습시킬 수 있도록 데이터를 날짜변 versioning 하여 적재합니다.

- 학습데이터 : 2024년부터 매일 10년 단위로 10,000건의 데이터를 과거로 거슬러 올라가며 적재 (ex : 2015 ~ 2024 > 2005 ~ 2014 > 1995 ~ 2024..)
- 추론데이터 : 2025년부터 6개월 이후 개봉예정작까지 총 1년 6개월을 기준으로 매일 새로운 데이터 추가

원본은 raw 폴더에 json 으로 변경 없이 저장되고 자동으로 전처리 과정을 거쳐 preprocess 폴더에 csv 파일 형태로 저장됩니다.

<AWS S3 디렉토리 구조>
```
mlops1-s3-v1/
├─ raw/
│  ├─ train/
│  │  └─ train_YYYYMMDD.json
│  └─ inference/
│     └─ inf_YYYYMMDD.json
│
└─ preprocess/
   ├─ train/
   │  └─ train_YYYYMMDD.csv
   └─ inference/
      └─ inf_YYYYMMDD.csv

```

<프로그램 목록>
 - collector_dag.py: Airflow 실행
 - main.py: 진입점
 - collector.py: 데이터수집
 - preprocessor.py: 전처리
 - load_test.py: 데이터로딩 샘플코드

Data PipeLine 은 Airflow 스케줄러를 통해 사람의 개입 없이 매일 자동으로 AWS S3 에 수집됩니다.
학습데이터와 추론데이터의 task 가 분리되어 학습데이터 수집 성공시 추론데이터가 수집됩니다.

스케줄링 주기 : daily (크론탭 0 0 * * *) UTC 기준 (한국시간 9:00 am)

만약 서버 중단이나 오류 등으로 누락된 날이 발생하면 서버재기동시 자동으로 backfill이 적용됩니다.
데이터 정합성에 문제가 없도록 날짜별로 job이 여러개 병렬로 돌아 누락된 날부터 데이터를 다시 쓰며 멱등성(Idempotency)을 검증합니다.
(ex: 오늘이 30일이고 28일부터 서버 중단시 Airflow를 재기동하면 28일, 29일, 30일치 job이 각각 날짜에 맞게 자동 실행 및 S3 적재)

AWS EC2에서 독자적인 인스턴스로 실행되며
Dockerfile을 이용해 data pipeline용 docker를 분리했고
Airflow는 이 docker image를 이용해 가상환경으로 실행됩니다.
Airflow는 3.1.5 버전을 사용하여 개발편의성을 높였습니다.

### 2. TMDB Data 를 기반으로 한 추천 모델
- 선정모델 : LightGBM
- 모델 선정 이유 : 빠른 학습 + 안정적인 성능 확보 가능
- 모델 학습 구조 : S3 데이터 로드 → 피처 변환 → LightBGM 학습 → Validation 평가 → 모델 / 메타데이터 저장
- 평가 지표 : RMSE(주 지표), MAE, R²

각 Python 스크립트 별 주요 기능 구현 요소들은 다음과 같습니다.

- data_loader.py : S3에서 최신 CSV 데이터를 탐색·다운로드하여 DataFrame과 메타데이터를 함께 반환하는 데이터 로딩 모듈
- features.py : 수치형·범주형·텍스트 피처를 결합한 ColumnTransformer 기반의 전처리 파이프라인 정의 파일
- train_model.py : 데이터 정제, train/valid 분리, 전처리 적용, LightGBM 학습 및 일부 파라미터를 W&B에 기록하는 모델 학습 단계
- evaluate.py : validation 데이터 기준으로 RMSE/MAE/R²를 계산하고 W&B에 로깅하는 모델 평가 모듈
- save.py : 학습된 모델과 전처리 객체를 pickle로 저장하고 메타데이터·지표를 아티팩트로 관리하는 모델 저장 단계
- train_orchestration.py : train → evaluate → save 단계를 순차적으로 실행하는 학습 파이프라인 오케스트레이션 모듈
- utils.py : seed 고정, run 디렉토리 생성, JSON 저장, W&B 사용 여부 판단 등 공통 기능을 제공하는 유틸리티 모듈
- main.py : S3에서 최신 데이터를 로드하고 seed/W&B 초기화 후 전체 학습 파이프라인을 실행하는 학습 엔트리포인트
- run_experiment.py : W&B 실험 구조와 로그 방식 이해를 위한 샘플 실험 스크립트 

### 3. AutomatedPipeLine
모델링 과정을 통해 나온 모델 과 S3에 적재된 데이터를 가져와서 학습 -> 평가 -> 결과 아티팩트 저장 의 흐름을 자동화하는 PipeLine 을 구현했습니다.

전체적인 PipeLine 은 다음과 같습니다.

```
Airflow Scheduler Start -> DockerOperator Execution -> mlops-train Docker Container Execution -> S3에 있는 최신 학습 Data Load -> Model 학습 -> Model 평가 -> Model Result(Model bundle + metadata) 생성 -> S3에 Model Result UpLoad
```

에어플로우 실행과 학습 로직은 서로 독립적인 Docker Image 에서 실행되며, 모델 학습을 수행하는 도커에서 command 명령으로 model/main.py 를 실행하며 Python 코드를 직접 실행시키지 않습니다.
```
image = "mlops-train:v1"
```
```
command = "python src/model/main.py"
```

S3에 있는 학습 데이터는 S3 Prefix를 기준으로 가장 최신 csv 파일을 선택하며,
서빙 코드 변경 없이 실행되도록 서빙 코드 디렉토리 구조 기반으로 최신 상태만 유지하도록 진행하였습니다.
서빙 시스템은 최신 Model Bundle 과 Metadata 만 참고하면 되기에 작동에는 문제가 없으나 데이터 관리를 위해 향후 날짜별로 쌓는 구조로 업데이트 하면 좋을 것 같습니다.

환경 변수는 역할에 따라 분리되었으며, 스케줄러는 매일(@daily, UTC 기준 한국시간 09:00 am) 작동하며, 과거 데이터는 실행하지 않도록 하였습니다. 일시적으로 실행이 실패할 시 1번 재시도를 진행하게 됩니다.
```
airflow/.env.common # AWS 인증 정보, 공통 설정
train/.env.train # 학습 전용 설정
```
```
schedule = "@daily"
catchup = False
retries = 1
```

### 4. 모델 서빙
Airflow 를 통해 학습 및 저장된 최신 모델을 S3에서 동기화하여, FastAPI 기반 API 서버를 통해 실시간 추론 결과를 제공하는 서빙 환경을 구현하였습니다.
FastAPI 서버는 EC2 상 Docker Compose 로 실행되며, 서버 기동 시 S3에 저장된 최신 Model Bundle 을 자동으로 Load 하여 서빙에 활용하게 됩니다.

우선 FastAPI 서버가 기동 시, S3에 저장된 최신 모델 번들을 불러오며 Airflow 파이프라인을 통해 생성된 모델을 항상 최신 상태로 서빙하게 됩니다.
모델이 최신 상태인지 확인하는 방법은 S3 저장소에 찍혀있는 model_bundle 의 시간값을 확인 할 수 있고, 다음 CLI Log Checking를 통해도 확인이 가능합니다.

```
[MODEL SYNC] downloading s3://mlops1-s3-v1/bundle/latest/model_bundle.joblib
[MODEL SYNC] done
```

또한 REST API 기반으로 추론 기능을 수행합니다. JSOn 형태의 입력 데이터를 받아 모델을 추론 수행을 진행하며, Pydantic 기반으로 스키마 검증을 통해 입력 데이터의 안정성을 확보합니다.
또한 예측 결과를 JSON 및 HTML 형태로 제공하는데, 사용자의 접근성을 위해 HTML 형태로 제공하는 것에 조금 더 초점을 두었습니다.

```
# Example Request , json
{
  "title": "Example Movie",
  "release_year": 2024,
  "popularity": 123.4,
  "vote_count": 560
}

# Example Response
{
  "prediction": 7.82
}
```

Swagger UI 기반으로 테스트 환경을 제공하며 http://<FastAPI_EC2_IP>:8000/docs 를 통해 해당 endpoint 에 진입하여 key value 값을 전달하여 별도의 클라이언트 없이 API 테스트가 가능합니다.

Jinja2 Template Engine 을 사용하여 사용자에게 조금 더 맞춤으로 제공할 수 있는 HTML 페이지로 렌더링 하여 제공하게 됩니다.

<img width="1897" height="911" alt="Image" src="https://github.com/user-attachments/assets/e9ea34b1-8f2b-4923-af71-a9c3bb49bced" /> 

또한 초반 리눅스 서버 셋팅으로 인해 EC2 인스턴스 실행시 systemd 를 통해 자동으로 Docker Compose 가 실행되며 FastAPI 서버가 무중단 기동이 가능하도록 설정하였습니다.

## 🚨 문제 및 인사이트 도출
### 1. Team MLOps Flow 의 잘못된 파악으로 역할군 정의 지연 문제
#### 문제
프로젝트 초기 단계에서 MLOps의 최소 단위 흐름을 명확히 이해하지 못한 채, 각자 단일 파이프라인 구현이 아닌 완전한 MLOps 구조를 한번에 구현하려는 방향으로 논의가 확장되었습니다.
이로 인해 각 역할군에 대한 범위 정의가 지연되었고, 초기 일정의 상당 부분이 회의에 소모되는 문제가 발생하였습니다.

#### 해결
강사님께 빠른 피드백을 받았고, 피드백 기반으로 팀원들과 빠르게 소통하여 역할규정을 정의하였습니다.

---

### 2. 모델링 저장 확장자 관련 규칙 정의 문제
#### 문제
프로젝트를 병렬적으로 진행하면서, 모델링 파이프라인 과 서빙 파이프라인 간의 모델 산출물에 대한 사전 합의가 이루어지지 않았었습니다.
따라서 서빙 파이프라인 에서 코드 구현을 확인하기 위해 작성했던 코드 구조가 어떤 방향성을 가져야 하는지에 대한 고민과, 향후 모델링이 끝난 후에 산출물과 서빙 구조의 차이점이 발생하여 서빙 코드를 수정해야 했고,
초기 연동 과정에서 재작업으로 인해 시간 지연이 발생하였습니다.

#### 해결
재작업이 진행되긴 했지만 중간에 해당 이슈에 대한 커뮤니케이션을 진행하였고, 해당 커뮤니케이션을 통해 병렬작업 중에 모델 결과물을 통일 할 수 있어서 불필요한 재작업이 더 발생하지 않도록 조정하였습니다.

---

### 3. 추론 Predict 값이 비정상적으로 튀는 문제
#### 문제
FastAPI /docs 환경에서 직접 추론 테스트를 진행하던 중, 모델의 예측 값이 반복적으로 비정상적인 수치(약 21 근사치)로 출력되는 현상이 있었습니다.

#### 해결
초기에는 모델의 파라미터값이나 데이터의 결측치 또는 이상치 처리에 대한 부분이 의심되어 데이터 CSV 와 모델의 파라미터값을 열어봤는데 이상이 없었습니다.
결국엔 /docs 환경에서 추론을 하기 위해서 넘겨준 Key Value 값에 이상한 feature 가 들어있어서 값이 튄 걸 알아냈고 다른 Key Value 값을 만들어 전달하여 해결하였습니다.

---

### 4. 모델 버전 관리에 대한 문제
#### 문제
FastAPI 서빙 단계에서, 현재 서빙 중인 모델이 Airflow 파이프라인(Automated PipeLine)을 통해 S3 에 저장된 최신 모델인지 여부를 확인하기 어려운 문제가 발생하였습니다.
저희는 일단 모델 성능보다는 흐름구조 구현에 초점을 맞췄기 때문에 모델 버전을 하나만 가용하여 사용하였고, 초기 테스트 과정에서는 로컬에 저장된 모델을 불러서 쓴 이후 서버에서는 S3에 있는 모델 결과물을 불러오기로 목표를 설정했습니다.
모델 버전을 한가지만 사용하다 보니 이게 로컬에서 온 모델인지, S3에서 온 모델인지 판단이 어려웠던 문제가 있었습니다.

#### 해결
서버에 올라간 S3 Load 방식의 Log 를 찍어 본 결과 로컬에서 모델이 불러와지는 Log 를 잡아냈고, 구현 부분에 코드를 수정하여 S3 저장소에서 모델을 불러올 수 있게 로직을 수정하였습니다.

---

### 5. 서빙 추론 데이터 구성 및 API 변경 이슈
#### 문제
서빙 단계에서 추론용 데이터가 필요하여, 학습 데이터를 분리하여 활용하는 방안에 대해 논의하였습니다. 모델러와 서빙 개발자의 편의를 위해 S3 내 학습 데이터와 추론 데이터를 각각 독립적인 폴더 구조로 적재하기로 결정하였는데,
학습 데이터와 추론 데이터를 별도로 호출하는 과정에서 데이터 누수(Data Leakage) 문제가 발생하였으며, 기존에 사용했던 TMDB Popular API는 해당 구조에 적합하지 않다는 한계가 드러났습니다.

#### 해결
데이터 누수를 방지하기 위해 Popular API 사용을 중단하고, 조견 기반 조회가 가능한 TMDB Discover API로 변경하였습니다.
이를 통해 학습 데이터와 추론 데이터를 명확히 분리하여 수집할 수 있는 구조를 구축하였습니다.

---

### 6. 데이터 부족 및 TMDB API 호출 제한 문제
#### 문제
초기 데이터 수집량이 부족하여 모델이 안정적으로 학습되지 않는 문제가 발생하였습니다. 데이터 양을 늘리는 과정에서 TMDB API 의 최대 수출 제한(10,000건)에 도달하게 되었으며, 단일 쿼리 방식으로는 추가 데이터 확보가 어려운 상황이였습니다.

#### 해결
API 호출 단계에서 날짜, 평점, 장르 등의 필터링 쿼리 조건을 일자별로 분산하여 적용하였습니다. 이를 통해 매일 서로 다른 조건의 데이터를 수집하도록 설계함으로써 API 호출 제한을 우회하고 데이터 규모를 점진적으로 확장하였습니다.

---

### 7. 서버 미기동으로 인한 데이터 누락 및 Backfill 처리 문제
#### 문제
서버가 기동되지 않은 날이 발생하면서, 해당 날짜의 데이터 적재가 누락되는 문제가 확인되었습니다.
초기에는 Airflow Variables를 활용하여 수동으로 offset을 조정하는 방안을 고려하였으나, 데이터 정합성 및 운영 안정성 측면에서 한계가 존재하였습니다.
또한 누락된 날짜의 데이터를 재적재하는 과정에서 여러 날짜의 데이터가 당일 날짜 파일로 반복적으로 덮어써지는 현상이 발생하였습니다.

#### 해결
수동 offset 조정 방식 대신, 서버 재기동 시 누락된 날짜를 자동으로 계산하여 적재하는 backfill 로직을 개발하였습니다.
이 과정에서 offset task를 제거하고, Airflow가 제공하는 스케줄 정보를 기반으로 누락된 날짜부터 데이터를 적재하도록 개선하였습니다.
특히 파일 저장 시 datetime.now() 대신 Airflow의 **logical_date**를 활용하여, 누락된 날짜의 데이터가 실제 해당 날짜 기준으로 적재되도록 수정함으로써 데이터 정합성과 멱등성(Idempotency)을 검증하였습니다.

---

### 8. BashOperator 와 Python Operator 로 작성된 초기 DAG 연동성 문제
#### 문제
초기 DAG는 BashOperator와 PythonOperator를 기반으로 작성되었습니다. 그러나 해당 방식은 로컬 환경에서 Bash를 직접 실행하는 구조로, 실행 환경에 대한 의존성이 발생하고 자동화된 파이프라인으로서의 한계가 존재하였습니다.
멘토링 과정에서 “BashOperator는 로컬 환경에 종속되기 쉬우며, DockerOperator 또는 클라우드 기반 컨테이너 오퍼레이터를 사용하는 것이 바람직하다” 라는 피드백을 받았습니다.

#### 해결
이에 따라 기존 DAG를 DockerOperator 기반으로 전환하고, 학습 및 처리 스크립트가 도커 컨테이너 내부에서 독립적으로 실행되도록 구조를 수정하였습니다.
이를 통해 실행 환경에 대한 의존성을 제거하고, 동일한 조건에서 언제든지 재현 가능한 파이프라인을 구성할 수 있었습니다.

---

### 9. Docker 이미지 관리 및 Airflow 연동 불일치 문제
#### 문제
Airflow DockerOperator 실행 시, 의도한 학습 이미지(mlops-train:v1)가 아닌 다른 이미지가 실행되거나, 이미지 내부 코드가 최신 상태가 아닌 문제가 발생하였습니다.
이는 다음과 같은 원인에서 비롯되었습니다.

- docker build 시 이미지 태그를 명확히 구분하지 않음
- docker-compose, Airflow DAG, Dockerfile 간 역할 분리 부족
- Airflow 컨테이너와 모델 학습 컨테이너의 실행 디렉토리 및 환경변수 혼재
  
#### 해결
Airflow와 모델 학습 도커의 실행 디렉토리를 명확히 분리하고, 각각의 .env 파일을 통해 환경변수를 독립적으로 관리하도록 개선하였습니다.
또한 이미지 빌드 → docker-compose down → 재실행 → Airflow 연동(sync) 이라는 명확한 작업 순서를 정립하여, 항상 최신 코드가 반영된 이미지가 실행되도록 운영 방식을 정리하였습니다.

---

### 10. 로컬 및 컨테이너 환경에서 실행할 때 성능이 미묘하게 달라지는 문제
#### 문제
로컬 환경과 컨테이너 환경에서 동일한 코드를 실행했음에도 불구하고 모델 성능이 미묘하게 달라지는 문제가 발생하였습니다.

#### 해결
라이브러리 버전 차이와 random seed 가 고정되지 않은 상태에서 실험을 진행하여 성능이 달라지는 원인을 찾아 seed 고정 및 버전 차이를 완화하였습니다.

---

### 11. W&B 대시보드에서 실험 간 config 값이 동일하게 기록되는 현상
#### 문제
LightGBM 학습 과정에서 하이퍼파라미터 값을 변경하며 여러 실험을 수행했음에도 불구하고, W&B 대시보드 상에서는 실험 간 config 값이 모두 동일하게 표시되는 문제가 발생하였습니다.

#### 해결
이는 실제 학습 함수에 전달된 파라미터 딕셔너리와 wandb.config에 기록된 값이 서로 동기화되지 않은 상태에서 실험을 진행했기 때문으로 확인되었습니다.
즉, 코드 상에서는 파라미터가 변경되었지만, W&B에는 해당 변경 사항이 명시적으로 전달되지 않았습니다. 
W&B는 코드 변경을 자동으로 추적하는 도구가 아니라, 실험 조건을 명시적으로 기록해야 하는 실험 관리 도구라는 특성을 이해하게 되었으며, 실험 설계 시 config 중심으로 구조를 설계하고 entrypoint 단계에서 실험 조건을 고정하는 방식이 필요하다는 인사이트를 얻었습니다.

## 🔎 프로젝트 한계 및 개선사항

## 📍 회고
👑 이진성 : 지금까지 몇 번의 프로젝트 들이 있었지만, 이렇게 고도화되고 높은 개념이 적용되는 프로젝트는 처음이였던지라 많은 어려움이 있었는데 그래도 책임감 강한 팀원들 덕분에 성공적으로 프로젝트를 마무리 했던 것 같습니다. 처음에 공부했을 때에는 굳이 왜 이런 어려운 흐름에 프로젝트를 올려서 진행하는거지? 에 대한 의문이 들었는데 실제로 해당 흐름에 들어가서 전체적으로 프로젝트를 바라보고 진행해보니 MLOps 가 왜 필요한지에 대한 중요성을 느꼈던 프로젝트 였던 것 같습니다. 매번 느끼는거지만 부족한 팀장과 함께해준 팀원들에게 죄송하고 감사했습니다. 뜻깊은 프로젝트 였습니다.

🙍 고아연 : MLOps에 대한 개념을 숙지하고 프로젝트 과정에 참가했다고 생각했는데, 전혀 아니었어서 기능 구현을 개발하는 동시에 개념까지 수시로 점검해야 했어서 시간이 오래 걸렸던게 아쉽고 팀원들에게 죄송한 마음이 들었습니다. 작업 과정에서 '문제 및 인사이트'에 적었던 것 외에도 로컬과 콘테이너의 경로 착각, Airflow 실행 오류 등 많은 이슈가 발생하여 밤새 트러블슈팅에 메달렸었는데, 마지막에 정상 작동하는 것도 좋았지만 여기에 도달하기 까지의 과정이 의미있게 느껴졌습니다. 이번 프로젝트를 통해 서버, 에어플로우, 리눅스, 깃을 직접 사용해보면서 이해도를 높일 수 있어서 유익했습니다. 시간 상 작업하지 못했던 CI/CD, 버전관리, 서빙을 자동화하는 DAG를 추가적으로 구현하여 자동화 수준을 높이면 더 좋을 것 같습니다. 그리고 저의 이해도 수준과 기능 구현 스킬도 높이면 더욱 좋을 것 같다고 생각합니다.

🙍 윤소정 : 본격적으로 프로젝트를 진행하기 전 MLOps 개념을 추가로 학습하고 시작했음에도, 실제 프로젝트를 진행하면서는 각 단계가 어떻게 구현되는지를 따라가는 데 급급했던 것 같습니다. 특히 담당했던 모델링 파트에서는 LLM의 도움을 받았음에도 코드를 작성하는 것뿐만 아니라, 리팩토링 과정에서도 많은 시간이 소요되었고, 코드 작성에 대한 어려움을 느꼈습니다.
상대적으로 난이도가 높지 않은 파트를 맡았음에도 다른 파트에 충분한 도움을 드리지 못한 점에 대해 팀원들에게 미안한 마음이 들었습니다. 또한 실험 버전 관리 부분까지 충분히 구현하지 못한 점 역시 아쉬움으로 남았으며, 추후에는 CI/CD까지 포함하여 실제 환경에 더 가까운 MLOps 파이프라인을 구현해보고 싶다는 생각이 들었습니다.
이번 프로젝트를 통해 여러 MLOps 도구를 직접 사용해보면서, 단순히 개념을 아는 것보다 실제로 경험하고 적용해보는 과정이 실력을 키우는 데 훨씬 중요하다는 점을 체감했습니다. 동시에 아직 부족한 부분이 많다는 것도 다시 한 번 인식하게 되었고, 이를 바탕으로 이후 프로젝트에서는 더 주도적으로 학습하고 기여하고자 합니다.

🙍 한혜숙 : 현업 개발자들보다도 의욕적이고 책임감 강한 팀원들과 함께 일한 덕분에 짧은 기간에 비해 기대 이상의 성과를 이뤄냈다고 생각합니다.
DevOps 플젝들에 참여해 본 경험은 있으나 인프라 담당이 아니었던지라 직접 Docker를 설치하고 관리해볼 경험은 거의 없었는데
이번 기회에 로컬로도 AWS로도 자유자재로 돌려보고 실험해 볼 기회를 얻어 즐거웠습니다.
MLOps는 혼자 공부했으면 절대 지금처럼 단기간에 습득할 수 없는 고급개념이다 보니
수준 높은 강사님, 적극적이고 성실한 팀원들과 함께 플젝할 수 있었던 경험이 매우 감사했습니다.

## 👨‍👩‍👧‍👧 협업
#### 🤝 협업일정 및 방식
- 협업일정 : 정규 수업시간 (09시 ~ 18시) 도 중 시간제약 없이 현재 진행상황 실시간 공유 및 18시 수업 종료 전 금일 진행했던 프로젝트 작업내용 공유
- 협업방식 : Slack
- 미팅방식 : 전 날 자정 전에 내일 회의하면 좋을 내용에 대한 부분 정리해서 협업 커뮤니케이션 애플리케이션(Slack)에 전송, 다음 날 출석체크 후 09:00 에 곧바로 스레드를 활용한 개인의견 작성 및 회의 진행

<img width="1920" height="1920" alt="Image" src="https://github.com/user-attachments/assets/2fadfb83-d73e-4d1f-b78d-f498fa42ba30" />

#### 📋 일정 및 프로젝트 관리 툴
- Notion
<img width="1658" height="893" alt="Image" src="https://github.com/user-attachments/assets/cd4b8fb1-7f8b-4d90-9ed5-4e24f5893ca0" />

## 🌐 기술스택
[Fast API] : https://fastapi.tiangolo.com/ko/tutorial/first-steps/ 

[Docker] : https://www.hostinger.com/kr/tutorials/what-is-docker?utm_campaign=Generic-Tutorials-DSA|NT:Se|Lang:KR|LO:KR&utm_medium=ppc&gad_source=1&gad_campaignid=23281809969&gbraid=0AAAAADMy-hYbi423OfY9fiAiGlw_D9uWT&gclid=CjwKCAiAjc7KBhBvEiwAE2BDOTHIv6CEVY6ayhwrpxpzo_6hmrH6-oUIHQ5lKd42oblywGVEBbTByhoCu3IQAvD_BwE 

[EC2] : https://docs.aws.amazon.com/ko_kr/AWSEC2/latest/UserGuide/EC2_GetStarted.html 
