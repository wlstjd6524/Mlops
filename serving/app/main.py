import os
import time
import requests
from fastapi import Request
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from .schemas import PredictRequest, PredictResponse
from .model_service import model_service
from .s3_io import load_csv_from_s3, sync_model_bundle_from_s3

# .env load
load_dotenv()

# templates 객체 추가 (html 결과 보기 전용)
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


# FastAPI 정책변경으로 lifespan 으로 Model Load
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    sync_model_bundle_from_s3()
    model_service.load()
    yield
    # shutdown, 

# FastAPI 앱 생성
app = FastAPI(title="MLOps Group1 Serving API", version="0.2.0", lifespan=lifespan)

# GET 엔드포인트 지정, / 요청시 아래함수 실행
@app.get("/")
def root():
    return {"message": "Server is running. Try /docs or /health"}

# /health 엔드포인트
@app.get("/health")
def health():
    return {"status": "ok", "model": getattr(model_service, "metadata", {})}

# /predict 엔드포인트
@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    start = time.time()
    try:
        pred = model_service.predict(req.features)
        latency_ms = int((time.time() - start) * 1000)
        print(f"[PREDICT] latency_ms={latency_ms} prediction={pred}")
        return PredictResponse(prediction=pred, model_version="local-dev")
    except ValueError as e:
        # 입력 누락/형식 오류 그니까 즉, 클라이언트 에러 반환
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 서버 에러
        raise HTTPException(status_code=500, detail=str(e))

# /predict_s3 Endpoint
@app.post("/predict_s3")
def predict_s3(s3_key: str):
    start = time.time()

    try:
        # 1. S3에서 CSV 로드
        df = load_csv_from_s3(s3_key)

        print("[DEBUG] columns=", df.columns.tolist())
        print("[DEBUG] sample rows titles=", df["title"].sample(5, random_state=42).tolist())

        if "vote_average" in df.columns:
            print("[DEBUG] vote_average > 10 count =", int((df["vote_average"] > 10).sum()))
            print("[DEBUG] vote_average max =", float(df["vote_average"].max()))
        else:
            print("[DEBUG] inference mode: no vote_average column")


        # 2. 모델이 요구하는 컬럼만 추출
        required_cols = [
            "popularity", "release_year",
            "original_language", "main_genre",
            "overview", "title"
        ]

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in inference CSV: {missing}")

        # vote_count는 inference (예측)에 없으니 기본값 처리
        if "vote_count" not in df.columns:
            df["vote_count"] = 0
            print("[DEBUG] vote_count missing → filled with 0")

        X = df[required_cols + ["vote_count"]].copy()


        # 3. 배치 예측
        preds = model_service.predict_batch(X)

        # DEBUG: 입력 + 예측 미리보기
        preview_df = df.head(5).copy()
        preview_df["prediction"] = preds[:5]

        print("[PREDICT_S3 PREVIEW]")

        preview_cols = ["title", "main_genre", "release_year", "popularity"]
        if "vote_count" in preview_df.columns:
            preview_cols.append("vote_count")
        preview_cols.append("prediction")

        print(preview_df[preview_cols])

        latency_ms = int((time.time() - start) * 1000)

        print(f"[PREDICT_S3] rows={len(df)} latency_ms={latency_ms}")

        return {
            "s3_key": s3_key,
            "rows": len(df),
            "latency_ms": latency_ms,
            "prediction_sample": preds[:5],  # 너무 길어줄 수 있어서 샘플만 추출
            "prediction_head": preds[:10],   # 앞 10개
            "prediction_tail": preds[-10:],  # 뒤 10개
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# html endpoint
@app.get("/results", response_class=HTMLResponse)
def results(
    request: Request, 
    s3_key: str = "preprocess/inference/inf_refined_20251226.csv", 
    limit: int = 24,
    # random 추출을 위해 추가한 index
    random: int = 0,
    seed: int = 42,
    ):

    start = time.time()

    # 1. S3에서 추론용 CSV 로드
    df = load_csv_from_s3(s3_key)

    # 2. inference에서 vote_count 없으면 0 채우기 (/predict_s3에서 했던 방식과 동일)
    if "vote_count" not in df.columns:
        df["vote_count"] = 0

    # 3. 모델 입력 구성 (required_cols)
    required_cols = [
        "popularity", "vote_count", "release_year",
        "original_language", "main_genre",
        "overview", "title"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns in inference CSV: {missing}")

    X = df[required_cols].copy()
    preds = model_service.predict_batch(X)
    df["prediction"] = preds

    # 4. HTML에 보여줄 상위 N개만 구성
    # view = df.head(limit).copy()
    if random == 1:
        view = df.sample(n=min(limit, len(df)), random_state=seed).copy()
    else:
        view = df.head(limit).copy()


    # 포스터 URL 만들기: TMDB_API_KEY 있으면 id로 poster_path 조회
    tmdb_key = os.getenv("TMDB_API_KEY", "").strip()
    poster_base = "https://image.tmdb.org/t/p/w500"
    poster_urls = [None] * len(view)

    if tmdb_key:
        try:
            for i, mid in enumerate(view["id"].tolist()):
                # TMDB Movie Details
                r = requests.get(
                    f"https://api.themoviedb.org/3/movie/{int(mid)}",
                    params={"api_key": tmdb_key, "language": "en-US"},
                    timeout=5,
                )
                if r.status_code == 200:
                    js = r.json()
                    pp = js.get("poster_path")
                    poster_urls[i] = (poster_base + pp) if pp else None
        except Exception:
            # 포스터 실패해도 페이지는 뜨게
            poster_urls = [None] * len(view)

    items = []
    for idx, row in view.iterrows():
        items.append({
            "title": row.get("title"),
            "main_genre": row.get("main_genre"),
            "release_year": int(row.get("release_year")) if row.get("release_year") == row.get("release_year") else None,
            "original_language": row.get("original_language"),
            "popularity": float(row.get("popularity")) if row.get("popularity") == row.get("popularity") else 0.0,
            "prediction": float(row.get("prediction")),
            "poster_url": poster_urls[len(items)] if tmdb_key else None,
        })

    latency_ms = int((time.time() - start) * 1000)

    return templates.TemplateResponse("results.html", {
        "request": request,
        "s3_key": s3_key,
        "rows": len(df),
        "latency_ms": latency_ms,
        "items": items,
    })
