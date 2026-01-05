from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer

NUM_COLS = ["popularity", "vote_count", "release_year"]
CAT_COLS = ["original_language", "main_genre"]
TEXT_COLS = ["overview", "title"]
TARGET_COL = "vote_average"


def _squeeze_1d(x):
    # ColumnTransformer에서 (n,1) 형태로 들어올 때를 대비
    try:
        return x.squeeze()
    except Exception:
        return x


def build_preprocess():
    text_pipe_overview = make_pipeline(
        FunctionTransformer(_squeeze_1d, validate=False),
        TfidfVectorizer(max_features=5000),
    )
    text_pipe_title = make_pipeline(
        FunctionTransformer(_squeeze_1d, validate=False),
        TfidfVectorizer(max_features=2000),
    )

    transformers = [
        ("num", "passthrough", NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
        ("overview_tfidf", text_pipe_overview, "overview"),
        ("title_tfidf", text_pipe_title, "title"),
    ]

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
    )