from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


TARGET_COL = "vote_average"


def build_preprocess():
    """
    TMDB 영화 평점 예측용 전처리 파이프라인
    - text   : overview (TF-IDF)
    - numeric: popularity, vote_count, release_year
    - cat    : main_genre
    """

    # Feature groups
    text_features = ["overview"]
    numeric_features = ["popularity", "vote_count", "release_year"]
    categorical_features = ["main_genre"]

    # Transformers
    text_transformer = TfidfVectorizer(
        max_features=300,  # 과적합 방지
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9,
    )

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
    )

    # ColumnTransformer
    preprocess = ColumnTransformer(
        transformers=[
            ("text", text_transformer, "overview"),
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    return preprocess

