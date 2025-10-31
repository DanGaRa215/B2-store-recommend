import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
# 形態素解析器 (Janomeを想定。実際はMeCabなどを使ってもOK)
from janome.tokenizer import Tokenizer



# 形態素解析の初期化
tokenizer = Tokenizer()

# ===============================================
# 共通機能 1: カテゴリのワンホットエンコーディング
# ===============================================

# OneHotEncoderのインスタンス（学習済みモデルを保存しておくために必要）
OHE_MODEL = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

def encode_category(df, column_name='ジャンル', fit_mode=False):
    """カテゴリ列をワンホットエンコーディングする。"""
    categories = df[[column_name]]
    
    if fit_mode:
        # 学習モード（特徴量行列作成時）
        return OHE_MODEL.fit_transform(categories)
    else:
        # 変換モード（ユーザーベクトル作成時）
        return OHE_MODEL.transform(categories)

# ===============================================
# 共通機能 2: 星の数の正規化 (Min-Maxスケーリング)
# ===============================================

# MinMaxScalerのインスタンス
SCALER_MODEL = MinMaxScaler(feature_range=(0, 1))

def scale_rating(df, column_name='平均評価', fit_mode=False):
    """星の数 (1.0~5.0) を0~1に正規化する。"""
    ratings = df[[column_name]]
    
    if fit_mode:
        # 学習モード（特徴量行列作成時）
        return SCALER_MODEL.fit_transform(ratings)
    else:
        # 変換モード（ユーザーベクトル作成時）
        return SCALER_MODEL.transform(ratings)

# ===============================================
# 共通機能 3: レビューのTF-IDFベクトル化の準備
# ===============================================

# TfidfVectorizerのインスタンス（学習済みモデルを保存しておくために必要）
# 実際にはここで形態素解析の前処理を組み込む必要があります
TFIDF_MODEL = TfidfVectorizer(max_features=5000) # 特徴量を5000単語に制限

def preprocess_review(text):
    """レビューテキストを形態素解析し、スペース区切り文字列に変換する前処理（ダミー）"""
    if pd.isna(text): return ""
    # 実際にはここで名詞、形容詞、動詞などを抽出する処理が入る
    tokens = [token.surface for token in tokenizer.tokenize(text) if token.part_of_speech.split(',')[0] in ['名詞', '形容詞']]
    return " ".join(tokens)

def vectorize_review(df, column_name='レビュー本文', fit_mode=False):
    """レビュー列をTF-IDFベクトルに変換する。"""
    # 前処理を実行
    processed_reviews = df[column_name].apply(preprocess_review)
    
    if fit_mode:
        # 学習モード（特徴量行列作成時）
        return TFIDF_MODEL.fit_transform(processed_reviews).toarray()
    else:
        # 変換モード（ユーザーベクトル作成時）
        return TFIDF_MODEL.transform(processed_reviews).toarray()