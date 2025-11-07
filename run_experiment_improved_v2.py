#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改善版v2: 特徴量正規化とバランス調整
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler, normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from janome.tokenizer import Tokenizer
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'

print("="*80)
print("改善版v2: お台場グルメ推薦システム - 特徴量正規化とバランス調整")
print("="*80)

# 初期化
print("\n[1/12] 初期化...")
tokenizer = Tokenizer()
print("✅ 完了")

# データ読み込み
print("\n[2/12] データ読み込み...")
CSV_FILE_PATH = '/home/user/B2-store-recommend/suku/odaiba_reviews_4.csv'
df = pd.read_csv(CSV_FILE_PATH)
print(f"✅ {len(df):,}行")

# データクレンジング
print("\n[3/12] データクレンジング...")
df['star_rating'] = pd.to_numeric(df['star_rating'], errors='coerce')
df.dropna(subset=['star_rating', 'category', 'review_text'], inplace=True)
df['category_list'] = df['category'].apply(lambda x: [c.strip() for c in x.split(',') if c.strip()])
print(f"✅ {len(df):,}行")

# 店舗集約
print("\n[4/12] 店舗集約...")
shop_grouped = df.groupby('shop_name').agg({
    'shop_url': 'first',
    'category': 'first',
    'category_list': 'first',
    'star_rating': 'mean',
    'review_text': lambda x: ' '.join(x.dropna().astype(str))
}).reset_index()
print(f"✅ {len(shop_grouped):,}店舗")

# ストップワード
print("\n[5/12] ストップワード定義...")
STOP_WORDS = set([
    'こと', 'もの', 'よう', 'ため', 'の', 'し', 'ん', 'さん', 'これ', 'それ', 'あれ',
    'この', 'その', 'あの', 'ここ', 'そこ', 'あそこ', '今', '時', '感じ', '的',
    '場合', '時間', '場所', 'お店', '店', '店舗', '利用', '訪問', 'レビュー'
])

def preprocess_review_improved(text):
    if pd.isna(text):
        return ""
    tokens = []
    for token in tokenizer.tokenize(text):
        pos = token.part_of_speech.split(',')[0]
        word = token.surface
        if pos in ['名詞', '形容詞'] and word not in STOP_WORDS and len(word) > 1:
            tokens.append(word)
    return " ".join(tokens)

print("✅ 完了")

# 特徴量エンジニアリング（改善版）
print("\n[6/12] 特徴量エンジニアリング（正規化版）...")
mlb = MultiLabelBinarizer()
scaler = MinMaxScaler()
tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))  # 3000→500に削減

# カテゴリ
category_features_raw = mlb.fit_transform(shop_grouped['category_list'])
# L2正規化
category_features = normalize(category_features_raw, norm='l2')
print(f"   カテゴリ特徴量: {category_features.shape} (L2正規化済み)")

# 星評価
rating_features_raw = scaler.fit_transform(shop_grouped[['star_rating']])
rating_features = normalize(rating_features_raw, norm='l2')
print(f"   星評価特徴量: {rating_features.shape} (L2正規化済み)")

# レビュー
shop_grouped['processed_review'] = shop_grouped['review_text'].apply(preprocess_review_improved)
review_features_raw = tfidf.fit_transform(shop_grouped['processed_review']).toarray()
# L2正規化
review_features = normalize(review_features_raw, norm='l2')
print(f"   レビュー特徴量: {review_features.shape} (L2正規化済み)")
print("✅ 完了")

# パターン別特徴量（重み付き結合）
print("\n[7/12] パターン別特徴量行列作成（重み付き）...")

# 重み設定
CATEGORY_WEIGHT = 2.0  # カテゴリの重要度を上げる
REVIEW_WEIGHT = 1.0
RATING_WEIGHT = 1.0

# P1: カテゴリ + 星の数
features_p1 = np.concatenate([
    category_features * CATEGORY_WEIGHT,
    rating_features * RATING_WEIGHT
], axis=1)
print(f"   P1: {features_p1.shape} (カテゴリ重み×{CATEGORY_WEIGHT})")

# P2: カテゴリ + レビュー
features_p2 = np.concatenate([
    category_features * CATEGORY_WEIGHT,
    review_features * REVIEW_WEIGHT
], axis=1)
print(f"   P2: {features_p2.shape} (カテゴリ重み×{CATEGORY_WEIGHT})")

# P3: カテゴリ + レビュー + 星の数
features_p3 = np.concatenate([
    category_features * CATEGORY_WEIGHT,
    review_features * REVIEW_WEIGHT,
    rating_features * RATING_WEIGHT
], axis=1)
print(f"   P3: {features_p3.shape} (カテゴリ重み×{CATEGORY_WEIGHT})")
print("✅ 完了")

# 推薦関数
def recommend_with_filter(user_vector, feature_matrix, shop_df,
                          category_filter=None, min_rating=None, top_k=5):
    if user_vector.ndim == 1:
        user_vector = user_vector.reshape(1, -1)

    similarities = cosine_similarity(user_vector, feature_matrix)[0]
    mask = np.ones(len(shop_df), dtype=bool)

    if category_filter:
        category_mask = shop_df['category_list'].apply(
            lambda cats: category_filter in cats
        )
        mask = mask & category_mask

    if min_rating is not None:
        rating_mask = shop_df['star_rating'] >= min_rating
        mask = mask & rating_mask

    filtered_indices = np.where(mask)[0]

    if len(filtered_indices) == 0:
        return pd.DataFrame()

    filtered_similarities = similarities[filtered_indices]
    top_local_indices = np.argsort(filtered_similarities)[-top_k:][::-1]
    top_global_indices = filtered_indices[top_local_indices]

    result = shop_df.iloc[top_global_indices].copy()
    result['類似度スコア'] = similarities[top_global_indices]

    return result[['shop_name', 'star_rating', 'category', '類似度スコア']]

# ユーザーベクトル作成（重み付き）
def create_user_vector_p1(category_query, rating_query):
    cat_list = [category_query] if isinstance(category_query, str) else category_query
    cat_vec = normalize(mlb.transform([cat_list]), norm='l2')
    rating_vec = normalize(scaler.transform([[rating_query]]), norm='l2')
    return np.concatenate([cat_vec * CATEGORY_WEIGHT, rating_vec * RATING_WEIGHT], axis=1).flatten()

def create_user_vector_p2(category_query, review_query):
    cat_list = [category_query] if isinstance(category_query, str) else category_query
    cat_vec = normalize(mlb.transform([cat_list]), norm='l2')
    processed = preprocess_review_improved(review_query)
    review_vec = normalize(tfidf.transform([processed]).toarray(), norm='l2')
    return np.concatenate([cat_vec * CATEGORY_WEIGHT, review_vec * REVIEW_WEIGHT], axis=1).flatten()

def create_user_vector_p3(category_query, review_query, rating_query):
    cat_list = [category_query] if isinstance(category_query, str) else category_query
    cat_vec = normalize(mlb.transform([cat_list]), norm='l2')
    processed = preprocess_review_improved(review_query)
    review_vec = normalize(tfidf.transform([processed]).toarray(), norm='l2')
    rating_vec = normalize(scaler.transform([[rating_query]]), norm='l2')
    return np.concatenate([
        cat_vec * CATEGORY_WEIGHT,
        review_vec * REVIEW_WEIGHT,
        rating_vec * RATING_WEIGHT
    ], axis=1).flatten()

print("\n✅ 推薦関数定義完了")

# テストシナリオ
print("\n[8/12] テストシナリオ設定...")
test_scenarios = [
    {
        'name': 'シナリオ1: イタリアン、クリーミーなパスタ',
        'category': 'イタリアン',
        'review': 'クリーミーなパスタが食べたい。チーズたっぷりで濃厚な味わい',
        'rating': 3.5
    },
    {
        'name': 'シナリオ2: 和食、静かで落ち着いた雰囲気',
        'category': '日本料理',
        'review': '静かで落ち着いた雰囲気。上品で繊細な味付け。丁寧な接客',
        'rating': 4.0
    },
    {
        'name': 'シナリオ3: 海鮮、新鮮な刺身',
        'category': '海鮮',
        'review': '新鮮な刺身が食べたい。魚の甘みが感じられる。海の幸',
        'rating': 3.8
    }
]
print("✅ 完了")

# 実験実行
print("\n[9/12] 実験実行...")
results = {}

for scenario in test_scenarios:
    print("\n" + "="*80)
    print(f"🔍 {scenario['name']}")
    print("="*80)

    cat = scenario['category']
    rev = scenario['review']
    rat = scenario['rating']

    print("\n【P1: カテゴリ + 星の数】")
    user_vec_p1 = create_user_vector_p1(cat, rat)
    recs_p1 = recommend_with_filter(user_vec_p1, features_p1, shop_grouped, category_filter=cat, top_k=5)
    print(recs_p1.to_string(index=False))

    print("\n【P2: カテゴリ + レビュー】")
    user_vec_p2 = create_user_vector_p2(cat, rev)
    recs_p2 = recommend_with_filter(user_vec_p2, features_p2, shop_grouped, category_filter=cat, top_k=5)
    print(recs_p2.to_string(index=False))

    print("\n【P3: カテゴリ + レビュー + 星の数】")
    user_vec_p3 = create_user_vector_p3(cat, rev, rat)
    recs_p3 = recommend_with_filter(user_vec_p3, features_p3, shop_grouped, category_filter=cat, top_k=5)
    print(recs_p3.to_string(index=False))

    results[scenario['name']] = {'P1': recs_p1, 'P2': recs_p2, 'P3': recs_p3}

print("\n✅ 完了")

# 評価
print("\n[10/12] 評価指標計算...")
score_stats = []
for scenario_name, patterns in results.items():
    for pattern_name, recs in patterns.items():
        if not recs.empty:
            scores = recs['類似度スコア'].values
            score_stats.append({
                'シナリオ': scenario_name,
                'パターン': pattern_name,
                '平均類似度': scores.mean(),
                '最大類似度': scores.max(),
                '最小類似度': scores.min(),
                '平均評価': recs['star_rating'].mean()
            })

stats_df = pd.DataFrame(score_stats)
print("\n📊 類似度スコア統計")
print(stats_df.to_string(index=False))
print("✅ 完了")

# 貢献度分析
print("\n[11/12] 貢献度分析...")
contribution = []
for scenario_name in results.keys():
    p1_score = results[scenario_name]['P1']['類似度スコア'].mean() if not results[scenario_name]['P1'].empty else 0
    p2_score = results[scenario_name]['P2']['類似度スコア'].mean() if not results[scenario_name]['P2'].empty else 0
    p3_score = results[scenario_name]['P3']['類似度スコア'].mean() if not results[scenario_name]['P3'].empty else 0

    contribution.append({
        'シナリオ': scenario_name,
        'レビュー貢献度 (P2-P1)': p2_score - p1_score,
        '星評価貢献度 (P3-P2)': p3_score - p2_score,
        'P1': p1_score,
        'P2': p2_score,
        'P3': p3_score
    })

contrib_df = pd.DataFrame(contribution)
print("\n📈 特徴量貢献度分析")
print(contrib_df.to_string(index=False))
print("✅ 完了")

# 可視化
print("\n[12/12] 可視化...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, scenario in enumerate(test_scenarios):
    scenario_name = scenario['name']
    if scenario_name in results:
        pattern_data = results[scenario_name]
        patterns = ['P1', 'P2', 'P3']
        avg_scores = [pattern_data[p]['類似度スコア'].mean() if not pattern_data[p].empty else 0 for p in patterns]
        axes[i].bar(patterns, avg_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[i].set_title(f"{scenario['name'].split(':')[1].strip()}\n(Category: {scenario['category']})", fontsize=10)
        axes[i].set_ylabel('Avg Similarity', fontsize=9)
        axes[i].set_ylim(0, 1)
        axes[i].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/B2-store-recommend/pattern_comparison_v2.png', dpi=150, bbox_inches='tight')
print("✅ pattern_comparison_v2.png")
plt.close()

pivot_data = stats_df.pivot_table(index='パターン', columns='シナリオ', values='平均類似度')
fig, ax = plt.subplots(figsize=(12, 6))
pivot_data.T.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax.set_title('Pattern Comparison (Normalized Features)', fontsize=14, fontweight='bold')
ax.set_xlabel('Scenario', fontsize=11)
ax.set_ylabel('Avg Similarity', fontsize=11)
ax.legend(title='Pattern', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('/home/user/B2-store-recommend/scenario_comparison_v2.png', dpi=150, bbox_inches='tight')
print("✅ scenario_comparison_v2.png")
plt.close()

# レポート
summary_path = '/home/user/B2-store-recommend/experiment_summary_v2.txt'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("改善版v2: 特徴量正規化とバランス調整 - 結果サマリー\n")
    f.write("="*80 + "\n\n")

    f.write("## 改善点\n")
    f.write("1. 特徴量のL2正規化を適用\n")
    f.write(f"2. TF-IDF次元数を削減: 3000 → 500\n")
    f.write(f"3. カテゴリの重み付けを強化: ×{CATEGORY_WEIGHT}\n")
    f.write("4. 各特徴量グループを独立に正規化\n\n")

    f.write("## データ概要\n")
    f.write(f"- レビュー総数: {len(df):,}件\n")
    f.write(f"- 店舗数: {len(shop_grouped):,}店舗\n")
    f.write(f"- カテゴリ数: {len(mlb.classes_)}種類\n")
    f.write(f"- TF-IDF語彙数: {len(tfidf.get_feature_names_out())}語\n\n")

    f.write("## パターン定義\n")
    f.write(f"- P1: カテゴリ(×{CATEGORY_WEIGHT}) + 星(×{RATING_WEIGHT}) (次元: {features_p1.shape[1]})\n")
    f.write(f"- P2: カテゴリ(×{CATEGORY_WEIGHT}) + レビュー(×{REVIEW_WEIGHT}) (次元: {features_p2.shape[1]})\n")
    f.write(f"- P3: カテゴリ(×{CATEGORY_WEIGHT}) + レビュー(×{REVIEW_WEIGHT}) + 星(×{RATING_WEIGHT}) (次元: {features_p3.shape[1]})\n\n")

    f.write("## 類似度スコア統計\n")
    f.write(stats_df.to_string(index=False))
    f.write("\n\n")

    f.write("## 特徴量貢献度分析\n")
    f.write(contrib_df.to_string(index=False))
    f.write("\n\n")

    f.write("## 結論\n")
    avg_review_contrib = contrib_df['レビュー貢献度 (P2-P1)'].mean()
    avg_rating_contrib = contrib_df['星評価貢献度 (P3-P2)'].mean()

    f.write(f"- レビュー情報の平均貢献度: {avg_review_contrib:+.4f}\n")
    f.write(f"- 星評価の平均貢献度: {avg_rating_contrib:+.4f}\n")

    if avg_review_contrib > 0:
        f.write("\n✅ レビュー情報の追加により、推薦精度が向上\n")
    else:
        f.write("\n⚠️ レビュー情報による精度向上は限定的\n")

    best_pattern = stats_df.groupby('パターン')['平均類似度'].mean().idxmax()
    f.write(f"\n🏆 最高性能: {best_pattern}\n")

    f.write("\n## 各パターンの平均類似度\n")
    pattern_avg = stats_df.groupby('パターン')['平均類似度'].mean()
    for pattern, score in pattern_avg.items():
        f.write(f"- {pattern}: {score:.4f}\n")

print(f"✅ {summary_path}")

print("\n" + "="*80)
print("🎉 改善版v2 実験完了！")
print("="*80)
print("\n生成ファイル:")
print("  1. pattern_comparison_v2.png")
print("  2. scenario_comparison_v2.png")
print("  3. experiment_summary_v2.txt")
