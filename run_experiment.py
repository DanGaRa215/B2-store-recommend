#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改善版推薦システム実験実行スクリプト
main_improved.ipynbの内容を実行
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # バックエンド設定（GUIなし環境用）
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from janome.tokenizer import Tokenizer
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'

print("="*80)
print("改善版: お台場グルメ推薦システム - 3パターン比較実験")
print("="*80)

# 形態素解析の初期化
print("\n[1/12] 形態素解析の初期化...")
tokenizer = Tokenizer()
print("✅ 完了")

# データ読み込み
print("\n[2/12] データ読み込み...")
CSV_FILE_PATH = '/home/user/B2-store-recommend/suku/odaiba_reviews_4.csv'
df = pd.read_csv(CSV_FILE_PATH)
print(f"✅ データ読み込み完了: {len(df):,}行")
print(f"   カラム: {df.columns.tolist()}")

# データクレンジング
print("\n[3/12] データクレンジング...")
df['star_rating'] = pd.to_numeric(df['star_rating'], errors='coerce')
df.dropna(subset=['star_rating', 'category', 'review_text'], inplace=True)
df['category_list'] = df['category'].apply(lambda x: [c.strip() for c in x.split(',') if c.strip()])
print(f"✅ クレンジング後: {len(df):,}行")

# 店舗ごとにレビューを集約
print("\n[4/12] 店舗ごとにレビューを集約...")
shop_grouped = df.groupby('shop_name').agg({
    'shop_url': 'first',
    'category': 'first',
    'category_list': 'first',
    'star_rating': 'mean',
    'review_text': lambda x: ' '.join(x.dropna().astype(str))
}).reset_index()
print(f"✅ 集約完了: {len(shop_grouped):,}店舗")

# ストップワード定義
print("\n[5/12] ストップワード定義...")
STOP_WORDS = set([
    'こと', 'もの', 'よう', 'ため', 'の', 'し', 'ん', 'さん', 'これ', 'それ', 'あれ',
    'この', 'その', 'あの', 'ここ', 'そこ', 'あそこ', '今', '時', '感じ', '的',
    '場合', '時間', '場所', 'お店', '店', '店舗', '利用', '訪問', 'レビュー'
])

def preprocess_review_improved(text):
    """改善版レビュー前処理: ストップワード除去 + 名詞・形容詞抽出"""
    if pd.isna(text):
        return ""

    tokens = []
    for token in tokenizer.tokenize(text):
        pos = token.part_of_speech.split(',')[0]
        word = token.surface

        if pos in ['名詞', '形容詞'] and word not in STOP_WORDS and len(word) > 1:
            tokens.append(word)

    return " ".join(tokens)

print("✅ ストップワード定義完了")

# 特徴量エンジニアリング
print("\n[6/12] 特徴量エンジニアリング...")
mlb = MultiLabelBinarizer()
scaler = MinMaxScaler()
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))

category_features = mlb.fit_transform(shop_grouped['category_list'])
print(f"   カテゴリ特徴量: {category_features.shape}")

rating_features = scaler.fit_transform(shop_grouped[['star_rating']])
print(f"   星評価特徴量: {rating_features.shape}")

shop_grouped['processed_review'] = shop_grouped['review_text'].apply(preprocess_review_improved)
review_features = tfidf.fit_transform(shop_grouped['processed_review']).toarray()
print(f"   レビュー特徴量: {review_features.shape}")
print("✅ 特徴量エンジニアリング完了")

# パターン別特徴量行列作成
print("\n[7/12] パターン別特徴量行列作成...")
features_p1 = np.concatenate([category_features, rating_features], axis=1)
features_p2 = np.concatenate([category_features, review_features], axis=1)
features_p3 = np.concatenate([category_features, review_features, rating_features], axis=1)
print(f"   P1 (カテゴリ+星): {features_p1.shape}")
print(f"   P2 (カテゴリ+レビュー): {features_p2.shape}")
print(f"   P3 (カテゴリ+レビュー+星): {features_p3.shape}")
print("✅ 3パターンの特徴量行列作成完了")

# 推薦関数定義
def recommend_with_filter(user_vector, feature_matrix, shop_df,
                          category_filter=None, min_rating=None, top_k=5):
    """カテゴリフィルタリング対応の推薦関数"""
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

# ユーザーベクトル作成関数
def create_user_vector_p1(category_query, rating_query):
    """P1: カテゴリ + 星の数"""
    cat_list = [category_query] if isinstance(category_query, str) else category_query
    cat_vec = mlb.transform([cat_list])
    rating_vec = scaler.transform([[rating_query]])
    return np.concatenate([cat_vec, rating_vec], axis=1).flatten()

def create_user_vector_p2(category_query, review_query):
    """P2: カテゴリ + レビュー"""
    cat_list = [category_query] if isinstance(category_query, str) else category_query
    cat_vec = mlb.transform([cat_list])
    processed = preprocess_review_improved(review_query)
    review_vec = tfidf.transform([processed]).toarray()
    return np.concatenate([cat_vec, review_vec], axis=1).flatten()

def create_user_vector_p3(category_query, review_query, rating_query):
    """P3: カテゴリ + レビュー + 星の数"""
    cat_list = [category_query] if isinstance(category_query, str) else category_query
    cat_vec = mlb.transform([cat_list])
    processed = preprocess_review_improved(review_query)
    review_vec = tfidf.transform([processed]).toarray()
    rating_vec = scaler.transform([[rating_query]])
    return np.concatenate([cat_vec, review_vec, rating_vec], axis=1).flatten()

print("\n✅ 推薦関数定義完了")

# テストシナリオ設定
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
print("✅ テストシナリオ設定完了")

# 実験実行
print("\n[9/12] 実験実行: 3パターン比較...")
results = {}

for scenario in test_scenarios:
    print("\n" + "="*80)
    print(f"🔍 {scenario['name']}")
    print("="*80)

    cat = scenario['category']
    rev = scenario['review']
    rat = scenario['rating']

    # P1
    print("\n【P1: カテゴリ + 星の数】")
    user_vec_p1 = create_user_vector_p1(cat, rat)
    recs_p1 = recommend_with_filter(
        user_vec_p1, features_p1, shop_grouped,
        category_filter=cat, min_rating=None, top_k=5
    )
    print(recs_p1.to_string(index=False))

    # P2
    print("\n【P2: カテゴリ + レビュー】")
    user_vec_p2 = create_user_vector_p2(cat, rev)
    recs_p2 = recommend_with_filter(
        user_vec_p2, features_p2, shop_grouped,
        category_filter=cat, min_rating=None, top_k=5
    )
    print(recs_p2.to_string(index=False))

    # P3
    print("\n【P3: カテゴリ + レビュー + 星の数】")
    user_vec_p3 = create_user_vector_p3(cat, rev, rat)
    recs_p3 = recommend_with_filter(
        user_vec_p3, features_p3, shop_grouped,
        category_filter=cat, min_rating=None, top_k=5
    )
    print(recs_p3.to_string(index=False))

    results[scenario['name']] = {
        'P1': recs_p1,
        'P2': recs_p2,
        'P3': recs_p3
    }

print("\n✅ 全シナリオの実験完了")

# 評価指標計算
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
print("✅ 評価指標計算完了")

# 貢献度分析
print("\n[11/12] 特徴量貢献度分析...")
contribution = []

for scenario_name in results.keys():
    p1_score = results[scenario_name]['P1']['類似度スコア'].mean() if not results[scenario_name]['P1'].empty else 0
    p2_score = results[scenario_name]['P2']['類似度スコア'].mean() if not results[scenario_name]['P2'].empty else 0
    p3_score = results[scenario_name]['P3']['類似度スコア'].mean() if not results[scenario_name]['P3'].empty else 0

    review_contribution = p2_score - p1_score
    rating_contribution = p3_score - p2_score

    contribution.append({
        'シナリオ': scenario_name,
        'レビュー貢献度 (P2-P1)': review_contribution,
        '星評価貢献度 (P3-P2)': rating_contribution,
        'P1平均類似度': p1_score,
        'P2平均類似度': p2_score,
        'P3平均類似度': p3_score
    })

contrib_df = pd.DataFrame(contribution)
print("\n📈 特徴量貢献度分析")
print(contrib_df.to_string(index=False))
print("✅ 貢献度分析完了")

# 可視化
print("\n[12/12] 可視化とレポート生成...")

# グラフ1: パターン別類似度比較
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, scenario in enumerate(test_scenarios):
    scenario_name = scenario['name']

    if scenario_name in results:
        pattern_data = results[scenario_name]

        patterns = ['P1', 'P2', 'P3']
        avg_scores = [
            pattern_data[p]['類似度スコア'].mean() if not pattern_data[p].empty else 0
            for p in patterns
        ]

        axes[i].bar(patterns, avg_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        axes[i].set_title(f"{scenario['name'].split(':')[1].strip()}\n(Category: {scenario['category']})",
                         fontsize=10)
        axes[i].set_ylabel('Average Similarity Score', fontsize=9)
        axes[i].set_ylim(0, 1)
        axes[i].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/user/B2-store-recommend/pattern_comparison.png', dpi=150, bbox_inches='tight')
print("✅ グラフ保存: pattern_comparison.png")
plt.close()

# グラフ2: シナリオ別パターン比較
pivot_data = stats_df.pivot_table(
    index='パターン',
    columns='シナリオ',
    values='平均類似度'
)

fig, ax = plt.subplots(figsize=(12, 6))
pivot_data.T.plot(kind='bar', ax=ax, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax.set_title('Pattern Comparison Across Scenarios', fontsize=14, fontweight='bold')
ax.set_xlabel('Scenario', fontsize=11)
ax.set_ylabel('Average Similarity Score', fontsize=11)
ax.legend(title='Pattern', fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('/home/user/B2-store-recommend/scenario_comparison.png', dpi=150, bbox_inches='tight')
print("✅ グラフ保存: scenario_comparison.png")
plt.close()

# サマリーレポート
summary_path = '/home/user/B2-store-recommend/experiment_summary.txt'

with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("お台場グルメ推薦システム - 3パターン比較実験 結果サマリー\n")
    f.write("="*80 + "\n\n")

    f.write("## データ概要\n")
    f.write(f"- レビュー総数: {len(df):,}件\n")
    f.write(f"- 店舗数: {len(shop_grouped):,}店舗\n")
    f.write(f"- カテゴリ数: {len(mlb.classes_)}種類\n")
    f.write(f"- TF-IDF語彙数: {len(tfidf.get_feature_names_out())}語\n\n")

    f.write("## パターン定義\n")
    f.write(f"- P1: カテゴリ + 星の数 (特徴量次元: {features_p1.shape[1]})\n")
    f.write(f"- P2: カテゴリ + レビュー (特徴量次元: {features_p2.shape[1]})\n")
    f.write(f"- P3: カテゴリ + レビュー + 星の数 (特徴量次元: {features_p3.shape[1]})\n\n")

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
        f.write("\n✅ レビュー情報の追加により、推薦精度が向上することを確認\n")

    best_pattern = stats_df.groupby('パターン')['平均類似度'].mean().idxmax()
    f.write(f"\n🏆 最高性能パターン: {best_pattern}\n")

    f.write("\n## 各パターンの平均類似度\n")
    pattern_avg = stats_df.groupby('パターン')['平均類似度'].mean()
    for pattern, score in pattern_avg.items():
        f.write(f"- {pattern}: {score:.4f}\n")

print(f"✅ 結果サマリー保存: {summary_path}")

# 完了メッセージ
print("\n" + "="*80)
print("🎉 実験完了！")
print("="*80)
print("\n生成ファイル:")
print("  1. pattern_comparison.png - パターン別類似度比較")
print("  2. scenario_comparison.png - シナリオ別比較")
print("  3. experiment_summary.txt - 実験結果サマリー")
print("\n次のステップ:")
print("  - 全国データへの拡張")
print("  - より多様なテストシナリオでの評価")
print("  - ハイパーパラメータチューニング")
