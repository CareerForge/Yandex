from typing import Literal
from datasets import Dataset, DatasetDict, load_dataset
from collections import defaultdict
import numpy as np
import pandas as pd

from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
from collections import defaultdict


# using this split train-test sets
HOUR_SECONDS = 60 * 60
DAY_SECONDS = 24 * HOUR_SECONDS
GAP_SIZE = HOUR_SECONDS // 2
TEST_SIZE = 1 * DAY_SECONDS
LAST_TIMESTAMP = 26000000
TEST_TIMESTAMP = LAST_TIMESTAMP - TEST_SIZE
TRAIN_TIMESTAMP = TEST_TIMESTAMP - GAP_SIZE

# for evaluations
def recall_at_k(predicted, actual, k):
    if not actual:
        return None
    return len(set(predicted[:k]) & set(actual)) / len(actual)

def dcg_at_k(predicted, actual, k):
    return sum(1 / np.log2(i + 2) for i, item in enumerate(predicted[:k]) if item in actual)

def idcg_at_k(actual, k):
    return sum(1 / np.log2(i + 2) for i in range(min(len(actual), k)))

def ndcg_at_k(predicted, actual, k):
    ideal = idcg_at_k(actual, k)
    if ideal == 0:
        return None
    return dcg_at_k(predicted, actual, k) / ideal

def coverage_at_k(all_predicted, item_catalog, k):
    unique_recs = set(item for recs in all_predicted for item in recs[:k])
    return len(unique_recs) / len(item_catalog)

## POP REC WITH LIKES ============================================================

# pop-recommender using Likes
likes = pd.DataFrame(load_dataset("yandex/yambda", data_dir="flat/50m", data_files="likes.parquet")['train'])
train_df = likes[likes['timestamp'] <= TRAIN_TIMESTAMP]
test_df = likes[likes['timestamp'] > TEST_TIMESTAMP]

# popular items
pop_items = train_df.groupby(['item_id'])['timestamp'].count().reset_index().sort_values(by=['timestamp'], ascending=False)['item_id'].tolist()[:10]

# ground truth
user_to_liked_items = defaultdict(set)
for _, row in test_df.iterrows():
    user_to_liked_items[row["uid"].item()].add(row["item_id"].item())

def evaluate_recommender(user_to_liked_items, predicted, k=10):
    recall_scores, ndcg_scores = [], []

    for user, liked_items in user_to_liked_items.items():
        recall = recall_at_k(predicted, liked_items, k)
        ndcg = ndcg_at_k(predicted, liked_items, k)

        if recall is not None:
            recall_scores.append(recall)
        if ndcg is not None:
            ndcg_scores.append(ndcg)

    return {
        "Recall@K": np.mean(recall_scores),
        "NDCG@K": np.mean(ndcg_scores)
    }

metrics = evaluate_recommender(user_to_liked_items, pop_items, k=10)
print(f"Popularity Recall@10: {metrics['Recall@K']:.4f}")
print(f"Popularity NDCG@10: {metrics['NDCG@K']:.4f}")

# this give performance as
# Popularity Recall@10: 0.0082
# Popularity NDCG@10: 0.0039
# TODO: Please let me know if this difference is because I am not using a validation set, or if I am missing something else

## POP REC WITH Listen+ ============================================================

# pop-recommender using Listen+
listens = pd.DataFrame(load_dataset("yandex/yambda", data_dir="flat/50m", data_files="listens.parquet")['train'])
listens = listens[listens['played_ratio_pct'] >= 50][['uid','item_id','timestamp']]


# popular items
train_df = listens[listens['timestamp'] <= TRAIN_TIMESTAMP]
pop_items = train_df.groupby(['item_id'])['timestamp'].count().reset_index().sort_values(by=['timestamp'], ascending=False)['item_id'].tolist()[:10]
print(pop_items)

# ground truth
test_df = listens[listens['timestamp'] > TEST_TIMESTAMP]

user_to_liked_items = defaultdict(set)
for _, row in test_df.iterrows():
    user_to_liked_items[row["uid"].item()].add(row["item_id"].item())

metrics = evaluate_recommender(user_to_liked_items, pop_items, k=10)
print(f"Popularity Recall@10: {metrics['Recall@K']:.4f}")
print(f"Popularity NDCG@10: {metrics['NDCG@K']:.4f}")


# Results are as follows
# Popularity Recall@10: 0.0070
# Popularity NDCG@10: 0.0174
# TODO: Please let me know if this difference is because I am not using a validation set, or if I am missing something else


## iALS WITH Listen+ ============================================================

# split the data
train_df = listens[listens['timestamp'] <= TRAIN_TIMESTAMP]
train_df = train_df.groupby(['uid','item_id'])['timestamp'].count().reset_index() 
# TODO: I am adding up counts, I see the benchmark code uses ones - but I dont see significant difference even if I use that

# Map user/item IDs to integer indices
user_id_map = {uid: idx for idx, uid in enumerate(train_df["uid"].unique())}
item_id_map = {iid: idx for idx, iid in enumerate(train_df["item_id"].unique())}

train_df["uid_idx"] = train_df["uid"].map(user_id_map)
train_df["item_idx"] = train_df["item_id"].map(item_id_map)
rows = train_df["uid_idx"].values
cols = train_df["item_idx"].values
data = train_df['timestamp'].values

user_item_matrix = coo_matrix((data, (rows, cols)))
user_item_matrix = user_item_matrix.tocsr()

model = AlternatingLeastSquares(factors=128, regularization=0.01, iterations=50, random_state=42)
model.fit(user_item_matrix)

# get user listend items
test_df = listens[listens['timestamp'] > TEST_TIMESTAMP]
# test_df = scores_df[scores_df['timestamp'] > TEST_TIMESTAMP]

test_df = test_df[test_df["uid"].isin(user_id_map)]
test_df = test_df[test_df["item_id"].isin(item_id_map)]

test_df["user_idx"] = test_df["uid"].map(user_id_map)
test_df["item_idx"] = test_df["item_id"].map(item_id_map)

user_to_liked_items = defaultdict(set)
for _, row in test_df.iterrows():
    user_to_liked_items[row["user_idx"]].add(row["item_idx"])


def evaluate_als(model, user_item_matrix, user_to_liked_items, k=10):
    recall_list = []
    ndcg_list = []

    for user_idx, liked_items in user_to_liked_items.items():

        # Get top-K ALS recommendations (filtering known training items)
        recommended = model.recommend(user_idx, user_item_matrix[user_idx], 
                                    N=k, filter_already_liked_items=True)
        rec_items, _  = recommended

        recall = recall_at_k(rec_items, liked_items, k)
        ndcg = ndcg_at_k(rec_items, liked_items, k)

        if recall is not None:
            recall_list.append(recall)
        if ndcg is not None:
            ndcg_list.append(ndcg)

    return {
        "Recall@K": np.mean(recall_list),
        "NDCG@K": np.mean(ndcg_list)
    }

# Evaluate ALS model
results = evaluate_als(model, user_item_matrix, user_to_liked_items, k=10)
print(f"Recall@10: {results['Recall@K']:.4f}")
print(f"NDCG@10: {results['NDCG@K']:.4f}")

# Results are as follows
# Recall@10: 0.0034
# NDCG@10: 0.0077
# TODO: Not sure if the performance difference is because of lack of HP tuning, since I dont have access to GPUs, I prefer this - please let me know if specific HP parameter will help
# Also I am using the CPU version of ALS, but hopefully that should not add too much of a difference.
