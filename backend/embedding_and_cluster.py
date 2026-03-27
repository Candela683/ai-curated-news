import re
import json
import sqlite3
import hashlib
import unicodedata
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import hdbscan

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# 配置
# =========================
GDELT_DB_PATH = "gdelt_rss.sqlite"
GDELT_TABLE_NAME = "gdelt_rss"
CLUSTER_DB_PATH = "cluster.sqlite"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

ASSIGN_THRESHOLD = 0.80          # 单条标题 -> 旧簇匹配阈值
SECOND_ALIGN_THRESHOLD = 0.84    # candidate 新簇 -> 旧簇 二次对齐阈值
MIN_CLUSTER_SIZE = 5             # HDBSCAN 最小簇
MIN_SAMPLES = 3
REP_TOPN = 3
BATCH_SIZE = 64
WINDOW_HOURS = 24

# 二次分裂配置
ENABLE_SECOND_SPLIT = True
SPLIT_TRIGGER_NEW_COUNT = 4          # 本轮新增超过这个数量才考虑二次分裂
SPLIT_MIN_CLUSTER_SIZE = 5           # 子簇最小样本
SPLIT_MIN_TOTAL_SIZE = 10            # 父簇至少这么大才尝试分裂
SPLIT_SECOND_ALIGN_THRESHOLD = 0.84  # 子簇与父簇/其他旧簇的对齐阈值


# =========================
# 基础工具
# =========================
def normalize_title(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", " ", text)
    return text


def detect_url_column(df: pd.DataFrame) -> str:
    candidates = [
        "url",
        "link",
        "sourceurl",
        "source_url",
        "documentidentifier",
        "document_identifier",
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for col in candidates:
        if col in lower_map:
            return lower_map[col]
    raise ValueError(
        f"未找到网址列。当前字段有: {df.columns.tolist()}，请手动指定 URL 列名。"
    )


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def stable_source_pk(url: str, title_clean: str, published_at: str) -> str:
    base = f"{url or ''}|{title_clean or ''}|{(published_at or '')[:16]}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def to_json_array(vec: np.ndarray) -> str:
    return json.dumps([float(x) for x in vec.tolist()], ensure_ascii=False)


def from_json_array(s: str) -> np.ndarray:
    arr = np.array(json.loads(s), dtype=np.float32)
    return l2_normalize(arr)


def now_utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# =========================
# cluster.sqlite 初始化
# =========================
def init_cluster_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS clusters (
            cluster_id INTEGER PRIMARY KEY AUTOINCREMENT,
            status TEXT NOT NULL DEFAULT 'active',
            parent_cluster_id INTEGER,
            canonical_title TEXT,
            centroid_json TEXT NOT NULL,
            example_titles_json TEXT,
            size_current INTEGER NOT NULL DEFAULT 0,
            summarize_count INTEGER NOT NULL DEFAULT 0,
            summarize_result TEXT NOT NULL DEFAULT '',
            summarize_title_count INTEGER,
            summarize_batch_at TEXT,
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cluster_items (
            item_id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_table TEXT NOT NULL,
            source_pk TEXT NOT NULL,
            url TEXT,
            title TEXT NOT NULL,
            title_clean TEXT NOT NULL,
            published_at TEXT,
            updated_at TEXT,
            cluster_id INTEGER NOT NULL,
            assigned_at TEXT NOT NULL,
            UNIQUE(source_table, source_pk)
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cluster_size_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_at TEXT NOT NULL,
            run_date TEXT NOT NULL,
            cluster_id INTEGER NOT NULL,
            title_count INTEGER NOT NULL,
            new_title_count INTEGER NOT NULL DEFAULT 0,
            example_titles_json TEXT
        )
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cluster_items_cluster_id
        ON cluster_items(cluster_id)
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cluster_items_published_at
        ON cluster_items(published_at)
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cluster_size_history_cluster_id
        ON cluster_size_history(cluster_id, snapshot_at)
        """
    )
    conn.execute(
    """
    CREATE INDEX IF NOT EXISTS idx_clusters_summarize_batch_at
    ON clusters(summarize_batch_at)
    """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_clusters_last_seen_at
        ON clusters(last_seen_at)
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_clusters_first_seen_at
        ON clusters(first_seen_at)
        """
    )
    conn.execute(
    """
    CREATE INDEX IF NOT EXISTS idx_clusters_cluster_id
    ON clusters(cluster_id)
    """
    )

    ensure_cluster_schema(conn)
 
    conn.commit()



def ensure_cluster_schema(conn: sqlite3.Connection) -> None:
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(clusters)").fetchall()}
    alter_sqls = []
    if 'parent_cluster_id' not in existing_cols:
        alter_sqls.append("ALTER TABLE clusters ADD COLUMN parent_cluster_id INTEGER")
    if 'summarize_count' not in existing_cols:
        alter_sqls.append("ALTER TABLE clusters ADD COLUMN summarize_count INTEGER NOT NULL DEFAULT 0")
    if 'summarize_result' not in existing_cols:
        alter_sqls.append("ALTER TABLE clusters ADD COLUMN summarize_result TEXT NOT NULL DEFAULT ''")
    if 'summarize_title_count' not in existing_cols:
        alter_sqls.append("ALTER TABLE clusters ADD COLUMN summarize_title_count INTEGER")
    if 'summarize_batch_at' not in existing_cols:
        alter_sqls.append("ALTER TABLE clusters ADD COLUMN summarize_batch_at TEXT")
    for sql in alter_sqls:
        conn.execute(sql)


# =========================
# 读取最近 N 小时数据
# =========================
def load_recent_data(db_path: str, table_name: str, start_utc: str, end_utc: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(
            f"""
            SELECT *
            FROM {table_name}
            WHERE published_at >= ? AND published_at <= ?
            ORDER BY published_at DESC, updated_at DESC
            """,
            conn,
            params=(start_utc, end_utc),
        )
    finally:
        conn.close()
    return df



def prepare_recent_df(df: pd.DataFrame, source_table: str) -> Tuple[pd.DataFrame, str]:
    if df.empty:
        return df.copy(), ""

    df = df.copy()
    df["title_clean"] = df["title"].fillna("").map(normalize_title)
    df = df[df["title_clean"].str.len() > 0].reset_index(drop=True)
    if df.empty:
        return df, ""

    url_col = detect_url_column(df)
    df[url_col] = df[url_col].fillna("")
    df["published_at"] = df.get("published_at", "").fillna("")
    if "updated_at" not in df.columns:
        df["updated_at"] = ""
    else:
        df["updated_at"] = df["updated_at"].fillna("")

    df["source_table"] = source_table
    df["source_pk"] = df.apply(
        lambda r: stable_source_pk(
            url=str(r[url_col]),
            title_clean=str(r["title_clean"]),
            published_at=str(r["published_at"]),
        ),
        axis=1,
    )

    df = df.drop_duplicates(subset=["source_pk"]).reset_index(drop=True)
    return df, url_col


# =========================
# 增量：过滤已入库 item
# =========================
def filter_new_items(cluster_conn: sqlite3.Connection, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    pks = df["source_pk"].tolist()
    existing = set()
    chunk_size = 900
    for i in range(0, len(pks), chunk_size):
        batch = pks[i:i + chunk_size]
        placeholders = ",".join(["?"] * len(batch))
        rows = cluster_conn.execute(
            f"SELECT source_pk FROM cluster_items WHERE source_pk IN ({placeholders})",
            batch,
        ).fetchall()
        existing.update(r[0] for r in rows)

    return df[~df["source_pk"].isin(existing)].reset_index(drop=True)


# =========================
# 历史簇读取
# =========================
def load_active_clusters(cluster_conn: sqlite3.Connection) -> List[Dict]:
    rows = cluster_conn.execute(
        """
        SELECT cluster_id, parent_cluster_id, canonical_title, centroid_json, example_titles_json,
               size_current, summarize_count, summarize_result, summarize_title_count,
               first_seen_at, last_seen_at
        FROM clusters
        WHERE status = 'active'
        ORDER BY cluster_id
        """
    ).fetchall()

    out: List[Dict] = []
    for row in rows:
        out.append(
            {
                "cluster_id": int(row[0]),
                "parent_cluster_id": row[1],
                "canonical_title": row[2] or "",
                "centroid": from_json_array(row[3]),
                "example_titles_json": row[4] or "[]",
                "size_current": int(row[5]),
                "summarize_count": int(row[6]),
                "summarize_result": row[7] or "",
                "summarize_title_count": row[8],
                "first_seen_at": row[9],
                "last_seen_at": row[10],
            }
        )
    return out



def load_active_cluster_map(cluster_conn: sqlite3.Connection) -> Dict[int, Dict]:
    return {c["cluster_id"]: c for c in load_active_clusters(cluster_conn)}


# =========================
# 代表标题
# =========================
def pick_representative_indices(embeddings: np.ndarray, labels: np.ndarray, probs: np.ndarray, topn: int = 3) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for c in sorted(set(labels.tolist())):
        if c == -1:
            continue
        idx = np.where(labels == c)[0]
        sub_emb = embeddings[idx]
        centroid = l2_normalize(sub_emb.mean(axis=0)).reshape(1, -1)
        sims = cosine_similarity(sub_emb, centroid).ravel()
        score = 0.7 * sims + 0.3 * probs[idx]
        top_idx = idx[np.argsort(-score)[:topn]]
        out[int(c)] = top_idx.tolist()
    return out



def top_titles_from_df(df: pd.DataFrame, n: int = 3) -> List[str]:
    return df["title"].fillna("").head(n).tolist()


# =========================
# 旧簇匹配
# =========================
def assign_to_existing_clusters(
    item_embeddings: np.ndarray,
    active_clusters: List[Dict],
    threshold: float,
) -> Tuple[List[Optional[int]], List[float]]:
    n = item_embeddings.shape[0]
    assigned_cluster_ids: List[Optional[int]] = [None] * n
    assigned_scores: List[float] = [float("nan")] * n

    if n == 0 or not active_clusters:
        return assigned_cluster_ids, assigned_scores

    centroid_matrix = np.vstack([c["centroid"] for c in active_clusters])
    sims = cosine_similarity(item_embeddings, centroid_matrix)

    best_idx = np.argmax(sims, axis=1)
    best_score = np.max(sims, axis=1)

    for i in range(n):
        if float(best_score[i]) >= threshold:
            assigned_cluster_ids[i] = int(active_clusters[int(best_idx[i])]["cluster_id"])
            assigned_scores[i] = float(best_score[i])

    return assigned_cluster_ids, assigned_scores


# =========================
# centroid 加权更新
# =========================
def weighted_update_centroid(old_centroid: np.ndarray, old_size: int, new_embeddings: np.ndarray) -> np.ndarray:
    if new_embeddings.size == 0:
        return l2_normalize(old_centroid)

    new_sum = new_embeddings.sum(axis=0)
    updated = old_centroid * float(old_size) + new_sum
    return l2_normalize(updated)


# =========================
# DB 写入
# =========================
def insert_cluster_item_rows(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    url_col: str,
    assigned_at: str,
) -> int:
    if df.empty:
        return 0

    before = conn.total_changes
    rows = [
        (
            str(r["source_table"]),
            str(r["source_pk"]),
            str(r[url_col]) if url_col in df.columns else "",
            str(r["title"]),
            str(r["title_clean"]),
            str(r.get("published_at", "") or ""),
            str(r.get("updated_at", "") or ""),
            int(r["cluster_id"]),
            assigned_at,
        )
        for _, r in df.iterrows()
    ]

    conn.executemany(
        """
        INSERT OR IGNORE INTO cluster_items (
            source_table, source_pk, url, title, title_clean,
            published_at, updated_at, cluster_id, assigned_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return conn.total_changes - before



def update_one_cluster(
    conn: sqlite3.Connection,
    cluster_id: int,
    old_centroid: np.ndarray,
    old_size: int,
    new_embeddings: np.ndarray,
    canonical_title: str,
    example_titles: List[str],
    now_utc: str,
) -> Tuple[np.ndarray, int]:
    new_centroid = weighted_update_centroid(old_centroid, old_size, new_embeddings)
    new_size = old_size + int(new_embeddings.shape[0])
    conn.execute(
        """
        UPDATE clusters
        SET centroid_json = ?,
            canonical_title = ?,
            example_titles_json = ?,
            size_current = ?,
            last_seen_at = ?
        WHERE cluster_id = ?
        """,
        (
            to_json_array(new_centroid),
            canonical_title,
            json.dumps(example_titles, ensure_ascii=False),
            new_size,
            now_utc,
            cluster_id,
        ),
    )
    return new_centroid, new_size



def recompute_and_update_cluster_profile(
    conn: sqlite3.Connection,
    cluster_id: int,
    df_cluster: pd.DataFrame,
    embeddings: np.ndarray,
    now_utc: str,
) -> None:
    centroid = l2_normalize(embeddings.mean(axis=0))
    probs = np.ones(len(df_cluster), dtype=float)
    rep_idx_map = pick_representative_indices(
        embeddings=embeddings,
        labels=np.zeros(len(df_cluster), dtype=int),
        probs=probs,
        topn=min(REP_TOPN, len(df_cluster)),
    )
    rep_idx = rep_idx_map.get(0, list(range(min(REP_TOPN, len(df_cluster)))))
    rep_titles = df_cluster.iloc[rep_idx]["title"].tolist() if len(df_cluster) else []
    canonical_title = rep_titles[0] if rep_titles else (str(df_cluster.iloc[0]["title"]) if len(df_cluster) else "")
    conn.execute(
        """
        UPDATE clusters
        SET centroid_json = ?,
            canonical_title = ?,
            example_titles_json = ?,
            size_current = ?,
            last_seen_at = ?
        WHERE cluster_id = ?
        """,
        (
            to_json_array(centroid),
            canonical_title,
            json.dumps(rep_titles, ensure_ascii=False),
            int(len(df_cluster)),
            now_utc,
            cluster_id,
        ),
    )



def update_existing_clusters(
    conn: sqlite3.Connection,
    assigned_df: pd.DataFrame,
    all_embeddings: np.ndarray,
    now_utc: str,
) -> Dict[int, int]:
    if assigned_df.empty:
        return {}

    new_count_by_cluster: Dict[int, int] = {}
    cluster_ids = assigned_df["cluster_id"].astype(int).unique().tolist()
    for cid in cluster_ids:
        sub = assigned_df[assigned_df["cluster_id"] == cid]
        emb = all_embeddings[sub.index.to_numpy()]

        row = conn.execute(
            "SELECT centroid_json, size_current FROM clusters WHERE cluster_id = ?",
            (cid,),
        ).fetchone()
        if row is None:
            continue

        old_centroid = from_json_array(row[0])
        old_size = int(row[1])
        canonical_title = str(sub.iloc[0]["title"])
        example_titles = sub["title"].head(REP_TOPN).tolist()

        update_one_cluster(
            conn=conn,
            cluster_id=cid,
            old_centroid=old_centroid,
            old_size=old_size,
            new_embeddings=emb,
            canonical_title=canonical_title,
            example_titles=example_titles,
            now_utc=now_utc,
        )
        new_count_by_cluster[cid] = len(sub)

    return new_count_by_cluster



def create_new_cluster(
    conn: sqlite3.Connection,
    centroid: np.ndarray,
    canonical_title: str,
    example_titles: List[str],
    size_current: int,
    now_utc: str,
    parent_cluster_id: Optional[int] = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO clusters (
            status, parent_cluster_id, canonical_title, centroid_json, example_titles_json,
            size_current, summarize_count, summarize_result, summarize_title_count,
            first_seen_at, last_seen_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "active",
            parent_cluster_id,
            canonical_title,
            to_json_array(centroid),
            json.dumps(example_titles, ensure_ascii=False),
            int(size_current),
            0,
            "",
            None,
            now_utc,
            now_utc,
        ),
    )
    return int(cur.lastrowid)


# =========================
# unmatched -> HDBSCAN
# =========================
def cluster_unmatched(unmatched_df: pd.DataFrame, unmatched_embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if unmatched_df.empty:
        return np.array([], dtype=int), np.array([], dtype=float)

    n = len(unmatched_df)
    if n < MIN_CLUSTER_SIZE:
        return np.full(n, -1, dtype=int), np.zeros(n, dtype=float)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(unmatched_embeddings)
    probs = getattr(clusterer, "probabilities_", np.zeros(n, dtype=float))
    return labels.astype(int), probs.astype(float)


# =========================
# 二次对齐 + 新簇入库
# =========================
def second_align_candidate_cluster(
    candidate_centroid: np.ndarray,
    active_cluster_map: Dict[int, Dict],
    threshold: float,
    exclude_cluster_ids: Optional[Set[int]] = None,
) -> Tuple[Optional[int], float]:
    if not active_cluster_map:
        return None, float("nan")

    exclude_cluster_ids = exclude_cluster_ids or set()
    cluster_ids = [cid for cid in active_cluster_map.keys() if cid not in exclude_cluster_ids]
    if not cluster_ids:
        return None, float("nan")

    centroid_matrix = np.vstack([active_cluster_map[cid]["centroid"] for cid in cluster_ids])
    sims = cosine_similarity(candidate_centroid.reshape(1, -1), centroid_matrix).ravel()
    best_pos = int(np.argmax(sims))
    best_score = float(sims[best_pos])
    if best_score >= threshold:
        return int(cluster_ids[best_pos]), best_score
    return None, best_score



def persist_clusters_from_unmatched_with_second_align(
    conn: sqlite3.Connection,
    unmatched_df: pd.DataFrame,
    unmatched_embeddings: np.ndarray,
    url_col: str,
    now_utc: str,
    second_align_threshold: float,
) -> Tuple[int, int, int, Dict[int, int]]:
    if unmatched_df.empty:
        return 0, 0, 0, {}

    labels, probs = cluster_unmatched(unmatched_df, unmatched_embeddings)
    unmatched_df = unmatched_df.copy()
    unmatched_df["cluster_tmp"] = labels
    unmatched_df["cluster_prob"] = probs

    created_cluster_count = 0
    aligned_to_old_cluster_count = 0
    inserted_item_count = 0
    new_count_by_cluster: Dict[int, int] = {}

    valid_labels = sorted([c for c in set(labels.tolist()) if c != -1])
    if not valid_labels:
        return 0, 0, 0, {}

    rep = pick_representative_indices(
        embeddings=unmatched_embeddings,
        labels=labels,
        probs=probs,
        topn=REP_TOPN,
    )

    active_cluster_map = load_active_cluster_map(conn)

    for c in valid_labels:
        pos_idx = np.where(labels == c)[0]
        sub = unmatched_df.iloc[pos_idx].copy()
        emb = unmatched_embeddings[pos_idx]
        centroid = l2_normalize(emb.mean(axis=0))

        rep_titles = unmatched_df.iloc[rep[int(c)]]["title"].tolist() if int(c) in rep else top_titles_from_df(sub, REP_TOPN)
        canonical_title = rep_titles[0] if rep_titles else str(sub.iloc[0]["title"])

        matched_old_cluster_id, _ = second_align_candidate_cluster(
            candidate_centroid=centroid,
            active_cluster_map=active_cluster_map,
            threshold=second_align_threshold,
        )

        if matched_old_cluster_id is not None:
            sub["cluster_id"] = matched_old_cluster_id
            inserted_item_count += insert_cluster_item_rows(conn, sub, url_col=url_col, assigned_at=now_utc)

            old_info = active_cluster_map[matched_old_cluster_id]
            new_centroid, new_size = update_one_cluster(
                conn=conn,
                cluster_id=matched_old_cluster_id,
                old_centroid=old_info["centroid"],
                old_size=int(old_info["size_current"]),
                new_embeddings=emb,
                canonical_title=canonical_title,
                example_titles=rep_titles,
                now_utc=now_utc,
            )
            old_info["centroid"] = new_centroid
            old_info["size_current"] = new_size
            old_info["canonical_title"] = canonical_title
            aligned_to_old_cluster_count += 1
            new_count_by_cluster[matched_old_cluster_id] = new_count_by_cluster.get(matched_old_cluster_id, 0) + len(sub)
        else:
            new_cluster_id = create_new_cluster(
                conn=conn,
                centroid=centroid,
                canonical_title=canonical_title,
                example_titles=rep_titles,
                size_current=len(sub),
                now_utc=now_utc,
            )
            created_cluster_count += 1
            sub["cluster_id"] = new_cluster_id
            inserted_item_count += insert_cluster_item_rows(conn, sub, url_col=url_col, assigned_at=now_utc)
            new_count_by_cluster[new_cluster_id] = new_count_by_cluster.get(new_cluster_id, 0) + len(sub)
            active_cluster_map[new_cluster_id] = {
                "cluster_id": new_cluster_id,
                "parent_cluster_id": None,
                "canonical_title": canonical_title,
                "centroid": centroid,
                "example_titles_json": json.dumps(rep_titles, ensure_ascii=False),
                "size_current": len(sub),
                "summarize_count": 0,
                "summarize_result": "",
                "summarize_title_count": None,
                "first_seen_at": now_utc,
                "last_seen_at": now_utc,
            }

    return created_cluster_count, aligned_to_old_cluster_count, inserted_item_count, new_count_by_cluster


# =========================
# 簇内二次分裂
# =========================
def load_cluster_items_df(conn: sqlite3.Connection, cluster_id: int) -> pd.DataFrame:
    rows = conn.execute(
        """
        SELECT item_id, source_table, source_pk, url, title, title_clean, published_at, updated_at, cluster_id, assigned_at
        FROM cluster_items
        WHERE cluster_id = ?
        ORDER BY published_at DESC, assigned_at DESC, item_id DESC
        """,
        (cluster_id,),
    ).fetchall()
    cols = ["item_id", "source_table", "source_pk", "url", "title", "title_clean", "published_at", "updated_at", "cluster_id", "assigned_at"]
    return pd.DataFrame(rows, columns=cols)



def cluster_for_split(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = embeddings.shape[0]
    if n < max(SPLIT_MIN_TOTAL_SIZE, SPLIT_MIN_CLUSTER_SIZE * 2):
        return np.full(n, -1, dtype=int), np.zeros(n, dtype=float)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=SPLIT_MIN_CLUSTER_SIZE,
        min_samples=MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(embeddings)
    probs = getattr(clusterer, "probabilities_", np.zeros(n, dtype=float))
    return labels.astype(int), probs.astype(float)



def apply_second_split(
    conn: sqlite3.Connection,
    model: SentenceTransformer,
    touched_cluster_new_counts: Dict[int, int],
    now_utc: str,
) -> Tuple[int, Dict[int, int]]:
    if not ENABLE_SECOND_SPLIT or not touched_cluster_new_counts:
        return 0, {}

    split_created_clusters = 0
    final_new_count_updates: Dict[int, int] = {}
    active_cluster_map = load_active_cluster_map(conn)

    candidate_cluster_ids = [
        cid for cid, new_cnt in touched_cluster_new_counts.items()
        if new_cnt >= SPLIT_TRIGGER_NEW_COUNT
    ]
    if not candidate_cluster_ids:
        return 0, {}

    for parent_cluster_id in candidate_cluster_ids:
        parent_info = active_cluster_map.get(parent_cluster_id)
        if not parent_info:
            continue
        if int(parent_info["size_current"]) < SPLIT_MIN_TOTAL_SIZE:
            continue

        df_cluster = load_cluster_items_df(conn, parent_cluster_id)
        if df_cluster.empty or len(df_cluster) < max(SPLIT_MIN_TOTAL_SIZE, SPLIT_MIN_CLUSTER_SIZE * 2):
            continue

        embeddings = model.encode(
            df_cluster["title_clean"].tolist(),
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        labels, probs = cluster_for_split(embeddings)
        valid_labels = [c for c in sorted(set(labels.tolist())) if c != -1]
        if len(valid_labels) < 2:
            continue

        label_sizes = {c: int(np.sum(labels == c)) for c in valid_labels}
        stable_labels = [c for c in valid_labels if label_sizes[c] >= SPLIT_MIN_CLUSTER_SIZE]
        if len(stable_labels) < 2:
            continue

        parent_centroid = parent_info["centroid"]
        child_meta = []
        for lab in stable_labels:
            pos_idx = np.where(labels == lab)[0]
            child_emb = embeddings[pos_idx]
            child_centroid = l2_normalize(child_emb.mean(axis=0))
            sim_to_parent = float(cosine_similarity(child_centroid.reshape(1, -1), parent_centroid.reshape(1, -1)).ravel()[0])
            child_meta.append((lab, len(pos_idx), sim_to_parent, pos_idx, child_centroid))

        # 主子簇：优先保留与父簇最接近且体量最大的子簇
        child_meta.sort(key=lambda x: (x[2], x[1]), reverse=True)
        keep_label, keep_size, _, keep_pos_idx, keep_centroid = child_meta[0]

        # 更新父簇明细和画像
        keep_df = df_cluster.iloc[keep_pos_idx].copy()
        recompute_and_update_cluster_profile(
            conn=conn,
            cluster_id=parent_cluster_id,
            df_cluster=keep_df,
            embeddings=embeddings[keep_pos_idx],
            now_utc=now_utc,
        )

        created_any = False
        for lab, size, _, pos_idx, child_centroid in child_meta[1:]:
            sub_df = df_cluster.iloc[pos_idx].copy()
            rep = pick_representative_indices(
                embeddings=embeddings[pos_idx],
                labels=np.zeros(len(pos_idx), dtype=int),
                probs=np.ones(len(pos_idx), dtype=float),
                topn=min(REP_TOPN, len(pos_idx)),
            )
            rep_idx = rep.get(0, list(range(min(REP_TOPN, len(sub_df)))))
            rep_titles = sub_df.iloc[rep_idx]["title"].tolist() if len(sub_df) else []
            canonical_title = rep_titles[0] if rep_titles else str(sub_df.iloc[0]["title"])

            # 子簇先尝试对齐其他旧簇，避免把本可并回的部分硬拆成新簇
            matched_old_cluster_id, _ = second_align_candidate_cluster(
                candidate_centroid=child_centroid,
                active_cluster_map=active_cluster_map,
                threshold=SPLIT_SECOND_ALIGN_THRESHOLD,
                exclude_cluster_ids={parent_cluster_id},
            )

            if matched_old_cluster_id is not None:
                conn.execute(
                    f"UPDATE cluster_items SET cluster_id = ? WHERE item_id IN ({','.join(['?'] * len(sub_df))})",
                    [matched_old_cluster_id] + sub_df["item_id"].astype(int).tolist(),
                )
                target_df = load_cluster_items_df(conn, matched_old_cluster_id)
                if not target_df.empty:
                    target_embeddings = model.encode(
                        target_df["title_clean"].tolist(),
                        batch_size=BATCH_SIZE,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                    )
                    target_embeddings = np.asarray(target_embeddings, dtype=np.float32)
                    recompute_and_update_cluster_profile(
                        conn=conn,
                        cluster_id=matched_old_cluster_id,
                        df_cluster=target_df,
                        embeddings=target_embeddings,
                        now_utc=now_utc,
                    )
                    active_cluster_map[matched_old_cluster_id]["centroid"] = l2_normalize(target_embeddings.mean(axis=0))
                    active_cluster_map[matched_old_cluster_id]["size_current"] = len(target_df)
            else:
                new_cluster_id = create_new_cluster(
                    conn=conn,
                    centroid=child_centroid,
                    canonical_title=canonical_title,
                    example_titles=rep_titles,
                    size_current=len(sub_df),
                    now_utc=now_utc,
                    parent_cluster_id=parent_cluster_id,
                )
                conn.execute(
                    f"UPDATE cluster_items SET cluster_id = ? WHERE item_id IN ({','.join(['?'] * len(sub_df))})",
                    [new_cluster_id] + sub_df["item_id"].astype(int).tolist(),
                )
                active_cluster_map[new_cluster_id] = {
                    "cluster_id": new_cluster_id,
                    "parent_cluster_id": parent_cluster_id,
                    "canonical_title": canonical_title,
                    "centroid": child_centroid,
                    "example_titles_json": json.dumps(rep_titles, ensure_ascii=False),
                    "size_current": len(sub_df),
                    "summarize_count": 0,
                    "summarize_result": "",
                    "summarize_title_count": None,
                    "first_seen_at": now_utc,
                    "last_seen_at": now_utc,
                }
                split_created_clusters += 1
                created_any = True

        # 父簇保留主子簇后，重算父簇最新画像
        keep_df_after = load_cluster_items_df(conn, parent_cluster_id)
        if not keep_df_after.empty:
            keep_embeddings_after = model.encode(
                keep_df_after["title_clean"].tolist(),
                batch_size=BATCH_SIZE,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            keep_embeddings_after = np.asarray(keep_embeddings_after, dtype=np.float32)
            recompute_and_update_cluster_profile(
                conn=conn,
                cluster_id=parent_cluster_id,
                df_cluster=keep_df_after,
                embeddings=keep_embeddings_after,
                now_utc=now_utc,
            )
            active_cluster_map[parent_cluster_id]["centroid"] = l2_normalize(keep_embeddings_after.mean(axis=0))
            active_cluster_map[parent_cluster_id]["size_current"] = len(keep_df_after)


    # 统一重建最终 touched clusters 的 size_current，确保 split 后数字正确
    touched_after = set(touched_cluster_new_counts.keys()) | set(final_new_count_updates.keys())
    for cid in list(touched_after):
        row = conn.execute(
            "SELECT COUNT(*) FROM cluster_items WHERE cluster_id = ?",
            (cid,),
        ).fetchone()
        if row is not None:
            conn.execute(
                "UPDATE clusters SET size_current = ?, last_seen_at = ? WHERE cluster_id = ?",
                (int(row[0]), now_utc, cid),
            )

    return split_created_clusters, final_new_count_updates


# =========================
# cluster size history
# =========================
def insert_cluster_size_history(
    conn: sqlite3.Connection,
    run_date_utc: str,
    snapshot_at: str,
    new_count_by_cluster: Dict[int, int],
) -> int:
    if not new_count_by_cluster:
        return 0

    rows_to_insert = []
    for cluster_id, new_count in sorted(new_count_by_cluster.items()):
        row = conn.execute(
            """
            SELECT size_current, example_titles_json
            FROM clusters
            WHERE cluster_id = ?
            """,
            (cluster_id,),
        ).fetchone()
        if row is None:
            continue
        rows_to_insert.append(
            (
                snapshot_at,
                run_date_utc,
                cluster_id,
                int(row[0]),
                int(new_count),
                row[1] or "[]",
            )
        )

    before = conn.total_changes
    conn.executemany(
        """
        INSERT INTO cluster_size_history (
            snapshot_at, run_date, cluster_id, title_count, new_title_count, example_titles_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows_to_insert,
    )
    return conn.total_changes - before


# =========================
# 主流程
# =========================
def main() -> None:
    end_dt_utc = datetime.now(timezone.utc)
    start_dt_utc = end_dt_utc - timedelta(hours=WINDOW_HOURS)
    run_date_utc = end_dt_utc.strftime("%Y-%m-%d")
    start_utc = start_dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_utc = end_dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    now_utc = end_utc

    print(f"滚动窗口: 最近 {WINDOW_HOURS} 小时")
    print("窗口开始(UTC):", start_utc)
    print("窗口结束(UTC):", end_utc)

    raw_df = load_recent_data(GDELT_DB_PATH, GDELT_TABLE_NAME, start_utc, end_utc)
    print("窗口原始数据量:", len(raw_df))
    if raw_df.empty:
        print("最近窗口内没有数据，结束。")
        return

    df, url_col = prepare_recent_df(raw_df, GDELT_TABLE_NAME)
    print("清洗后数据量:", len(df))
    if df.empty:
        print("窗口内清洗后没有有效标题，结束。")
        return
    print("识别到网址列:", url_col)

    cluster_conn = sqlite3.connect(CLUSTER_DB_PATH)
    try:
        init_cluster_db(cluster_conn)

        df_new = filter_new_items(cluster_conn, df)
        print("待处理新增标题数:", len(df_new))
        if df_new.empty:
            print("没有新增标题，结束。")
            return

        model = SentenceTransformer(MODEL_NAME)
        embeddings = model.encode(
            df_new["title_clean"].tolist(),
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        active_clusters = load_active_clusters(cluster_conn)
        print("active clusters 数量:", len(active_clusters))

        assigned_cluster_ids, assigned_scores = assign_to_existing_clusters(
            item_embeddings=embeddings,
            active_clusters=active_clusters,
            threshold=ASSIGN_THRESHOLD,
        )
        df_new = df_new.copy()
        df_new["cluster_id"] = assigned_cluster_ids
        df_new["match_score"] = assigned_scores

        matched_df = df_new[df_new["cluster_id"].notna()].copy()
        unmatched_df = df_new[df_new["cluster_id"].isna()].copy()
        matched_df["cluster_id"] = matched_df["cluster_id"].astype(int)

        unmatched_embeddings = embeddings[unmatched_df.index.to_numpy()] if not unmatched_df.empty else np.empty((0, embeddings.shape[1]), dtype=np.float32)

        print("匹配到旧簇标题数:", len(matched_df))
        print("未匹配标题数:", len(unmatched_df))

        matched_inserted = insert_cluster_item_rows(cluster_conn, matched_df, url_col=url_col, assigned_at=now_utc)
        matched_count_by_cluster = update_existing_clusters(
            conn=cluster_conn,
            assigned_df=matched_df,
            all_embeddings=embeddings,
            now_utc=now_utc,
        )

        new_cluster_count, second_aligned_cluster_count, unmatched_inserted, unmatched_count_by_cluster = persist_clusters_from_unmatched_with_second_align(
            conn=cluster_conn,
            unmatched_df=unmatched_df,
            unmatched_embeddings=unmatched_embeddings,
            url_col=url_col,
            now_utc=now_utc,
            second_align_threshold=SECOND_ALIGN_THRESHOLD,
        )

        total_new_count_by_cluster: Dict[int, int] = {}
        for d in (matched_count_by_cluster, unmatched_count_by_cluster):
            for cid, cnt in d.items():
                total_new_count_by_cluster[cid] = total_new_count_by_cluster.get(cid, 0) + cnt

        split_new_cluster_count = 0
        if ENABLE_SECOND_SPLIT:
            split_new_cluster_count, _ = apply_second_split(
                conn=cluster_conn,
                model=model,
                touched_cluster_new_counts=total_new_count_by_cluster,
                now_utc=now_utc,
            )

        history_rows = insert_cluster_size_history(
            conn=cluster_conn,
            run_date_utc=run_date_utc,
            snapshot_at=now_utc,
            new_count_by_cluster=total_new_count_by_cluster,
        )

        cluster_conn.commit()

        print("\n=== 本次结果 ===")
        print("新增入库到旧簇的标题数:", matched_inserted)
        print("候选新簇二次对齐回旧簇的簇数:", second_aligned_cluster_count)
        print("unmatched 写入标题数:", unmatched_inserted)
        print("新建 cluster 数:", new_cluster_count)
        print("簇内二次分裂新增 cluster 数:", split_new_cluster_count)
        print("cluster_size_history 新增记录数:", history_rows)
        print("cluster.sqlite 已更新。")

    finally:
        cluster_conn.close()


if __name__ == "__main__":
    main()
