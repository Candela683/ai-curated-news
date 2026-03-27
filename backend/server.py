from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timezone
import sqlite3
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
DB_PATH = "cluster.sqlite"

app = FastAPI(title="Cluster Summary API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # 或 ["http://localhost:5173","http://127.0.0.1:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ClusterSummaryItem(BaseModel):
    cluster_id: int
    summarize_result: str
    url_set: List[str]
    sort_time: str


class ClusterSummaryResponse(BaseModel):
    snapshot_at: str
    count: int
    has_more: bool
    next_cursor_time: Optional[str] = None
    next_cursor_id: Optional[int] = None
    items: List[ClusterSummaryItem]


def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@app.get("/")
def root():
    return {
        "service": "cluster-summary-api",
        "db_path": DB_PATH,
        "endpoints": ["/health", "/clusters/recent"],
    }


@app.get("/health")
def health():
    conn = get_conn()
    try:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM clusters").fetchone()
        return {
            "status": "ok",
            "clusters_count": int(row["cnt"]) if row else 0,
            "now_utc": utc_now_str(),
        }
    finally:
        conn.close()


@app.get("/clusters/recent", response_model=ClusterSummaryResponse)
def get_recent_clusters(
    limit: int = Query(20, ge=1, le=100, description="每页条数，默认 20"),
    snapshot_at: Optional[str] = Query(None, description="快照时间。第一页不传，后续翻页沿用第一页返回值"),
    cursor_time: Optional[str] = Query(None, description="上一页最后一条的 sort_time"),
    cursor_id: Optional[int] = Query(None, description="上一页最后一条的 cluster_id"),
):
    if (cursor_time is None) != (cursor_id is None):
        raise HTTPException(status_code=400, detail="cursor_time 和 cursor_id 必须同时传，或者同时不传")

    if snapshot_at is None:
        snapshot_at = utc_now_str()

    conn = get_conn()
    try:
        params = [snapshot_at]
        cursor_clause = ""

        if cursor_time is not None and cursor_id is not None:
            cursor_clause = """
              AND (
                    COALESCE(summarize_batch_at, last_seen_at, first_seen_at) < ?
                    OR (
                        COALESCE(summarize_batch_at, last_seen_at, first_seen_at) = ?
                        AND cluster_id < ?
                    )
                  )
            """
            params.extend([cursor_time, cursor_time, cursor_id])

        params.append(limit + 1)

        sql = f"""
        SELECT
            cluster_id,
            summarize_result,
            COALESCE(summarize_batch_at, last_seen_at, first_seen_at) AS sort_time
        FROM clusters
        WHERE COALESCE(summarize_result, '') <> ''
          AND COALESCE(summarize_batch_at, last_seen_at, first_seen_at) <= ?
          {cursor_clause}
        ORDER BY sort_time DESC, cluster_id DESC
        LIMIT ?
        """

        rows = conn.execute(sql, params).fetchall()

        has_more = len(rows) > limit
        rows = rows[:limit]

        items: List[ClusterSummaryItem] = []
        for row in rows:
            cluster_id = int(row["cluster_id"])
            summarize_result = row["summarize_result"] or ""
            sort_time = row["sort_time"]

            url_rows = conn.execute(
                """
                SELECT DISTINCT url
                FROM cluster_items
                WHERE cluster_id = ?
                  AND COALESCE(url, '') <> ''
                ORDER BY url
                """,
                (cluster_id,),
            ).fetchall()

            url_set = [r["url"] for r in url_rows]

            items.append(
                ClusterSummaryItem(
                    cluster_id=cluster_id,
                    summarize_result=summarize_result,
                    url_set=url_set,
                    sort_time=sort_time,
                )
            )

        next_cursor_time = None
        next_cursor_id = None
        if items:
            last_item = items[-1]
            next_cursor_time = last_item.sort_time
            next_cursor_id = last_item.cluster_id

        return ClusterSummaryResponse(
            snapshot_at=snapshot_at,
            count=len(items),
            has_more=has_more,
            next_cursor_time=next_cursor_time,
            next_cursor_id=next_cursor_id,
            items=items,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)