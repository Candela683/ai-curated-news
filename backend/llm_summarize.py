import json
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
from openai import OpenAI


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {p}")
    return data


def resolve_api_key(api_key_value: Any, provider: str = "") -> str:
    provider = (provider or "").strip().lower()

    if api_key_value is None or (isinstance(api_key_value, str) and not api_key_value.strip()):
        if provider == "ollama":
            return "ollama"
        raise ValueError("llm.api_key must be provided for non-ollama providers")

    if not isinstance(api_key_value, str):
        raise ValueError("llm.api_key must be a string")

    s = api_key_value.strip()
    m = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", s)
    if m:
        env_name = m.group(1)
        env_val = os.getenv(env_name, "").strip()
        if env_val:
            return env_val
        if provider == "ollama":
            return "ollama"
        raise ValueError(f"Environment variable '{env_name}' is not set or empty")

    return s


def is_null_like_text(text: Any) -> bool:
    if text is None:
        return True
    s = str(text).strip()
    if not s:
        return True
    return s.lower() == "null"


def extract_text_from_response(resp: Any) -> str:
    try:
        content = resp.choices[0].message.content
    except Exception as e:
        raise ValueError(f"Cannot extract text from model response: {e}") from e

    if is_null_like_text(content):
        raise ValueError("Model returned null-like content")

    return str(content).strip()


def build_user_content(cluster_row: sqlite3.Row, items: List[sqlite3.Row]) -> str:
    lines: List[str] = []
    lines.append(f"cluster: {cluster_row['cluster_id']}")
    lines.append(f"size: {cluster_row['size_current']}")

    canonical_title = str(cluster_row["canonical_title"] or "").strip()
    if canonical_title:
        lines.append(f"canonical_title: {canonical_title}")

    lines.append("items:")
    for i, row in enumerate(items, 1):
        title = str(row["title"] or "").strip()
        url = str(row["url"] or "").strip()
        line = f"{i}. {title}" if title else f"{i}."
        if url:
            line += f"\n   url: {url}"
        lines.append(line)

    return "\n".join(lines)


def build_messages(prompt_text: str, cluster_row: sqlite3.Row, items: List[sqlite3.Row]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": prompt_text},
        {"role": "user", "content": build_user_content(cluster_row, items)},
    ]


def query_one(
    client: OpenAI,
    model: str,
    prompt_text: str,
    cluster_row: sqlite3.Row,
    items: List[sqlite3.Row],
) -> str:
    messages = build_messages(prompt_text, cluster_row, items)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return extract_text_from_response(resp)


def ensure_cluster_schema(conn: sqlite3.Connection) -> None:
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(clusters)")
    }

    required = {
        "summarize_count": "ALTER TABLE clusters ADD COLUMN summarize_count INTEGER NOT NULL DEFAULT 0",
        "summarize_result": "ALTER TABLE clusters ADD COLUMN summarize_result TEXT NOT NULL DEFAULT ''",
        "summarize_title_count": "ALTER TABLE clusters ADD COLUMN summarize_title_count INTEGER",
        "summarize_batch_at": "ALTER TABLE clusters ADD COLUMN summarize_batch_at TEXT",
        "size_current": "ALTER TABLE clusters ADD COLUMN size_current INTEGER NOT NULL DEFAULT 0",
        "canonical_title": "ALTER TABLE clusters ADD COLUMN canonical_title TEXT",
        "last_seen_at": "ALTER TABLE clusters ADD COLUMN last_seen_at TEXT",
    }

    for col, ddl in required.items():
        if col not in columns:
            conn.execute(ddl)

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_cluster_items_cluster_assigned
        ON cluster_items(cluster_id, assigned_at DESC, item_id DESC)
        """
    )


def init_result_db(db_path: str | Path) -> None:
    p = Path(db_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(p)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id INTEGER NOT NULL,
                summary TEXT NOT NULL,
                url TEXT NOT NULL,
                ts INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_results_cluster_ts
            ON results(cluster_id, ts DESC, id DESC)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_results_ts_id
            ON results(ts DESC, id DESC)
            """
        )
        conn.commit()
    finally:
        conn.close()


def insert_result(conn: sqlite3.Connection, cluster_id: int, summary: str, url: str, ts: int) -> None:
    conn.execute(
        """
        INSERT INTO results (cluster_id, summary, url, ts)
        VALUES (?, ?, ?, ?)
        """,
        (cluster_id, summary, url, ts),
    )


def dump_jsonl(path: str | Path, records: List[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def pick_url(items: List[sqlite3.Row]) -> str:
    for row in items:
        url = str(row["url"] or "").strip()
        if url:
            return url
    return ""


def fetch_clusters_to_summarize(conn: sqlite3.Connection, min_delta: int) -> List[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT
                cluster_id,
                status,
                canonical_title,
                size_current,
                summarize_count,
                summarize_result,
                summarize_title_count,
                first_seen_at,
                last_seen_at
            FROM clusters
            WHERE COALESCE(status, 'active') = 'active'
              AND size_current - COALESCE(summarize_title_count, 0) > ?
            ORDER BY size_current DESC, cluster_id ASC
            """,
            (min_delta,),
        )
    )


def fetch_cluster_items(conn: sqlite3.Connection, cluster_id: int, limit: Optional[int]) -> List[sqlite3.Row]:
    sql = """
        SELECT item_id, title, title_clean, url, published_at, updated_at, assigned_at
        FROM cluster_items
        WHERE cluster_id = ?
        ORDER BY assigned_at DESC, item_id DESC
    """
    params: Tuple[Any, ...]
    if limit is None:
        params = (cluster_id,)
    else:
        sql += " LIMIT ?"
        params = (cluster_id, limit)
    return list(conn.execute(sql, params))


def update_cluster_summary(
    conn: sqlite3.Connection,
    cluster_id: int,
    summary_text: str,
    size_current: int,
    summarize_batch_at: str,
) -> None:
    conn.execute(
        """
        UPDATE clusters
        SET summarize_result = ?,
            summarize_title_count = ?,
            summarize_count = COALESCE(summarize_count, 0) + 1,
            summarize_batch_at = ?
        WHERE cluster_id = ?
        """,
        (summary_text, size_current, summarize_batch_at, cluster_id),
    )


def process_cluster_sqlite(
    cluster_db_path: str | Path,
    prompt_yaml: str | Path,
    llm_yaml: str | Path,
) -> None:
    prompt_cfg = load_yaml(prompt_yaml)
    llm_cfg = load_yaml(llm_yaml)

    prompt_text = prompt_cfg.get("prompt")
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ValueError("prompt.yaml must contain a non-empty 'prompt' field")

    llm = llm_cfg.get("llm")
    if not isinstance(llm, dict):
        raise ValueError("llm_config.yaml must contain top-level 'llm' mapping")

    sqlite_cfg = llm_cfg.get("sqlite", {}) or {}
    if not isinstance(sqlite_cfg, dict):
        raise ValueError("llm_config.yaml sqlite must be a mapping")

    provider = llm.get("provider", "")
    base_url = llm.get("base_url")
    api_key_raw = llm.get("api_key")
    model = llm.get("model")

    if not base_url or not model:
        raise ValueError("llm_config.yaml must contain llm.base_url and llm.model")

    api_key = resolve_api_key(api_key_raw, provider=provider)

    result_db_path = sqlite_cfg.get("path", "result.sqlite")
    temp_result_path = sqlite_cfg.get("temp_result_path", "temp_result.jsonl")
    summarize_min_delta = int(sqlite_cfg.get("summarize_min_delta", 3))
    cluster_item_limit_raw = sqlite_cfg.get("summarize_item_limit", 50)
    cluster_item_limit: Optional[int]
    if cluster_item_limit_raw in (None, "", "null", "none", "inf", "infinity"):
        cluster_item_limit = None
    else:
        cluster_item_limit = int(cluster_item_limit_raw)
        if cluster_item_limit <= 0:
            cluster_item_limit = None

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    init_result_db(result_db_path)
    batch_ts = int(time.time())
    batch_time_text = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(batch_ts))
    latest_run_records: List[Dict[str, Any]] = []

    cluster_conn = sqlite3.connect(cluster_db_path)
    cluster_conn.row_factory = sqlite3.Row
    result_conn = sqlite3.connect(result_db_path)
    try:
        ensure_cluster_schema(cluster_conn)

        rows = fetch_clusters_to_summarize(cluster_conn, summarize_min_delta)
        print(f"clusters_to_summarize={len(rows)} min_delta={summarize_min_delta}")

        for idx, cluster_row in enumerate(rows, 1):
            cluster_id = int(cluster_row["cluster_id"])
            size_current = int(cluster_row["size_current"] or 0)

            items = fetch_cluster_items(cluster_conn, cluster_id, cluster_item_limit)
            url = pick_url(items)

            if not items:
                print(f"[{idx}] cluster={cluster_id} size={size_current} status=skip_no_items")
                print("-" * 80)
                continue

            try:
                result_text = query_one(
                    client=client,
                    model=model,
                    prompt_text=prompt_text,
                    cluster_row=cluster_row,
                    items=items,
                )

                if is_null_like_text(result_text):
                    print(f"[{idx}] cluster={cluster_id} size={size_current} status=skip_null")
                    print(f"url: {url}")
                    print("-" * 80)
                    continue

                update_cluster_summary(
                    conn=cluster_conn,
                    cluster_id=cluster_id,
                    summary_text=result_text,
                    size_current=size_current,
                    summarize_batch_at=batch_time_text,
                )
                cluster_conn.commit()

                insert_result(
                    conn=result_conn,
                    cluster_id=cluster_id,
                    summary=result_text,
                    url=url,
                    ts=batch_ts,
                )
                result_conn.commit()

                latest_run_records.append(
                    {
                        "cluster_id": cluster_id,
                        "size_current": size_current,
                        "summarize_title_count": size_current,
                        "url": url,
                        "summary": result_text,
                        "ts": batch_ts,
                        "summarize_batch_at": batch_time_text,
                    }
                )

                print(f"[{idx}] cluster={cluster_id} size={size_current} status=ok")
                print(f"url: {url}")
                print("output preview:")
                print(result_text[:1000])
                print("-" * 80)

            except Exception as e:
                print(f"[{idx}] cluster={cluster_id} size={size_current} status=skip_error")
                print(f"url: {url}")
                print(f"error: {str(e)}")
                print("-" * 80)
                continue

    finally:
        cluster_conn.close()
        result_conn.close()

    dump_jsonl(temp_result_path, latest_run_records)
    print(f"done. batch_ts={batch_ts}")
    print(f"latest summarize results saved to: {temp_result_path}")


def main() -> None:
    process_cluster_sqlite(
        cluster_db_path="cluster.sqlite",
        prompt_yaml="./config/prompt.yaml",
        llm_yaml="./config/llm_config.yaml",
    )


if __name__ == "__main__":
    main()
