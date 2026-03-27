# -*- coding: utf-8 -*-
"""
GDELT RSS -> SQLite (filtered by allowed media types) [CSV domain codes version]

改动：
- 同一轮 RSS 抓取出来的文章，published_at 统一使用本轮抓取时间 batch_published_at
- 这样后续 embedding_and_cluster.py 按 MAX(published_at) 读取时，拿到的是整批数据
- 仍然保留 limits.max_rows / limits.time 的清理逻辑
- 新增：如果数据库中存在同一 (title, domain) 的重复记录数 > 3，则将该 domain 从 domain_codes 列表中移除
  这样该 domain 后续抓取时会因为无法匹配 type 而被过滤掉
"""

import sqlite3
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import yaml

# ---------- Paths (same directory as this .py) ----------
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config/gdelt_rss_ETL_config.yaml"
CSV_PATH = BASE_DIR / "blacklist/domaincodes_final.csv"

DEFAULT_TYPE = "Others"
DEFAULT_COMM_POL = "NA"

ALLOWED_TYPES = {
    "Quality newspaper/magazine",
    "Tabloid newspaper",
    # "Digital-born news outlet",
    "Commercial broadcaster",
    "Public broadcaster",
}

DUP_TITLE_DOMAIN_THRESHOLD = 3

# ---------- Optional: registrable domain normalization ----------
try:
    import tldextract  # pip install tldextract
except Exception:
    tldextract = None


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_limit_value(x):
    """
    Support:
      - int / numeric string, e.g. 20000, "86400"
      - INF / inf / "INF" => unlimited (None)
    """
    if x is None:
        return None

    if isinstance(x, str):
        s = x.strip()
        if s.upper() == "INF":
            return None
        return int(s)

    if isinstance(x, (int, float)):
        if isinstance(x, float) and x == float("inf"):
            return None
        return int(x)

    raise ValueError(f"Unsupported limit value: {x!r}")


def to_abs_path(p: str | Path) -> str:
    p = Path(p)
    if not p.is_absolute():
        p = BASE_DIR / p
    return str(p)


def current_batch_published_at() -> str:
    """
    Return one unified UTC timestamp for the whole ETL batch.
    Example: 2026-03-27T08:15:00Z
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def extract_domain_basic(url: str) -> str | None:
    """Fallback: netloc only (no registrable-domain normalization)."""
    if not url:
        return None
    try:
        u = url.strip()
        if u.startswith("//"):
            u = "https:" + u
        if "://" not in u:
            u = "https://" + u

        netloc = urlparse(u).netloc.strip().lower()
        if not netloc:
            return None
        if ":" in netloc:
            netloc = netloc.split(":", 1)[0]
        if netloc.startswith("www."):
            netloc = netloc[4:]
        try:
            netloc = netloc.encode("idna").decode("ascii")
        except Exception:
            pass
        return netloc or None
    except Exception:
        return None


def extract_registrable_domain(url: str) -> str | None:
    """
    Prefer registrable domain:
      - forum.finexpert.e15.cz -> e15.cz
      - edition.cnn.com -> cnn.com
      - eveningnews24.co.uk -> eveningnews24.co.uk
    If tldextract not available, fallback to basic netloc.
    """
    if not url:
        return None
    u = url.strip()
    if u.startswith("//"):
        u = "https:" + u
    if "://" not in u:
        u = "https://" + u

    if tldextract is None:
        return extract_domain_basic(u)

    try:
        ext = tldextract.extract(u)
        if not ext.domain or not ext.suffix:
            return None
        return f"{ext.domain}.{ext.suffix}".lower()
    except Exception:
        return extract_domain_basic(u)


def normalize_domain_code_value(x) -> str | None:
    """
    Normalize domain strings from CSV. Accept:
      - 'cnn.com'
      - 'www.cnn.com'
      - 'https://cnn.com/a'
      - 'cnn.com/'
    -> registrable domain if possible.
    """
    if x is None:
        return None
    s = str(x).strip().lower()
    if not s or s == "nan":
        return None
    if s.startswith("www."):
        s = s[4:]

    # If looks like a URL or contains a path, normalize via URL routine
    if "://" in s or "/" in s:
        return extract_registrable_domain(s)

    # Bare domain
    return extract_registrable_domain("https://" + s)


def load_domain_codes(csv_path: Path) -> pd.DataFrame:
    """
    CSV must contain columns: domain,type,comm_pol
    """
    dc = pd.read_csv(csv_path, usecols=["domain", "type", "comm_pol"]).copy()
    dc["domain"] = dc["domain"].map(normalize_domain_code_value)
    dc = dc.dropna(subset=["domain"]).drop_duplicates(subset=["domain"], keep="last").reset_index(drop=True)
    return dc


def get_blacklisted_domains_by_duplicate_titles(
    conn: sqlite3.Connection,
    duplicate_threshold: int = DUP_TITLE_DOMAIN_THRESHOLD,
) -> set[str]:
    """
    如果数据库中同一 (title, domain) 组合的记录数 > duplicate_threshold，
    则将对应 domain 加入黑名单。

    例如 threshold=3 时，count >= 4 的 domain 会被移除。
    """
    sql = """
    SELECT DISTINCT domain
    FROM (
        SELECT domain, title, COUNT(*) AS cnt
        FROM gdelt_rss
        WHERE domain IS NOT NULL
          AND TRIM(domain) <> ''
          AND title IS NOT NULL
          AND TRIM(title) <> ''
        GROUP BY domain, title
        HAVING COUNT(*) > ?
    )
    """
    try:
        rows = conn.execute(sql, (duplicate_threshold,)).fetchall()
        return {row[0] for row in rows if row and row[0]}
    except sqlite3.OperationalError:
        # 表不存在或字段不存在时，视为没有黑名单域名
        return set()


def remove_blacklisted_domains(domain_codes: pd.DataFrame, blacklisted_domains: set[str]) -> pd.DataFrame:
    if not blacklisted_domains:
        return domain_codes

    before = len(domain_codes)
    filtered = domain_codes[~domain_codes["domain"].isin(blacklisted_domains)].copy().reset_index(drop=True)
    after = len(filtered)

    print(
        f"Removed {before - after} domain codes due to duplicated (title, domain) rows > {DUP_TITLE_DOMAIN_THRESHOLD}."
    )
    print("Blacklisted domains:", ", ".join(sorted(blacklisted_domains)))
    return filtered


def delete_blacklisted_domain_rows(conn: sqlite3.Connection, blacklisted_domains: set[str]) -> int:
    """
    反向删除库里已经存在的黑名单 domain 历史数据。
    返回删除的行数。
    """
    if not blacklisted_domains:
        return 0

    placeholders = ",".join(["?"] * len(blacklisted_domains))
    sql = f"DELETE FROM gdelt_rss WHERE domain IN ({placeholders})"
    cur = conn.execute(sql, tuple(sorted(blacklisted_domains)))
    conn.commit()
    return int(cur.rowcount or 0)


def fetch_rss_to_df(url: str, domain_codes: pd.DataFrame, batch_published_at: str) -> pd.DataFrame:
    r = requests.get(url, timeout=30, headers={"User-Agent": "rss2sqlite/1.0"})
    r.raise_for_status()

    root = ET.fromstring(r.content)
    rows: list[dict] = []

    for item in root.findall(".//item"):
        guid = (item.findtext("guid") or "").strip()
        link = (item.findtext("link") or "").strip()

        rows.append(
            {
                "guid": guid,
                "title": (item.findtext("title") or "").strip(),
                "link": link,
                "description": (item.findtext("description") or "").strip(),
                # 不再使用 RSS 原始 pubDate 作为入库 published_at
                "published_at": batch_published_at,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "uniq_key",
                "title",
                "link",
                "domain",
                "type",
                "comm_pol",
                "published_at",
                "description",
            ]
        )

    # uniq_key
    df["uniq_key"] = df["guid"].fillna("").astype(str).str.strip()
    df.loc[df["uniq_key"].eq(""), "uniq_key"] = df["link"].fillna("").astype(str).str.strip()

    df = df[df["uniq_key"].ne("")].copy()
    df = df.drop_duplicates(subset=["uniq_key"], keep="last").reset_index(drop=True)

    # domain (registrable)
    df["domain"] = df["link"].map(extract_registrable_domain)

    # join domain -> type/comm_pol
    df = df.merge(domain_codes, how="left", on="domain")

    # fallback for unmatched
    df["type"] = df["type"].fillna(DEFAULT_TYPE)
    df["comm_pol"] = df["comm_pol"].fillna(DEFAULT_COMM_POL)

    # --- filter: only keep allowed media types ---
    before = len(df)
    df = df[df["type"].isin(ALLOWED_TYPES)].copy()
    after = len(df)

    print(f"Batch published_at: {batch_published_at}")
    print(f"Rows before type filter: {before}, after: {after}")
    if after > 0:
        print("Type counts:\n", df["type"].value_counts().to_string())

    return df[
        [
            "uniq_key",
            "title",
            "link",
            "domain",
            "type",
            "comm_pol",
            "published_at",
            "description",
        ]
    ]


def init_db(conn: sqlite3.Connection):
    conn.execute(
        """
    CREATE TABLE IF NOT EXISTS gdelt_rss (
        uniq_key TEXT PRIMARY KEY,
        title TEXT,
        link TEXT,
        domain TEXT,
        type TEXT,
        comm_pol TEXT,
        published_at TEXT,
        description TEXT,
        wordcloud TEXT,
        tag TEXT,
        updated_at TEXT DEFAULT (datetime('now'))
    )
    """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_gdelt_published ON gdelt_rss(published_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_gdelt_domain ON gdelt_rss(domain)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_gdelt_type ON gdelt_rss(type)")
    conn.commit()


def upsert_df(conn: sqlite3.Connection, df: pd.DataFrame):
    if df.empty:
        return
    conn.executemany(
        """
    INSERT INTO gdelt_rss (uniq_key, title, link, domain, type, comm_pol, published_at, description, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
    ON CONFLICT(uniq_key) DO UPDATE SET
        title=excluded.title,
        link=excluded.link,
        domain=excluded.domain,
        type=excluded.type,
        comm_pol=excluded.comm_pol,
        published_at=excluded.published_at,
        description=excluded.description,
        updated_at=datetime('now')
    """,
        df.itertuples(index=False, name=None),
    )
    conn.commit()


def delete_older_than(conn: sqlite3.Connection, max_age_seconds: int | None):
    """
    Delete rows whose published_at is older than max_age_seconds.
    max_age_seconds = None means unlimited (INF), so do nothing.
    """
    if max_age_seconds is None:
        return

    if max_age_seconds <= 0:
        conn.execute("DELETE FROM gdelt_rss")
        conn.commit()
        return

    conn.execute(
        """
    DELETE FROM gdelt_rss
    WHERE published_at IS NOT NULL
      AND published_at < strftime('%Y-%m-%dT%H:%M:%SZ', 'now', ?)
    """,
        (f"-{int(max_age_seconds)} seconds",),
    )
    conn.commit()


def trim_to_max(conn: sqlite3.Connection, max_rows: int | None):
    if max_rows is None:
        return

    if max_rows <= 0:
        conn.execute("DELETE FROM gdelt_rss")
        conn.commit()
        return

    conn.execute(
        """
    DELETE FROM gdelt_rss
    WHERE uniq_key IN (
        SELECT uniq_key FROM gdelt_rss
        ORDER BY
            CASE WHEN published_at IS NULL THEN 1 ELSE 0 END,
            published_at DESC,
            updated_at DESC
        LIMIT -1 OFFSET ?
    )
    """,
        (max_rows,),
    )
    conn.commit()


def main():
    cfg = load_config(CONFIG_PATH)
    rss_url = cfg["rss"]["url"]
    db_path = to_abs_path(cfg["database"]["path"])
    max_rows = parse_limit_value(cfg["limits"]["max_rows"])
    max_age_seconds = parse_limit_value(cfg["limits"]["time"])

    domain_codes = load_domain_codes(CSV_PATH)

    with sqlite3.connect(db_path) as conn:
        init_db(conn)

        # 先基于当前库内容，找出重复(title, domain)过多的域名
        blacklisted_domains = get_blacklisted_domains_by_duplicate_titles(conn, DUP_TITLE_DOMAIN_THRESHOLD)
        domain_codes = remove_blacklisted_domains(domain_codes, blacklisted_domains)

        # 反向删除库里这些域名的旧数据，避免历史脏数据继续留在库中
        deleted_rows = delete_blacklisted_domain_rows(conn, blacklisted_domains)
        if deleted_rows > 0:
            print(f"Deleted {deleted_rows} historical rows from blacklisted domains.")

        # 同一轮抓取统一一个 published_at
        batch_published_at = current_batch_published_at()
        df = fetch_rss_to_df(rss_url, domain_codes, batch_published_at=batch_published_at)

        if df.empty:
            print("No rows to write (after filtering).")
        else:
            upsert_df(conn, df)

        delete_older_than(conn, max_age_seconds)
        trim_to_max(conn, max_rows)

    print(f"Done. batch_published_at={batch_published_at}")


if __name__ == "__main__":
    main()
