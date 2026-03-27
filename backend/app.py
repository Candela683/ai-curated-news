from pathlib import Path
import subprocess
import sys
import time
import logging
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
RUN_INTERVAL_SECONDS = 15 * 60
SCRIPT_TIMEOUT_SECONDS = 14 * 60

PIPELINE_SCRIPTS = [
    BASE_DIR / "gdelt_rss_ETL.py",
    BASE_DIR / "embedding_and_cluster.py",
    BASE_DIR / "llm_summarize.py",
]

LOG_FILE = BASE_DIR / "pipeline_runner.log"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def run_script(script_path: Path, timeout: int) -> bool:
    if not script_path.exists():
        logging.error("脚本不存在: %s", script_path)
        return False

    logging.info("开始执行脚本: %s", script_path.name)
    start_ts = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )

        duration = time.time() - start_ts

        if result.stdout:
            logging.info("[%s stdout]\n%s", script_path.name, result.stdout.strip())

        if result.stderr:
            logging.warning("[%s stderr]\n%s", script_path.name, result.stderr.strip())

        if result.returncode != 0:
            logging.error(
                "脚本执行失败: %s | returncode=%s | duration=%.2fs",
                script_path.name,
                result.returncode,
                duration,
            )
            return False

        logging.info("脚本执行成功: %s | duration=%.2fs", script_path.name, duration)
        return True

    except subprocess.TimeoutExpired as e:
        duration = time.time() - start_ts
        logging.error(
            "脚本执行超时: %s | timeout=%ss | duration=%.2fs",
            script_path.name,
            timeout,
            duration,
        )
        if e.stdout:
            logging.info("[%s partial stdout]\n%s", script_path.name, e.stdout.strip())
        if e.stderr:
            logging.warning("[%s partial stderr]\n%s", script_path.name, e.stderr.strip())
        return False

    except Exception:
        logging.exception("脚本执行异常: %s", script_path.name)
        return False


def run_pipeline_once():
    logging.info("========== 新一轮 pipeline 开始 ==========")
    round_start = time.time()

    for script in PIPELINE_SCRIPTS:
        ok = run_script(script, SCRIPT_TIMEOUT_SECONDS)
        if not ok:
            logging.warning("脚本失败，继续执行下一个: %s", script.name)

    total_duration = time.time() - round_start
    logging.info("========== 本轮 pipeline 结束 | total_duration=%.2fs ==========", total_duration)


def main():
    setup_logging()
    logging.info("pipeline runner 已启动")
    logging.info("BASE_DIR=%s", BASE_DIR)
    logging.info("RUN_INTERVAL_SECONDS=%s", RUN_INTERVAL_SECONDS)
    logging.info("SCRIPT_TIMEOUT_SECONDS=%s", SCRIPT_TIMEOUT_SECONDS)

    while True:
        cycle_start = time.time()

        try:
            run_pipeline_once()
        except Exception:
            logging.exception("本轮 pipeline 发生未捕获异常")

        elapsed = time.time() - cycle_start
        sleep_seconds = max(0, RUN_INTERVAL_SECONDS - elapsed)

        next_run_at = datetime.fromtimestamp(time.time() + sleep_seconds).strftime("%Y-%m-%d %H:%M:%S")
        logging.info(
            "距离下一轮启动还有 %.2f 秒 | next_run_at=%s",
            sleep_seconds,
            next_run_at,
        )

        time.sleep(sleep_seconds)


if __name__ == "__main__":
    main()