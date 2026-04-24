from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_MODEL_ROUTE = "gemma-3-4b"
DEFAULT_SAE_ROUTE = "17-gemmascope-2-res-16k"
DEFAULT_NUM_FEATURES = 16384

CSV_COLUMNS = [
    "feature_index",
    "status",
    "http_status",
    "neuronpedia_title",
    "neuronpedia_source",
    "neuronpedia_url",
    "fetched_at_utc",
    "error",
]


@dataclass
class FetchResult:
    feature_index: int
    status: str
    http_status: int | None
    neuronpedia_title: str
    neuronpedia_source: str
    neuronpedia_url: str
    fetched_at_utc: str
    error: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-route", default=DEFAULT_MODEL_ROUTE)
    parser.add_argument("--sae-route", default=DEFAULT_SAE_ROUTE)
    parser.add_argument("--num-features", type=int, default=DEFAULT_NUM_FEATURES)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=float, default=1.5)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--output-csv",
        default=(
            "fast-ai-detector/outputs/neuronpedia_lookup/"
            "gemma_3_4b__17_gemmascope_2_res_16k__feature_lookup.csv"
        ),
    )
    parser.add_argument("--stats-json", default=None)
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def feature_url(model_route: str, sae_route: str, feature_index: int) -> str:
    return f"https://www.neuronpedia.org/{model_route}/{sae_route}/{feature_index}"


def parse_title_and_source(html_text: str) -> tuple[str, str]:
    if "No Explanations Found" in html_text:
        return "NO_EXPLANATION", ""
    title_match = re.search(r'Explanations.*?<h1[^>]*title=\"([^\"]+)\"', html_text, re.S)
    if title_match is None:
        return "PARSE_FAIL", ""
    title = html.unescape(title_match.group(1))
    source_match = re.search(
        r'Explanations.*?<a[^>]*href=\"/explanation-type/[^\"]+\"[^>]*>([^<]+)</a>',
        html_text,
        re.S,
    )
    source = html.unescape(source_match.group(1)).strip() if source_match else ""
    return title, source


def fetch_feature(
    *,
    model_route: str,
    sae_route: str,
    feature_index: int,
    timeout: float,
    retries: int,
    retry_backoff: float,
) -> FetchResult:
    url = feature_url(model_route, sae_route, feature_index)
    headers = {
        "User-Agent": "fast-ai-detector-neuronpedia-lookup/0.1",
    }
    last_error = ""
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers=headers)
            with urlopen(request, timeout=timeout) as response:
                status = int(getattr(response, "status", 200))
                html_text = response.read().decode("utf-8", errors="replace")
            title, source = parse_title_and_source(html_text)
            row_status = "ok"
            if title == "NO_EXPLANATION":
                row_status = "no_explanation"
            elif title == "PARSE_FAIL":
                row_status = "parse_fail"
            return FetchResult(
                feature_index=feature_index,
                status=row_status,
                http_status=status,
                neuronpedia_title=title,
                neuronpedia_source=source,
                neuronpedia_url=url,
                fetched_at_utc=utc_now_iso(),
                error="",
            )
        except HTTPError as exc:
            last_error = f"HTTPError {exc.code}: {exc.reason}"
            status = int(exc.code)
            if status == 404:
                return FetchResult(
                    feature_index=feature_index,
                    status="http_404",
                    http_status=status,
                    neuronpedia_title="",
                    neuronpedia_source="",
                    neuronpedia_url=url,
                    fetched_at_utc=utc_now_iso(),
                    error=last_error,
                )
        except URLError as exc:
            last_error = f"URLError: {exc.reason}"
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(retry_backoff ** attempt)
    return FetchResult(
        feature_index=feature_index,
        status="request_error",
        http_status=None,
        neuronpedia_title="",
        neuronpedia_source="",
        neuronpedia_url=url,
        fetched_at_utc=utc_now_iso(),
        error=last_error,
    )


def load_completed_indices(output_csv: Path) -> set[int]:
    if not output_csv.exists():
        return set()
    completed: set[int] = set()
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = row.get("feature_index")
            if value is None or value == "":
                continue
            completed.add(int(value))
    return completed


def ensure_csv_header(output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists() and output_csv.stat().st_size > 0:
        return
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()


def append_rows(output_csv: Path, rows: Iterable[FetchResult]) -> int:
    count = 0
    with output_csv.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        for row in rows:
            writer.writerow(asdict(row))
            count += 1
        handle.flush()
    return count


def summarize_status_counts(output_csv: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not output_csv.exists():
        return counts
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            status = row.get("status", "")
            counts[status] = counts.get(status, 0) + 1
    return counts


def write_stats(
    *,
    stats_path: Path,
    output_csv: Path,
    total_target: int,
    completed_before_start: int,
    processed_this_run: int,
    started_at: float,
    remaining_now: int,
) -> None:
    elapsed = max(time.time() - started_at, 1e-9)
    status_counts = summarize_status_counts(output_csv)
    payload = {
        "output_csv": str(output_csv.resolve()),
        "updated_at_utc": utc_now_iso(),
        "total_target": total_target,
        "completed_before_start": completed_before_start,
        "processed_this_run": processed_this_run,
        "completed_total": completed_before_start + processed_this_run,
        "remaining": remaining_now,
        "elapsed_seconds": elapsed,
        "rows_per_second_this_run": processed_this_run / elapsed,
        "status_counts_total": status_counts,
    }
    stats_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_csv = Path(args.output_csv)
    stats_path = Path(args.stats_json) if args.stats_json else output_csv.with_suffix(".stats.json")
    end_index = args.num_features if args.end_index is None else min(args.end_index, args.num_features)
    if args.start_index < 0 or end_index < args.start_index:
        raise ValueError("Invalid start/end range.")

    ensure_csv_header(output_csv)
    completed = load_completed_indices(output_csv)
    target_indices = list(range(args.start_index, end_index))
    pending_indices = [feature_index for feature_index in target_indices if feature_index not in completed]

    print(
        json.dumps(
            {
                "model_route": args.model_route,
                "sae_route": args.sae_route,
                "start_index": args.start_index,
                "end_index": end_index,
                "total_target": len(target_indices),
                "already_completed": len(target_indices) - len(pending_indices),
                "pending": len(pending_indices),
                "output_csv": str(output_csv.resolve()),
                "stats_json": str(stats_path.resolve()),
                "max_workers": args.max_workers,
            },
            indent=2,
            sort_keys=True,
        )
    )

    started_at = time.time()
    completed_before_start = len(target_indices) - len(pending_indices)
    processed_this_run = 0
    rows_since_log: list[FetchResult] = []
    in_flight: dict[Future[FetchResult], int] = {}

    def submit_more(executor: ThreadPoolExecutor, source_iter: Iterable[int]) -> None:
        while len(in_flight) < args.max_workers:
            try:
                feature_index = next(source_iter)
            except StopIteration:
                return
            future = executor.submit(
                fetch_feature,
                model_route=args.model_route,
                sae_route=args.sae_route,
                feature_index=feature_index,
                timeout=args.timeout,
                retries=args.retries,
                retry_backoff=args.retry_backoff,
            )
            in_flight[future] = feature_index

    pending_iter = iter(pending_indices)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        submit_more(executor, pending_iter)
        while in_flight:
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            finished_rows: list[FetchResult] = []
            for future in done:
                feature_index = in_flight.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = FetchResult(
                        feature_index=feature_index,
                        status="worker_error",
                        http_status=None,
                        neuronpedia_title="",
                        neuronpedia_source="",
                        neuronpedia_url=feature_url(args.model_route, args.sae_route, feature_index),
                        fetched_at_utc=utc_now_iso(),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                finished_rows.append(result)
            finished_rows.sort(key=lambda row: row.feature_index)
            append_rows(output_csv, finished_rows)
            processed_this_run += len(finished_rows)
            rows_since_log.extend(finished_rows)

            completed_total = completed_before_start + processed_this_run
            remaining_now = len(target_indices) - completed_total
            if processed_this_run % max(args.log_every, 1) < len(finished_rows) or remaining_now == 0:
                elapsed = max(time.time() - started_at, 1e-9)
                rate = processed_this_run / elapsed
                status_counts_batch: dict[str, int] = {}
                for row in rows_since_log:
                    status_counts_batch[row.status] = status_counts_batch.get(row.status, 0) + 1
                print(
                    json.dumps(
                        {
                            "processed_this_run": processed_this_run,
                            "completed_total": completed_total,
                            "remaining": remaining_now,
                            "elapsed_seconds": round(elapsed, 2),
                            "rows_per_second_this_run": round(rate, 2),
                            "batch_status_counts": status_counts_batch,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                rows_since_log = []
                write_stats(
                    stats_path=stats_path,
                    output_csv=output_csv,
                    total_target=len(target_indices),
                    completed_before_start=completed_before_start,
                    processed_this_run=processed_this_run,
                    started_at=started_at,
                    remaining_now=remaining_now,
                )
            submit_more(executor, pending_iter)

    write_stats(
        stats_path=stats_path,
        output_csv=output_csv,
        total_target=len(target_indices),
        completed_before_start=completed_before_start,
        processed_this_run=processed_this_run,
        started_at=started_at,
        remaining_now=0,
    )
    print(
        json.dumps(
            {
                "status": "done",
                "processed_this_run": processed_this_run,
                "completed_total": len(target_indices),
                "output_csv": str(output_csv.resolve()),
                "stats_json": str(stats_path.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
