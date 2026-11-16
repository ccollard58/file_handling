#!/usr/bin/env python3
"""Export ZwiftPower profile data and visible cycling tables to an Excel file.

Usage:
    python zwiftpower_export.py
    python zwiftpower_export.py --profile-id 115014 --output cycling_data.xlsx

The script uses a visible Chromium window so you can sign in to ZwiftPower
yourself if the profile or its data requires authentication. It does not save
credentials or browser cookies.

Install the browser dependency once with:
    pip install playwright
    python -m playwright install chromium
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    from openpyxl.utils import get_column_letter
except ImportError as exc:  # pragma: no cover - dependency message for end users
    raise SystemExit(
        "openpyxl is required. Install it with: python -m pip install openpyxl"
    ) from exc

try:
    from playwright.async_api import async_playwright
except ImportError as exc:  # pragma: no cover - dependency message for end users
    raise SystemExit(
        "Playwright is required. Install it with:\n"
        "  python -m pip install playwright\n"
        "  python -m playwright install chromium"
    ) from exc


DEFAULT_PROFILE_ID = "115014"
ZWIFTPOWER_HOST = "zwiftpower.com"
TAB_NAMES = ("Activities", "Analysis")
EXCEL_MAX_ROWS = 1_048_576
EXCEL_MAX_CELL_CHARS = 32_767
RAW_JSON_CHUNK_SIZE = 30_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect ZwiftPower profile and cycling data into an .xlsx workbook. "
            "A visible browser window is used for optional manual sign-in."
        )
    )
    parser.add_argument(
        "--profile-id",
        default=DEFAULT_PROFILE_ID,
        help=f"ZwiftPower profile ID (default: {DEFAULT_PROFILE_ID})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Excel output path (default: zwiftpower_<ID>_<timestamp>.xlsx)",
    )
    parser.add_argument(
        "--no-browser-prompt",
        action="store_true",
        help="Do not pause for a manual login before collecting the profile.",
    )
    args = parser.parse_args()
    if not args.profile_id.isdigit():
        parser.error("--profile-id must contain digits only")
    return args


def canonical_url(url: str) -> str:
    """Remove cache-busting query parameters so duplicate payloads collapse."""
    parts = urlparse(url)
    query = [(key, value) for key, value in parse_qsl(parts.query) if key != "_"]
    return urlunparse(parts._replace(query=urlencode(query)))


def is_profile_data_response(url: str) -> bool:
    parts = urlparse(url)
    host = (parts.hostname or "").lower()
    if host != ZWIFTPOWER_HOST and not host.endswith("." + ZWIFTPOWER_HOST):
        return False
    path = parts.path.lower()
    return "/cache3/" in path and any(
        token in path for token in ("profile", "activity", "analysis", "race")
    )


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def flatten_record(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested objects into columns; retain nested lists as JSON text."""
    flattened: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            column = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, dict):
                flattened.update(flatten_record(child, column))
            elif isinstance(child, list):
                if child and all(not isinstance(item, (dict, list)) for item in child):
                    flattened[column] = ", ".join(clean_text(item) for item in child)
                else:
                    flattened[column] = json.dumps(
                        child, ensure_ascii=False, separators=(",", ":")
                    )
            else:
                flattened[column] = child
    else:
        flattened[prefix or "value"] = value
    return flattened


def collect_json_datasets(payload: Any, prefix: str) -> list[tuple[str, list[dict[str, Any]]]]:
    """Find arrays of records in an API payload and make spreadsheet datasets."""
    datasets: list[tuple[str, list[dict[str, Any]]]] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else str(key)
                if isinstance(child, list) and child and all(
                    isinstance(item, dict) for item in child
                ):
                    datasets.append(
                        (child_path, [flatten_record(item) for item in child])
                    )
                walk(child, child_path)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                if isinstance(child, dict):
                    walk(child, f"{path}[{index}]")

    walk(payload, "")
    if not datasets:
        if isinstance(payload, list) and payload:
            datasets.append(("records", [flatten_record(item) for item in payload]))
        elif isinstance(payload, dict):
            datasets.append(("profile", [flatten_record(payload)]))
    return [(f"{prefix} · {name}", rows) for name, rows in datasets if rows]


def safe_cell_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, str):
        # Avoid converting text scraped from the page into an Excel formula.
        if value.startswith(("=", "+", "-", "@")):
            value = "'" + value
        if len(value) > EXCEL_MAX_CELL_CHARS:
            value = value[: EXCEL_MAX_CELL_CHARS - 20] + "… [cell truncated]"
    return value


def unique_sheet_name(label: str, used: set[str]) -> str:
    name = re.sub(r"[\\/*?:\[\]]", "_", label).strip(" '\"") or "Data"
    name = name[:31]
    base = name
    suffix = 2
    while name.casefold() in used:
        marker = f"_{suffix}"
        name = base[: 31 - len(marker)] + marker
        suffix += 1
    used.add(name.casefold())
    return name


def add_table_sheet(
    workbook: Workbook,
    used_names: set[str],
    label: str,
    records: list[dict[str, Any]],
) -> None:
    if not records:
        return
    sheet = workbook.create_sheet(unique_sheet_name(label, used_names))
    columns: list[str] = []
    seen_columns: set[str] = set()
    for record in records:
        for column in record:
            if column not in seen_columns:
                columns.append(str(column))
                seen_columns.add(str(column))

    sheet.append(columns or ["value"])
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")

    for record in records[: EXCEL_MAX_ROWS - 1]:
        sheet.append([safe_cell_value(record.get(column)) for column in columns])
    sheet.freeze_panes = "A2"
    if columns:
        sheet.auto_filter.ref = sheet.dimensions
        for index, column in enumerate(columns, start=1):
            sample_width = max(
                [len(str(column))]
                + [
                    len(str(safe_cell_value(record.get(column)) or ""))
                    for record in records[:100]
                ]
            )
            sheet.column_dimensions[get_column_letter(index)].width = min(
                max(sample_width + 2, 12), 44
            )


async def read_visible_tables(page: Any, tab_label: str) -> list[tuple[str, list[dict[str, Any]]]]:
    """Read currently visible HTML tables as a fallback and for tabular data."""
    tables = await page.evaluate(
        """() => Array.from(document.querySelectorAll('table')).filter(table => {
          const style = window.getComputedStyle(table);
          const rect = table.getBoundingClientRect();
          return style.display !== 'none' && style.visibility !== 'hidden' &&
            rect.width > 0 && rect.height > 0;
        }).map((table, index) => {
          const headerRows = Array.from(table.querySelectorAll('thead tr'));
          const firstRow = table.querySelector('tr');
          const headerCells = headerRows.length
            ? Array.from(headerRows[headerRows.length - 1].querySelectorAll('th,td'))
            : (firstRow ? Array.from(firstRow.querySelectorAll('th,td')) : []);
          const headers = headerCells.map(cell => (cell.innerText || cell.textContent || '').trim());
          const bodyRows = Array.from(table.querySelectorAll('tbody tr'));
          const rows = (bodyRows.length ? bodyRows : Array.from(table.querySelectorAll('tr')).slice(1))
            .map(row => Array.from(row.querySelectorAll('th,td'))
              .map(cell => (cell.innerText || cell.textContent || '').replace(/\\s+/g, ' ').trim()))
            .filter(row => row.some(value => value));
          const caption = table.querySelector('caption')?.innerText?.trim() || '';
          const id = table.id || '';
          const classes = typeof table.className === 'string' ? table.className : '';
          return {index, headers, rows, caption, id, classes};
        })"""
    )

    collected: list[tuple[str, list[dict[str, Any]]]] = []
    for item in tables:
        rows = item.get("rows") or []
        headers = item.get("headers") or []
        width = max([len(headers)] + [len(row) for row in rows])
        if not width:
            continue
        names = []
        for index in range(width):
            name = clean_text(headers[index]) if index < len(headers) else ""
            names.append(name or f"Column {index + 1}")
        records = [
            {names[index]: row[index] if index < len(row) else "" for index in range(width)}
            for row in rows
        ]
        if records:
            title = item.get("caption") or item.get("id") or item.get("classes") or f"Table {item['index'] + 1}"
            collected.append((f"{tab_label} · {title}", records))
    return collected


async def click_next_page(page: Any) -> bool:
    selectors = (
        ".paginate_button.next",
        "button[aria-label='Next']",
        "a[aria-label='Next']",
        ".pagination .next a",
        ".pagination .next button",
    )
    for selector in selectors:
        candidates = page.locator(selector)
        for index in range(await candidates.count()):
            candidate = candidates.nth(index)
            try:
                if not await candidate.is_visible() or not await candidate.is_enabled():
                    continue
                disabled = await candidate.evaluate(
                    "el => el.classList.contains('disabled') || "
                    "el.getAttribute('aria-disabled') === 'true' || "
                    "el.closest('.disabled') !== null"
                )
                if disabled:
                    continue
                await candidate.click(timeout=3_000)
                await page.wait_for_timeout(800)
                return True
            except Exception:
                continue

    # Some table widgets expose plain text buttons instead of a standard class.
    next_text = page.get_by_role("button", name=re.compile(r"^\s*next\s*$", re.I))
    for index in range(await next_text.count()):
        candidate = next_text.nth(index)
        try:
            if await candidate.is_visible() and await candidate.is_enabled():
                await candidate.click(timeout=3_000)
                await page.wait_for_timeout(800)
                return True
        except Exception:
            continue
    return False


async def read_tables_across_pages(
    page: Any, tab_label: str
) -> list[tuple[str, list[dict[str, Any]]]]:
    """Collect paginated table rows so the workbook includes older activities."""
    accumulated: dict[str, list[dict[str, Any]]] = {}
    seen_rows: dict[str, set[str]] = {}
    max_pages = 250
    for _ in range(max_pages):
        for label, records in await read_visible_tables(page, tab_label):
            destination = accumulated.setdefault(label, [])
            seen = seen_rows.setdefault(label, set())
            for record in records:
                signature = json.dumps(record, ensure_ascii=False, sort_keys=True)
                if signature not in seen:
                    destination.append(record)
                    seen.add(signature)
        if not await click_next_page(page):
            break
    else:
        print(
            f"Note: stopped pagination after {max_pages} pages on the {tab_label} view."
        )
    return [(label, records) for label, records in accumulated.items()]


async def click_profile_tab(page: Any, tab_name: str) -> bool:
    pattern = re.compile(rf"^\s*{re.escape(tab_name)}\s*$", re.IGNORECASE)
    for role in ("tab", "link", "button"):
        candidates = page.get_by_role(role, name=pattern)
        try:
            count = await candidates.count()
        except Exception:
            continue
        for index in range(count):
            candidate = candidates.nth(index)
            try:
                if await candidate.is_visible():
                    await candidate.click(timeout=5_000)
                    await page.wait_for_timeout(1_500)
                    return True
            except Exception:
                continue
    return False


def add_source_sheet(
    workbook: Workbook,
    used_names: set[str],
    profile_id: str,
    profile_url: str,
    final_url: str,
    page_title: str,
    captured_payloads: dict[str, Any],
    table_count: int,
    visited_tabs: list[str],
) -> None:
    sheet = workbook.create_sheet(unique_sheet_name("Export Info", used_names), 0)
    rows = [
        ("Profile ID", profile_id),
        ("Source URL", profile_url),
        ("Final page URL", final_url),
        ("Page title", page_title),
        ("Exported at", datetime.now().astimezone().isoformat(timespec="seconds")),
        ("Profile JSON responses", len(captured_payloads)),
        ("Visible tables collected", table_count),
        ("Profile tabs visited", ", ".join(visited_tabs)),
        (
            "Coverage note",
            "Includes data returned by ZwiftPower profile endpoints and visible tables "
            "on the Profile, Activities, and Analysis views. Data hidden by account "
            "privacy settings or not loaded by ZwiftPower is not available to export.",
        ),
    ]
    sheet.append(["Field", "Value"])
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
    for row in rows:
        sheet.append([safe_cell_value(value) for value in row])
    sheet.column_dimensions["A"].width = 28
    sheet.column_dimensions["B"].width = 105
    sheet.freeze_panes = "A2"


async def scrape_profile(profile_id: str, manual_login_prompt: bool) -> tuple[
    dict[str, Any], list[tuple[str, list[dict[str, Any]]]], str, str, list[str]
]:
    profile_url = f"https://zwiftpower.com/profile.php?z={profile_id}"
    payloads: dict[str, Any] = {}
    table_sets: list[tuple[str, list[dict[str, Any]]]] = []
    response_tasks: set[asyncio.Task[Any]] = set()
    visited_tabs = ["Profile"]

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        async def capture_response(response: Any) -> None:
            if not is_profile_data_response(response.url):
                return
            try:
                content_type = (await response.all_headers()).get("content-type", "")
                if "json" not in content_type.lower() and not response.url.lower().endswith(".json"):
                    return
                payload = await response.json()
                payloads[canonical_url(response.url)] = payload
            except Exception:
                # Some profile requests can be empty, denied, or non-JSON.
                return

        def on_response(response: Any) -> None:
            task = asyncio.create_task(capture_response(response))
            response_tasks.add(task)
            task.add_done_callback(response_tasks.discard)

        page.on("response", on_response)
        await page.goto(profile_url, wait_until="domcontentloaded", timeout=60_000)
        await page.wait_for_timeout(2_000)

        if manual_login_prompt:
            print(
                "A Chromium window is open. If ZwiftPower asks you to sign in, "
                "complete sign-in in that window."
            )
            await asyncio.to_thread(
                input,
                "When the profile is ready (or if it is already visible), press Enter here... ",
            )
            await page.goto(profile_url, wait_until="domcontentloaded", timeout=60_000)
            await page.wait_for_timeout(2_000)

        table_sets.extend(await read_tables_across_pages(page, "Profile"))

        for tab_name in TAB_NAMES:
            # Reload the profile between views so one tab's navigation cannot
            # prevent the other tab from being visited.
            await page.goto(profile_url, wait_until="domcontentloaded", timeout=60_000)
            await page.wait_for_timeout(1_000)
            if await click_profile_tab(page, tab_name):
                visited_tabs.append(tab_name)
                table_sets.extend(await read_tables_across_pages(page, tab_name))
            else:
                print(f"Note: could not find the '{tab_name}' tab; continuing.")

        if response_tasks:
            await asyncio.gather(*list(response_tasks), return_exceptions=True)
        final_url = page.url
        page_title = await page.title()
        await browser.close()

    return payloads, table_sets, final_url, page_title, visited_tabs


def build_workbook(
    profile_id: str,
    output_path: Path,
    payloads: dict[str, Any],
    table_sets: list[tuple[str, list[dict[str, Any]]]],
    final_url: str,
    page_title: str,
    visited_tabs: list[str],
) -> int:
    workbook = Workbook()
    used_names: set[str] = set()
    workbook.remove(workbook.active)

    count_of_tables = len(table_sets)
    data_sheets = 0
    raw_sheet_name = unique_sheet_name("Raw JSON", used_names)
    raw_sheet = workbook.create_sheet(raw_sheet_name)
    raw_sheet.append(["Response URL", "Chunk", "Chunks", "JSON payload"])
    for cell in raw_sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")

    for response_url, payload in sorted(payloads.items()):
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        chunks = [
            serialized[index : index + RAW_JSON_CHUNK_SIZE]
            for index in range(0, len(serialized), RAW_JSON_CHUNK_SIZE)
        ] or [""]
        for chunk_index, chunk in enumerate(chunks, start=1):
            raw_sheet.append([response_url, chunk_index, len(chunks), chunk])

        endpoint_name = Path(urlparse(response_url).path).name or "profile"
        for dataset_name, records in collect_json_datasets(payload, endpoint_name):
            add_table_sheet(workbook, used_names, dataset_name, records)
            data_sheets += 1

    for label, records in table_sets:
        add_table_sheet(workbook, used_names, label, records)
        data_sheets += 1

    raw_sheet.freeze_panes = "A2"
    raw_sheet.auto_filter.ref = raw_sheet.dimensions
    raw_sheet.column_dimensions["A"].width = 64
    raw_sheet.column_dimensions["B"].width = 10
    raw_sheet.column_dimensions["C"].width = 10
    raw_sheet.column_dimensions["D"].width = 80

    add_source_sheet(
        workbook,
        used_names,
        profile_id,
        f"https://zwiftpower.com/profile.php?z={profile_id}",
        final_url,
        page_title,
        payloads,
        count_of_tables,
        visited_tabs,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)
    return data_sheets


async def async_main() -> int:
    args = parse_args()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or Path(
        f"zwiftpower_{args.profile_id}_{timestamp}.xlsx"
    )

    try:
        payloads, table_sets, final_url, page_title, visited_tabs = await scrape_profile(
            args.profile_id,
            manual_login_prompt=not args.no_browser_prompt,
        )
    except Exception as exc:
        message = str(exc)
        if "Executable doesn't exist" in message or "playwright install" in message.lower():
            print(
                "Chromium for Playwright is not installed. Run:\n"
                "  python -m playwright install chromium",
                file=sys.stderr,
            )
        else:
            print(f"Could not collect the ZwiftPower profile: {exc}", file=sys.stderr)
        return 1

    if not payloads and not table_sets:
        print(
            "No profile data was collected. Confirm the profile is available in the "
            "browser and that ZwiftPower has finished loading, then run the script again.",
            file=sys.stderr,
        )
        return 1

    data_sheets = build_workbook(
        args.profile_id,
        output_path,
        payloads,
        table_sets,
        final_url,
        page_title,
        visited_tabs,
    )
    print(f"Saved Excel workbook: {output_path.resolve()}")
    print(
        f"Collected {len(payloads)} profile JSON responses and "
        f"{len(table_sets)} visible tables across {len(visited_tabs)} profile view(s)."
    )
    if data_sheets == 0:
        print("The workbook contains the raw JSON responses and export information.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main()))
