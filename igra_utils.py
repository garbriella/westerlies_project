"""NOAA IGRA 2.2 관측소 탐색, 다운로드 및 고정폭 sounding 파서."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from io import BytesIO, TextIOWrapper
from pathlib import Path
import re
import tempfile
from typing import Callable, Iterable, TypeVar
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
import requests

from wind_analysis import add_season_columns, interpolate_standard_levels, wind_components

STATION_LIST_URL = "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/doc/igra2-station-list.txt"
IGRA_POR_URL = "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por/"
IGRA_Y2D_URL = "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-y2d/"
IGRA_FORMAT_URL = "https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/doc/igra2-data-format.txt"

F = TypeVar("F", bound=Callable[..., object])


def _cache_data(**kwargs: object) -> Callable[[F], F]:
    """Streamlit 외부 테스트에서도 동작하는 cache_data 데코레이터."""
    try:
        import streamlit as st

        return st.cache_data(**kwargs)  # type: ignore[return-value]
    except ImportError:
        return lambda function: function


def _get_text(url: str, timeout: int = 30) -> str:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


@_cache_data(ttl=86_400, show_spinner=False)
def download_igra_station_list(url: str = STATION_LIST_URL) -> pd.DataFrame:
    """IGRA 관측소 목록을 내려받아 위치와 자료기간 DataFrame으로 반환한다."""
    return parse_igra_station_list(_get_text(url))


def parse_igra_station_list(text: str) -> pd.DataFrame:
    """IGRA 고정폭 관측소 목록 문자열을 파싱한다."""
    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        if len(line) < 80:
            continue
        try:
            row = {
                "station_id": line[0:11].strip(),
                "latitude": float(line[12:20]),
                "longitude": float(line[21:30]),
                "elevation_m": float(line[31:37]),
                "state": line[38:40].strip(),
                "station_name": line[41:71].strip(),
                "first_year": int(line[72:76]),
                "last_year": int(line[77:81]),
                "observation_count": int(line[82:88]),
            }
        except (ValueError, IndexError):
            continue
        if row["station_id"]:
            rows.append(row)
    return pd.DataFrame(rows)


def korean_igra_stations(stations: pd.DataFrame) -> pd.DataFrame:
    """국가 코드가 KS인 대한민국 IGRA 관측소를 반환한다."""
    if stations.empty:
        return stations.copy()
    return stations[stations["station_id"].str.startswith("KS", na=False)].sort_values(
        ["last_year", "station_name"], ascending=[False, True]
    ).reset_index(drop=True)


@_cache_data(ttl=21_600, show_spinner=False)
def discover_igra_files(station_id: str) -> list[dict[str, str]]:
    """공식 POR, Y2D 디렉터리에서 관측소 ID와 일치하는 자료 파일을 탐색한다.

    전체기간(POR)을 먼저 반환하고 파일이 없거나 접근 실패하면 Y2D를 탐색한다.
    """
    escaped = re.escape(station_id.upper())
    pattern = re.compile(r'href=["\']([^"\']*' + escaped + r'[^"\']*\.(?:zip|txt))["\']', re.IGNORECASE)
    errors: list[str] = []
    found: list[dict[str, str]] = []
    for source, base_url in (("전체기간", IGRA_POR_URL), ("최근자료", IGRA_Y2D_URL)):
        try:
            listing = _get_text(base_url)
            matches = sorted(set(pattern.findall(listing)))
            if matches:
                found.extend({"source": source, "filename": name, "url": base_url + name} for name in matches)
        except requests.RequestException as exc:
            errors.append(f"{source}: {exc}")
    if found:
        return found
    if errors:
        raise ConnectionError("IGRA 파일 목록에 접근하지 못했습니다. " + " | ".join(errors))
    raise FileNotFoundError(f"{station_id}와 일치하는 IGRA 자료 파일이 없습니다.")


@_cache_data(ttl=21_600, show_spinner=False)
def download_igra_archive(url: str) -> str:
    """IGRA 압축자료를 청크 단위로 ``data_cache``에 저장하고 상대경로를 반환한다.

    40 MB 이상 ZIP과 압축 해제 문자열을 cache_data가 여러 번 복제하지 않도록 파일 자체는
    git에서 제외된 로컬 캐시에 둔다. 함수 반환값이 캐시되므로 같은 세션에서 재다운로드하지 않는다.
    """
    cache_dir = Path("data_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = re.sub(r"[^A-Za-z0-9_.-]", "_", url.rstrip("/").split("/")[-1])
    if not filename:
        raise ValueError("IGRA URL에서 안전한 파일명을 찾지 못했습니다.")
    target = cache_dir / filename
    if target.is_file() and target.stat().st_size > 0:
        return str(target)
    response = requests.get(url, timeout=120, stream=True)
    response.raise_for_status()
    total = 0
    with tempfile.NamedTemporaryFile(dir=cache_dir, prefix="igra_", suffix=".part", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    temporary.write(chunk)
                    total += len(chunk)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    if total == 0:
        temporary_path.unlink(missing_ok=True)
        raise ValueError("IGRA 응답 파일이 비어 있습니다.")
    temporary_path.replace(target)
    return str(target)


def extract_igra_text(content: bytes, filename: str = "") -> str:
    """ZIP이면 첫 텍스트 파일을 풀고, 아니면 원문을 UTF-8 문자열로 변환한다."""
    if filename.lower().endswith(".zip") or content[:2] == b"PK":
        try:
            with ZipFile(BytesIO(content)) as archive:
                members = [name for name in archive.namelist() if not name.endswith("/")]
                if not members:
                    raise ValueError("IGRA ZIP 안에 자료 파일이 없습니다.")
                return archive.read(members[0]).decode("utf-8", errors="replace")
        except BadZipFile as exc:
            raise ValueError("IGRA ZIP 파일이 손상되었습니다.") from exc
    return content.decode("utf-8", errors="replace")


def _integer(field: str) -> int | None:
    try:
        value = int(field)
    except (TypeError, ValueError):
        return None
    return None if value in (-8888, -9999) else value


def _parse_igra_lines(
    lines: Iterable[str],
    station_name: str = "",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """IGRA 라인 iterable을 sounding 단위로 소비해 메모리 사용을 제한한다."""
    start = pd.Timestamp(start_date, tz="UTC") if start_date else None
    end = (pd.Timestamp(end_date, tz="UTC") + timedelta(days=1)) if end_date else None
    rows: list[dict[str, object]] = []
    iterator = iter(lines)
    for header in iterator:
        if not header.startswith("#"):
            continue
        try:
            station_id = header[1:12].strip()
            year, month, day, hour = (int(header[a:b]) for a, b in ((13, 17), (18, 20), (21, 23), (24, 26)))
            num_levels = int(header[32:36])
        except (ValueError, IndexError):
            continue
        records = []
        for _ in range(num_levels):
            try:
                records.append(next(iterator))
            except StopIteration:
                break
        if hour == 99:
            continue
        try:
            sounding_time = pd.Timestamp(datetime(year, month, day, hour, tzinfo=timezone.utc))
        except ValueError:
            continue
        if (start is not None and sounding_time < start) or (end is not None and sounding_time >= end):
            continue
        kst_time = sounding_time.tz_convert("Asia/Seoul")
        for line in records:
            if line.startswith("#") or len(line) < 51:
                continue
            pressure = _integer(line[9:15])
            if pressure is None or pressure <= 0:
                continue
            height = _integer(line[16:21])
            temperature = _integer(line[22:27])
            direction = _integer(line[40:45])
            speed = _integer(line[46:51])
            direction_value = float(direction) if direction is not None and 0 <= direction <= 360 else np.nan
            speed_value = speed / 10 if speed is not None and speed >= 0 else np.nan
            rows.append({
                "station_id": station_id,
                "station_name": station_name,
                "datetime_utc": sounding_time,
                "datetime_kst": kst_time,
                "pressure_hpa": pressure / 100.0,
                "height_m": float(height) if height is not None else np.nan,
                "temperature_c": temperature / 10 if temperature is not None else np.nan,
                "wind_direction_deg": direction_value,
                "wind_speed_ms": speed_value,
                "u_ms": np.nan,
                "v_ms": np.nan,
                "level_type": f"{line[0:1]}{line[1:2]}",
                "pressure_quality_flag": line[15:16].strip(),
                "height_quality_flag": line[21:22].strip(),
                "temperature_quality_flag": line[27:28].strip(),
            })
    if not rows:
        return pd.DataFrame(columns=[
            "station_id", "station_name", "datetime_utc", "datetime_kst", "pressure_hpa",
            "height_m", "temperature_c", "wind_direction_deg", "wind_speed_ms", "u_ms", "v_ms",
            "level_type", "pressure_quality_flag", "height_quality_flag", "temperature_quality_flag",
        ])
    frame = pd.DataFrame(rows)
    wind_valid = frame["wind_direction_deg"].notna() & frame["wind_speed_ms"].notna()
    if wind_valid.any():
        u_values, v_values = wind_components(
            frame.loc[wind_valid, "wind_speed_ms"].to_numpy(),
            frame.loc[wind_valid, "wind_direction_deg"].to_numpy(),
        )
    frame.loc[wind_valid, "u_ms"] = u_values
    frame.loc[wind_valid, "v_ms"] = v_values
    for column in ("pressure_hpa", "height_m", "temperature_c", "wind_direction_deg", "wind_speed_ms", "u_ms", "v_ms"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float32")
    for column in (
        "station_id", "station_name", "level_type", "pressure_quality_flag",
        "height_quality_flag", "temperature_quality_flag",
    ):
        frame[column] = frame[column].astype("category")
    frame = frame.sort_values(["datetime_utc", "pressure_hpa"], ascending=[True, False])
    frame = frame.drop_duplicates(["datetime_utc", "pressure_hpa"], keep="first").reset_index(drop=True)
    return add_season_columns(frame)


@_cache_data(show_spinner=False, max_entries=8)
def parse_igra_soundings(
    text: str,
    station_name: str = "",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """작은 IGRA 2.2 고정폭 문자열을 레벨별 DataFrame으로 변환한다.

    기압(hPa), 고도(m), 기온(°C), 풍속(m/s), u/v(m/s)와 원본 품질 플래그를 반환한다.
    대용량 공식 ZIP은 ``parse_igra_archive``가 압축을 스트리밍 해제하므로 더 적합하다.
    """
    return _parse_igra_lines(text.splitlines(), station_name, start_date, end_date)


@_cache_data(show_spinner=False, max_entries=8)
def parse_igra_archive(
    content: bytes | str,
    filename: str,
    station_name: str = "",
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """IGRA ZIP을 전체 문자열로 풀지 않고 줄 단위로 파싱한다.

    전체기간 파일이 수백 MB로 팽창해도 선택 기간의 레벨만 DataFrame에 보관한다.
    손상된 ZIP은 이해 가능한 ``ValueError``로 변환한다.
    """
    is_path = isinstance(content, str)
    signature = b""
    if is_path:
        path = Path(content)
        if not path.is_file():
            raise FileNotFoundError(f"IGRA 로컬 캐시 파일을 찾지 못했습니다: {path}")
        with path.open("rb") as handle:
            signature = handle.read(2)
    else:
        signature = content[:2]
    if filename.lower().endswith(".zip") or signature == b"PK":
        try:
            zip_source = Path(content) if is_path else BytesIO(content)
            with ZipFile(zip_source) as archive:
                members = [name for name in archive.namelist() if not name.endswith("/")]
                if not members:
                    raise ValueError("IGRA ZIP 안에 자료 파일이 없습니다.")
                with archive.open(members[0]) as raw, TextIOWrapper(raw, encoding="utf-8", errors="replace") as text:
                    return _parse_igra_lines(text, station_name, start_date, end_date)
        except BadZipFile as exc:
            raise ValueError("IGRA ZIP 파일이 손상되었습니다.") from exc
    raw_source = Path(content).open("rb") if is_path else BytesIO(content)
    with raw_source, TextIOWrapper(raw_source, encoding="utf-8", errors="replace") as text:
        return _parse_igra_lines(text, station_name, start_date, end_date)


@_cache_data(show_spinner=False, max_entries=8)
def cached_standard_levels(
    levels: pd.DataFrame,
    targets_hpa: tuple[int, ...],
    max_gap_hpa: float = 250.0,
) -> pd.DataFrame:
    """IGRA 레벨을 선택 표준기압면으로 보간하고 결과를 캐시한다."""
    return interpolate_standard_levels(levels, targets_hpa, max_gap_hpa)
