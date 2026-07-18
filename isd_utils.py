"""NOAA Integrated Surface Database(ISD) 관측소 대응과 자료 해석."""

from __future__ import annotations

from datetime import timedelta
from io import StringIO
from typing import Callable, TypeVar

import numpy as np
import pandas as pd
import requests

from wind_analysis import add_season_columns, haversine_km, match_nearest_times, wind_components

ISD_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
ISD_YEAR_URL = "https://www.ncei.noaa.gov/data/global-hourly/access/{year}/{usaf}{wban}.csv"
GOOD_QUALITY_CODES = {"0", "1", "4", "5"}

F = TypeVar("F", bound=Callable[..., object])


def _cache_data(**kwargs: object) -> Callable[[F], F]:
    try:
        import streamlit as st

        return st.cache_data(**kwargs)  # type: ignore[return-value]
    except ImportError:
        return lambda function: function


@_cache_data(ttl=86_400, show_spinner=False)
def download_isd_history(url: str = ISD_HISTORY_URL) -> pd.DataFrame:
    """ISD 관측소 이력 CSV를 내려받고 식별자를 문자열로 유지한다."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return parse_isd_history(response.text)


def parse_isd_history(text: str) -> pd.DataFrame:
    """ISD 관측소 CSV 문자열을 표준 열 이름으로 변환한다."""
    frame = pd.read_csv(StringIO(text), dtype={"USAF": str, "WBAN": str})
    frame["USAF"] = frame["USAF"].fillna("").str.zfill(6)
    frame["WBAN"] = frame["WBAN"].fillna("").str.zfill(5)
    frame = frame.rename(columns={
        "STATION NAME": "station_name",
        "CTRY": "country",
        "LAT": "latitude",
        "LON": "longitude",
        "ELEV(M)": "elevation_m",
        "BEGIN": "begin",
        "END": "end",
    })
    for column in ("latitude", "longitude", "elevation_m"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["station_key"] = frame["USAF"] + frame["WBAN"]
    return frame


def find_isd_candidates(
    stations: pd.DataFrame,
    latitude: float,
    longitude: float,
    station_name: str = "",
    country_code: str = "KS",
    limit: int = 10,
) -> pd.DataFrame:
    """대한민국 ISD 관측소를 이름 유사성과 위경도 거리로 정렬한다."""
    if stations.empty:
        return pd.DataFrame()
    candidates = stations[stations["country"].eq(country_code)].copy()
    candidates = candidates.dropna(subset=["latitude", "longitude"])
    candidates["distance_km"] = candidates.apply(
        lambda row: haversine_km(latitude, longitude, row["latitude"], row["longitude"]), axis=1
    )
    normalized = station_name.upper().replace("-", " ").strip()
    tokens = {token for token in normalized.split() if len(token) >= 3}
    candidates["name_match"] = candidates["station_name"].fillna("").str.upper().map(
        lambda name: int(bool(tokens) and any(token in name for token in tokens))
    )
    # 이름 일치를 우선하되, 최종 선택 화면에서 거리와 30 km 기준을 항상 함께 제시한다.
    return candidates.sort_values(["name_match", "distance_km"], ascending=[False, True]).head(limit).reset_index(drop=True)


def parse_wnd(value: object) -> dict[str, object]:
    """ISD WND ``방향,방향QC,유형,풍속(0.1m/s),풍속QC``를 해석한다."""
    missing = {
        "wind_direction_deg": np.nan,
        "wind_direction_quality": "",
        "wind_type_code": "",
        "wind_speed_ms": np.nan,
        "wind_speed_quality": "",
        "u_surface_ms": np.nan,
        "v_surface_ms": np.nan,
        "quality_rank": 99,
    }
    if not isinstance(value, str):
        return missing
    parts = value.split(",")
    if len(parts) < 5:
        return missing
    direction_raw, direction_qc, wind_type, speed_raw, speed_qc = parts[:5]
    try:
        direction = float(direction_raw)
        speed = float(speed_raw) / 10.0
    except ValueError:
        return {**missing, "wind_direction_quality": direction_qc, "wind_speed_quality": speed_qc}
    direction_good = direction != 999 and 0 <= direction <= 360 and direction_qc in GOOD_QUALITY_CODES
    speed_good = speed_raw != "9999" and speed >= 0 and speed_qc in GOOD_QUALITY_CODES
    direction_value = direction if direction_good else np.nan
    speed_value = speed if speed_good else np.nan
    if direction_good and speed_good:
        u, v = wind_components(speed_value, direction_value)
        u_value, v_value = float(u), float(v)
        rank = 0 if direction_qc in {"1", "5"} and speed_qc in {"1", "5"} else 1
    else:
        u_value = v_value = np.nan
        rank = 99
    return {
        "wind_direction_deg": direction_value,
        "wind_direction_quality": direction_qc,
        "wind_type_code": wind_type,
        "wind_speed_ms": speed_value,
        "wind_speed_quality": speed_qc,
        "u_surface_ms": u_value,
        "v_surface_ms": v_value,
        "quality_rank": rank,
    }


def _parse_scaled_field(value: object, missing_code: int, scale: float = 10.0) -> float:
    """ISD의 부호 포함 수치+품질 필드에서 유효한 값을 단위 변환한다."""
    if not isinstance(value, str):
        return np.nan
    parts = value.split(",")
    try:
        raw = int(parts[0])
    except (ValueError, IndexError):
        return np.nan
    quality = parts[1] if len(parts) > 1 else ""
    if abs(raw) == missing_code or quality not in GOOD_QUALITY_CODES:
        return np.nan
    return raw / scale


def parse_isd_observations(data: pd.DataFrame) -> pd.DataFrame:
    """ISD Global Hourly CSV의 핵심 열과 결합 필드를 품질관리하여 해석한다.

    TMP/DEW는 °C, SLP는 hPa, 풍속 및 u/v는 m/s로 반환한다.
    """
    required = {"DATE", "WND"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"ISD 자료에 필요한 열이 없습니다: {sorted(missing)}")
    result = data.copy()
    parsed_wind = pd.DataFrame(result["WND"].map(parse_wnd).tolist(), index=result.index)
    for column in parsed_wind:
        result[column] = parsed_wind[column]
    result["datetime_utc"] = pd.to_datetime(result["DATE"], utc=True, errors="coerce")
    result["datetime_kst"] = result["datetime_utc"].dt.tz_convert("Asia/Seoul")
    result["temperature_c"] = result.get("TMP", pd.Series(index=result.index, dtype=object)).map(
        lambda x: _parse_scaled_field(x, 9999)
    )
    result["dewpoint_c"] = result.get("DEW", pd.Series(index=result.index, dtype=object)).map(
        lambda x: _parse_scaled_field(x, 9999)
    )
    result["sea_level_pressure_hpa"] = result.get("SLP", pd.Series(index=result.index, dtype=object)).map(
        lambda x: _parse_scaled_field(x, 99999)
    )
    result = result[result["datetime_utc"].notna()].copy()
    return add_season_columns(result)


@_cache_data(ttl=21_600, show_spinner=False, max_entries=24)
def download_isd_year(usaf: str, wban: str, year: int) -> pd.DataFrame:
    """선택 ISD 관측소의 한 해 CSV만 다운로드하고 해석한다."""
    url = ISD_YEAR_URL.format(year=int(year), usaf=str(usaf).zfill(6), wban=str(wban).zfill(5))
    response = requests.get(url, timeout=90)
    if response.status_code == 404:
        return pd.DataFrame()
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text), dtype={"STATION": str})
    return parse_isd_observations(frame)


def download_isd_period(
    usaf: str,
    wban: str,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, list[int]]:
    """선택 기간에 필요한 연도 파일만 받아 결합하고, 없는 연도를 함께 반환한다."""
    start = pd.Timestamp(start_date, tz="UTC")
    end = pd.Timestamp(end_date, tz="UTC") + timedelta(days=1)
    frames: list[pd.DataFrame] = []
    missing_years: list[int] = []
    final_year = (end - timedelta(microseconds=1)).year
    for year in range(start.year, final_year + 1):
        frame = download_isd_year(str(usaf), str(wban), year)
        if frame.empty:
            missing_years.append(year)
        else:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(), missing_years
    combined = pd.concat(frames, ignore_index=True)
    mask = (combined["datetime_utc"] >= start) & (combined["datetime_utc"] < end)
    return combined.loc[mask].sort_values("datetime_utc").reset_index(drop=True), missing_years


@_cache_data(show_spinner=False, max_entries=8)
def match_surface_to_standard_levels(
    standard: pd.DataFrame,
    surface: pd.DataFrame,
    tolerance_minutes: int = 90,
) -> pd.DataFrame:
    """표준기압면 850/500/300 hPa를 넓은 형식으로 바꿔 지상관측과 대응한다."""
    if standard.empty or surface.empty:
        return pd.DataFrame()
    subset = standard[standard["pressure_hpa"].isin([850, 500, 300])]
    values = [column for column in ("u_ms", "v_ms", "wind_speed_ms", "wind_direction_deg") if column in subset]
    wide = subset.pivot_table(index="datetime_utc", columns="pressure_hpa", values=values)
    wide.columns = [f"{name.replace('_ms', '')}_{int(level)}_ms" if name in {"u_ms", "v_ms", "wind_speed_ms"}
                    else f"wind_direction_{int(level)}_deg" for name, level in wide.columns]
    wide = wide.reset_index()
    valid_surface = surface[surface["u_surface_ms"].notna()].copy() if "u_surface_ms" in surface else pd.DataFrame()
    matched = match_nearest_times(wide, valid_surface, tolerance_minutes=tolerance_minutes)
    if matched.empty:
        return matched
    # 명시적인 분석 열 이름을 유지한다.
    for level in (850, 500, 300):
        original = f"u_{level}_ms"
        if original not in matched and f"u_{level}" in matched:
            matched[original] = matched[f"u_{level}"]
        for column in (f"u_{level}_ms", f"v_{level}_ms", f"wind_speed_{level}_ms", f"wind_direction_{level}_deg"):
            if column not in matched:
                matched[column] = np.nan
    for column in ("u_surface_ms", "wind_speed_ms"):
        if column not in matched:
            matched[column] = np.nan
    matched["u_850_minus_surface_ms"] = matched.get("u_850_ms") - matched.get("u_surface_ms")
    matched["u_300_minus_surface_ms"] = matched.get("u_300_ms") - matched.get("u_surface_ms")
    matched["surface_to_850_speed_ratio"] = matched.get("wind_speed_ms") / matched.get("wind_speed_850_ms")
    return add_season_columns(matched)
