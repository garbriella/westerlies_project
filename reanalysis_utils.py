"""NCEP/DOE Reanalysis II OPeNDAP 부분집합과 에디 운동량 진단."""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Callable, Sequence, TypeVar

import numpy as np
import pandas as pd
import requests

from wind_analysis import eddy_diagnostics, season_months

CATALOG_URL = "https://psl.noaa.gov/thredds/catalog/Datasets/ncep.reanalysis2/Dailies/pressure/catalog.xml"
OPENDAP_URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis2/Dailies/pressure/{variable}.{year}.nc"

F = TypeVar("F", bound=Callable[..., object])


def _cache_data(**kwargs: object) -> Callable[[F], F]:
    try:
        import streamlit as st

        return st.cache_data(**kwargs)  # type: ignore[return-value]
    except ImportError:
        return lambda function: function


@_cache_data(ttl=21_600, show_spinner=False)
def discover_reanalysis_files(catalog_url: str = CATALOG_URL) -> pd.DataFrame:
    """THREDDS 카탈로그에서 uwnd/vwnd/air의 실제 연도별 파일을 탐색한다."""
    response = requests.get(catalog_url, timeout=60)
    response.raise_for_status()
    matches = re.findall(r"(?:name|urlPath)=[\"']([^\"']*(uwnd|vwnd|air)\.(\d{4})\.nc)[\"']", response.text)
    rows = {(variable, int(year), path) for path, variable, year in matches}
    if not rows:
        # 일부 THREDDS 응답은 속성 주변 구조가 달라 파일명만 추출한다.
        rows = {(var, int(year), f"{var}.{year}.nc") for var, year in re.findall(r"(uwnd|vwnd|air)\.(\d{4})\.nc", response.text)}
    return pd.DataFrame(rows, columns=["variable", "year", "path"]).sort_values(["year", "variable"])


def available_reanalysis_years(files: pd.DataFrame, variables: Sequence[str] = ("uwnd", "vwnd")) -> list[int]:
    """요청한 모든 변수가 카탈로그에 존재하는 연도 목록을 반환한다."""
    if files.empty:
        return []
    sets = [set(files.loc[files["variable"].eq(variable), "year"].astype(int)) for variable in variables]
    return sorted(set.intersection(*sets)) if sets else []


def latest_complete_year(years: Sequence[int]) -> int:
    """현재 진행 중인 연도를 제외한 가장 최근 완전 연도를 고른다."""
    current_year = datetime.now(timezone.utc).year
    complete = [int(year) for year in years if int(year) < current_year]
    if not complete:
        raise ValueError("완전한 재분석 연도를 카탈로그에서 찾지 못했습니다.")
    return max(complete)


def _coordinate_name(dataset: object, candidates: Sequence[str]) -> str:
    for candidate in candidates:
        if candidate in dataset.coords:  # type: ignore[attr-defined]
            return candidate
    raise ValueError(f"재분석 좌표를 찾지 못했습니다: {', '.join(candidates)}")


@_cache_data(show_spinner=False, max_entries=12)
def load_reanalysis_subset(
    variable: str,
    year: int,
    season: str,
    pressure_levels_hpa: tuple[int, ...] = (850, 300),
    latitude_min: float = 20.0,
    latitude_max: float = 70.0,
) -> "object":
    """OPeNDAP에서 기간·위도·기압면을 먼저 잘라 메모리에 적재한다.

    겨울은 ``season_year`` 기준으로 전년 12월과 선택 연도 1~2월을 결합한다.
    모든 경도는 경도 평균 에디 진단을 위해 유지한다.
    """
    if variable not in {"uwnd", "vwnd", "air"}:
        raise ValueError("지원하지 않는 재분석 변수입니다.")
    if latitude_min >= latitude_max:
        raise ValueError("최소 위도는 최대 위도보다 작아야 합니다.")
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError("xarray가 설치되지 않아 재분석을 열 수 없습니다.") from exc

    source_years = (year - 1, year) if season == "겨울" else (year,)
    arrays = []
    for source_year in source_years:
        url = OPENDAP_URL.format(variable=variable, year=source_year)
        try:
            dataset = xr.open_dataset(url, engine="netcdf4", decode_times=True)
            level_name = _coordinate_name(dataset, ("level", "lev"))
            lat_name = _coordinate_name(dataset, ("lat", "latitude"))
            lon_name = _coordinate_name(dataset, ("lon", "longitude"))
            time_name = _coordinate_name(dataset, ("time",))
            lat_values = dataset[lat_name].values
            lat_slice = slice(latitude_max, latitude_min) if lat_values[0] > lat_values[-1] else slice(latitude_min, latitude_max)
            subset = dataset[variable].sel({level_name: list(pressure_levels_hpa), lat_name: lat_slice})
            months = season_months(season)
            subset = subset.where(subset[time_name].dt.month.isin(months), drop=True)
            if season == "겨울":
                season_year = subset[time_name].dt.year + (subset[time_name].dt.month == 12)
                subset = subset.where(season_year == year, drop=True)
            else:
                subset = subset.where(subset[time_name].dt.year == year, drop=True)
            subset = subset.transpose(time_name, level_name, lat_name, lon_name).load()
            arrays.append(subset)
        except Exception as exc:
            raise ConnectionError(f"{variable} {source_year} OPeNDAP 부분집합을 불러오지 못했습니다: {exc}") from exc
        finally:
            try:
                dataset.close()
            except (NameError, AttributeError):
                pass
    if not arrays:
        raise ValueError("선택 기간에 재분석 자료가 없습니다.")
    combined = xr.concat(arrays, dim="time").sortby("time") if len(arrays) > 1 else arrays[0]
    if combined.sizes.get("time", 0) < 2:
        raise ValueError("에디 편차를 계산하기에 날짜 수가 부족합니다.")
    return combined


def calculate_eddy_from_subsets(u_data: "object", v_data: "object") -> pd.DataFrame:
    """xarray u/v 부분집합에서 기압면별 위도 진단 DataFrame을 계산한다."""
    try:
        u_aligned, v_aligned = __import__("xarray").align(u_data, v_data, join="inner")
    except Exception as exc:
        raise ValueError(f"u/v 재분석 격자를 맞추지 못했습니다: {exc}") from exc
    level_name = next(name for name in ("level", "lev") if name in u_aligned.coords)
    lat_name = next(name for name in ("lat", "latitude") if name in u_aligned.coords)
    rows: list[pd.DataFrame] = []
    for level in u_aligned[level_name].values:
        u_level = u_aligned.sel({level_name: level}).transpose("time", lat_name, ...)
        v_level = v_aligned.sel({level_name: level}).transpose("time", lat_name, ...)
        result = eddy_diagnostics(u_level.values, v_level.values, u_level[lat_name].values)
        rows.append(pd.DataFrame({
            "pressure_hpa": float(level),
            "latitude_deg": u_level[lat_name].values.astype(float),
            "mean_u_ms": result["mean_u_ms"],
            "eddy_flux_m2s2": result["eddy_flux_m2s2"],
            "eddy_acceleration_ms2": result["eddy_acceleration_ms2"],
            "eddy_acceleration_ms_day": result["eddy_acceleration_ms_day"],
            "processed_days": int(u_aligned.sizes.get("time", 0)),
        }))
    return pd.concat(rows, ignore_index=True).sort_values(["pressure_hpa", "latitude_deg"])


@_cache_data(show_spinner=False, max_entries=8)
def load_and_calculate_eddy(
    year: int,
    season: str,
    latitude_min: float = 20.0,
    latitude_max: float = 70.0,
    levels: tuple[int, ...] = (850, 300),
) -> pd.DataFrame:
    """u/v 부분집합을 각각 불러 기압면별 에디 진단을 반환한다."""
    u_data = load_reanalysis_subset("uwnd", year, season, levels, latitude_min, latitude_max)
    v_data = load_reanalysis_subset("vwnd", year, season, levels, latitude_min, latitude_max)
    return calculate_eddy_from_subsets(u_data, v_data)
