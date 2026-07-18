"""편서풍 분석에 사용하는 순수 계산 함수.

네트워크와 Streamlit에 의존하지 않아 단위 테스트에서 직접 사용할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

STANDARD_PRESSURES: tuple[int, ...] = (1000, 925, 850, 700, 500, 300, 250, 200)
EARTH_RADIUS_M = 6_371_000.0
OMEGA = 7.2921159e-5
R_D = 287.05

SEASON_BY_MONTH = {
    12: "겨울",
    1: "겨울",
    2: "겨울",
    3: "봄",
    4: "봄",
    5: "봄",
    6: "여름",
    7: "여름",
    8: "여름",
    9: "가을",
    10: "가을",
    11: "가을",
}


def wind_components(
    speed_ms: Sequence[float] | pd.Series | np.ndarray | float,
    direction_deg: Sequence[float] | pd.Series | np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """기상학적 풍향과 풍속을 u, v로 변환한다.

    Parameters
    ----------
    speed_ms: 풍속 (m/s).
    direction_deg: 바람이 *불어오는* 방향 (북쪽 기준 시계방향 degree).

    Returns
    -------
    (u, v): 동쪽 및 북쪽 방향 성분 (m/s). MetPy가 있으면 이를 우선 사용한다.
    """
    speed = np.asarray(speed_ms, dtype=float)
    direction = np.asarray(direction_deg, dtype=float)
    try:
        from metpy.calc import wind_components as metpy_wind_components
        from metpy.units import units

        u, v = metpy_wind_components(speed * units("m/s"), direction * units.degree)
        return np.asarray(u.magnitude, dtype=float), np.asarray(v.magnitude, dtype=float)
    except ImportError:
        radians = np.deg2rad(direction)
        return -speed * np.sin(radians), -speed * np.cos(radians)


def wind_direction_from_uv(u_ms: float | np.ndarray, v_ms: float | np.ndarray) -> np.ndarray:
    """u, v (m/s)를 기상학적 풍향(degree)으로 되돌린다."""
    u = np.asarray(u_ms, dtype=float)
    v = np.asarray(v_ms, dtype=float)
    return (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0


def is_westerly(u_ms: Sequence[float] | np.ndarray | float) -> np.ndarray:
    """동쪽으로 이동하는 서풍 성분(u > 0)을 판정한다."""
    return np.asarray(u_ms, dtype=float) > 0


def add_season_columns(data: pd.DataFrame, time_col: str = "datetime_utc") -> pd.DataFrame:
    """시각 열에 한국어 계절과 12월을 다음 해로 묶은 season_year를 추가한다."""
    result = data.copy()
    times = pd.to_datetime(result[time_col], utc=True, errors="coerce")
    result["season"] = pd.Categorical(
        times.dt.month.map(SEASON_BY_MONTH), categories=["겨울", "봄", "여름", "가을"]
    )
    result["season_year"] = times.dt.year + (times.dt.month == 12).astype(int)
    return result


def season_months(season: str) -> tuple[int, ...]:
    """계절 이름을 월 번호 튜플로 변환한다."""
    return {
        "겨울": (12, 1, 2),
        "봄": (3, 4, 5),
        "여름": (6, 7, 8),
        "가을": (9, 10, 11),
        "전체": tuple(range(1, 13)),
    }[season]


def _interpolate_one_profile(
    profile: pd.DataFrame,
    targets_hpa: Iterable[int],
    max_gap_hpa: float,
) -> pd.DataFrame:
    """단일 sounding을 로그 기압 좌표로 보간한다. 범위 밖 값은 만들지 않는다."""
    variables = ("height_m", "temperature_c", "u_ms", "v_ms")
    base = profile.sort_values("pressure_hpa").drop_duplicates("pressure_hpa")
    rows: list[dict[str, object]] = []
    metadata = {
        key: base[key].iloc[0]
        for key in ("station_id", "station_name", "datetime_utc", "datetime_kst")
        if key in base and not base.empty
    }
    for target in targets_hpa:
        row: dict[str, object] = {**metadata, "pressure_hpa": float(target)}
        pressures = base["pressure_hpa"].to_numpy(dtype=float)
        exact_indexes = np.flatnonzero(np.isclose(pressures, target, atol=0.01))
        exact_index = int(exact_indexes[0]) if exact_indexes.size else None
        row["is_interpolated"] = exact_index is None
        row["source_level_type"] = (
            "interpolated" if exact_index is None else str(base.iloc[exact_index].get("level_type", "observed"))
        )
        for variable in variables:
            row[variable] = np.nan
            row[f"{variable}_is_interpolated"] = np.nan
            values = pd.to_numeric(base[variable], errors="coerce").to_numpy(dtype=float)
            if exact_index is not None and np.isfinite(values[exact_index]):
                row[variable] = float(values[exact_index])
                row[f"{variable}_is_interpolated"] = False
                continue
            valid_mask = np.isfinite(pressures) & np.isfinite(values)
            valid_pressures = pressures[valid_mask]
            valid_values = values[valid_mask]
            if valid_pressures.size < 2:
                continue
            insertion = int(np.searchsorted(valid_pressures, target))
            if insertion == 0 or insertion == valid_pressures.size:
                continue
            p1, p2 = float(valid_pressures[insertion - 1]), float(valid_pressures[insertion])
            y1, y2 = float(valid_values[insertion - 1]), float(valid_values[insertion])
            if p2 - p1 > max_gap_hpa:
                continue
            fraction = (np.log(target) - np.log(p1)) / (np.log(p2) - np.log(p1))
            row[variable] = y1 + fraction * (y2 - y1)
            row[f"{variable}_is_interpolated"] = True
        if pd.notna(row["u_ms"]) and pd.notna(row["v_ms"]):
            row["wind_speed_ms"] = float(np.hypot(row["u_ms"], row["v_ms"]))
            row["wind_direction_deg"] = float(wind_direction_from_uv(row["u_ms"], row["v_ms"]))
        else:
            row["wind_speed_ms"] = np.nan
            row["wind_direction_deg"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def interpolate_standard_levels(
    levels: pd.DataFrame,
    targets_hpa: Iterable[int] = STANDARD_PRESSURES,
    max_gap_hpa: float = 250.0,
) -> pd.DataFrame:
    """모든 sounding을 표준 기압면으로 로그 기압 보간한다.

    양옆 레벨 간 기압 차가 ``max_gap_hpa``보다 크거나 외삽이 필요하면 NaN이다.
    풍향 대신 u와 v를 보간하며 실제값 여부를 ``is_interpolated``로 반환한다.
    """
    if levels.empty:
        return pd.DataFrame()
    required = {"datetime_utc", "pressure_hpa", "u_ms", "v_ms", "temperature_c", "height_m"}
    missing = required.difference(levels.columns)
    if missing:
        raise ValueError(f"표준기압면 보간에 필요한 열이 없습니다: {sorted(missing)}")
    groups = [
        _interpolate_one_profile(group, targets_hpa, max_gap_hpa)
        for _, group in levels.groupby("datetime_utc", sort=True)
    ]
    result = pd.concat(groups, ignore_index=True) if groups else pd.DataFrame()
    return add_season_columns(result) if not result.empty else result


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """두 위경도점의 대권거리(km)를 계산한다."""
    phi1, phi2 = np.deg2rad([lat1, lat2])
    dphi = phi2 - phi1
    dlambda = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return float(2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a)) / 1000)


def meridional_distance_m(lat_south: float, lat_north: float) -> float:
    """남북 위도 차에 대응하는 자오선 거리(m)를 반환한다."""
    return float(EARTH_RADIUS_M * np.deg2rad(lat_north - lat_south))


def match_nearest_times(
    sounding: pd.DataFrame,
    surface: pd.DataFrame,
    tolerance_minutes: int = 90,
    sounding_time: str = "datetime_utc",
    surface_time: str = "datetime_utc",
) -> pd.DataFrame:
    """각 sounding에 허용 범위 내 가장 가까운 지상관측을 UTC 기준 대응한다.

    지상 자료의 ``quality_rank``가 작을수록 좋은 자료로 간주하며 중복 시각에 우선한다.
    """
    if sounding.empty or surface.empty:
        return pd.DataFrame()
    left = sounding.copy()
    right = surface.copy()
    left[sounding_time] = pd.to_datetime(left[sounding_time], utc=True, errors="coerce")
    right[surface_time] = pd.to_datetime(right[surface_time], utc=True, errors="coerce")
    right["quality_rank"] = pd.to_numeric(right.get("quality_rank", 0), errors="coerce").fillna(99)
    right = right.sort_values([surface_time, "quality_rank"]).drop_duplicates(surface_time)
    right = right.rename(columns={surface_time: "surface_datetime_utc"})
    merged = pd.merge_asof(
        left.sort_values(sounding_time),
        right.sort_values("surface_datetime_utc"),
        left_on=sounding_time,
        right_on="surface_datetime_utc",
        direction="nearest",
        tolerance=timedelta(minutes=int(tolerance_minutes)),
        suffixes=("_upper", "_surface"),
    )
    merged = merged[merged["surface_datetime_utc"].notna()].copy()
    merged["time_difference_minutes"] = (
        merged["surface_datetime_utc"] - merged[sounding_time]
    ).abs().dt.total_seconds() / 60.0
    return merged


def coriolis_parameter(latitude_deg: float) -> float:
    """위도(degree)의 코리올리 매개변수(s^-1)를 반환한다."""
    f = float(2 * OMEGA * np.sin(np.deg2rad(latitude_deg)))
    if abs(f) < 1e-5:
        raise ValueError("코리올리 매개변수가 너무 작아 열풍 근사를 적용하기 어렵습니다.")
    return f


def thermal_wind_shear(
    temp_north_c: float,
    temp_south_c: float,
    distance_m: float,
    midpoint_latitude_deg: float,
    pressure_lower_hpa: float = 850.0,
    pressure_upper_hpa: float = 300.0,
) -> float:
    """두 지점 층평균 기온으로 열풍 동서 시어 Δu (m/s)를 근사한다."""
    if distance_m <= 0:
        raise ValueError("남북 거리는 0보다 커야 합니다.")
    if pressure_lower_hpa <= pressure_upper_hpa:
        raise ValueError("하층 기압은 상층 기압보다 커야 합니다.")
    gradient = (temp_north_c - temp_south_c) / distance_m
    return float(
        -(R_D / coriolis_parameter(midpoint_latitude_deg))
        * np.log(pressure_lower_hpa / pressure_upper_hpa)
        * gradient
    )


def paired_thermal_wind(
    north: pd.DataFrame,
    south: pd.DataFrame,
    north_lat: float,
    south_lat: float,
    lower_hpa: int = 850,
    upper_hpa: int = 300,
    tolerance_hours: int = 3,
) -> pd.DataFrame:
    """북·남 sounding 표준기압면을 시간 대응하여 열풍 예측/관측 시어를 계산한다."""
    if north_lat <= south_lat:
        raise ValueError("북쪽 관측소의 위도가 남쪽 관측소보다 높아야 합니다.")

    def reshape(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        selected = frame[frame["pressure_hpa"].isin([lower_hpa, upper_hpa])]
        wide = selected.pivot_table(index="datetime_utc", columns="pressure_hpa", values=["temperature_c", "u_ms"])
        wide.columns = [f"{prefix}_{var}_{int(level)}" for var, level in wide.columns]
        return wide.reset_index()

    north_wide = reshape(north, "north").sort_values("datetime_utc")
    south_wide = reshape(south, "south").sort_values("datetime_utc")
    if north_wide.empty or south_wide.empty:
        return pd.DataFrame()
    paired = pd.merge_asof(
        north_wide,
        south_wide.rename(columns={"datetime_utc": "south_datetime_utc"}),
        left_on="datetime_utc",
        right_on="south_datetime_utc",
        direction="nearest",
        tolerance=timedelta(hours=int(tolerance_hours)),
    ).dropna(subset=["south_datetime_utc"])
    distance = meridional_distance_m(south_lat, north_lat)
    midpoint = (north_lat + south_lat) / 2
    needed = [
        f"{side}_{var}_{level}"
        for side in ("north", "south")
        for var in ("temperature_c", "u_ms")
        for level in (lower_hpa, upper_hpa)
    ]
    if not set(needed).issubset(paired.columns):
        return pd.DataFrame()
    paired = paired.dropna(subset=needed).copy()
    if paired.empty:
        return paired
    paired["temp_north_layer_c"] = paired[[f"north_temperature_c_{lower_hpa}", f"north_temperature_c_{upper_hpa}"]].mean(axis=1)
    paired["temp_south_layer_c"] = paired[[f"south_temperature_c_{lower_hpa}", f"south_temperature_c_{upper_hpa}"]].mean(axis=1)
    paired["temperature_difference_c"] = paired["temp_north_layer_c"] - paired["temp_south_layer_c"]
    paired["meridional_distance_km"] = distance / 1000
    paired["temperature_gradient_k_per_1000km"] = paired["temperature_difference_c"] / distance * 1_000_000
    factor = -(R_D / coriolis_parameter(midpoint)) * np.log(lower_hpa / upper_hpa) / distance
    paired["predicted_shear_ms"] = factor * paired["temperature_difference_c"]
    for side in ("north", "south"):
        paired[f"observed_shear_{side}_ms"] = paired[f"{side}_u_ms_{upper_hpa}"] - paired[f"{side}_u_ms_{lower_hpa}"]
    paired["observed_shear_ms"] = paired[["observed_shear_north_ms", "observed_shear_south_ms"]].mean(axis=1)
    paired["sign_match"] = np.sign(paired["predicted_shear_ms"]) == np.sign(paired["observed_shear_ms"])
    paired["absolute_error_ms"] = (paired["predicted_shear_ms"] - paired["observed_shear_ms"]).abs()
    return add_season_columns(paired)


@dataclass(frozen=True)
class RegressionMetrics:
    """예측값-관측값 비교 통계."""

    count: int
    correlation: float
    slope: float
    rmse: float
    sign_match_rate: float


def regression_metrics(predicted: pd.Series, observed: pd.Series) -> RegressionMetrics:
    """상관, 원점 고정이 아닌 선형회귀 기울기, RMSE, 부호 일치율을 계산한다."""
    pair = pd.DataFrame({"x": predicted, "y": observed}).dropna()
    if len(pair) < 2:
        return RegressionMetrics(len(pair), np.nan, np.nan, np.nan, np.nan)
    correlation = pair["x"].corr(pair["y"])
    slope = float(np.polyfit(pair["x"], pair["y"], 1)[0]) if pair["x"].nunique() > 1 else np.nan
    rmse = float(np.sqrt(np.mean((pair["x"] - pair["y"]) ** 2)))
    sign_rate = float((np.sign(pair["x"]) == np.sign(pair["y"])).mean())
    return RegressionMetrics(len(pair), float(correlation), slope, rmse, sign_rate)


def vertical_wind_statistics(standard: pd.DataFrame) -> pd.DataFrame:
    """표준기압면별 평균 바람, 변동성과 서풍 발생 비율을 계산한다."""
    if standard.empty:
        return pd.DataFrame()
    stats = standard.groupby("pressure_hpa").agg(
        sounding_count=("datetime_utc", "nunique"),
        valid_u_count=("u_ms", "count"),
        mean_u_ms=("u_ms", "mean"),
        mean_v_ms=("v_ms", "mean"),
        mean_speed_ms=("wind_speed_ms", "mean"),
        std_u_ms=("u_ms", "std"),
        std_direction_deg=("wind_direction_deg", "std"),
    ).reset_index()
    westerly = standard.assign(westerly=standard["u_ms"] > 0).groupby("pressure_hpa")["westerly"].mean()
    stats["westerly_ratio"] = stats["pressure_hpa"].map(westerly)
    interpolation_column = "u_ms_is_interpolated" if "u_ms_is_interpolated" in standard else "is_interpolated"
    valid_u = standard[standard["u_ms"].notna()].copy()
    if interpolation_column in valid_u:
        valid_u[interpolation_column] = valid_u[interpolation_column].eq(True)
        sample_counts = valid_u.groupby("pressure_hpa")[interpolation_column].agg(
            observed_u_count=lambda values: int((~values).sum()),
            interpolated_u_count=lambda values: int(values.sum()),
        )
        stats = stats.merge(sample_counts, left_on="pressure_hpa", right_index=True, how="left")
    else:
        stats["observed_u_count"] = stats["valid_u_count"]
        stats["interpolated_u_count"] = 0
    stats[["observed_u_count", "interpolated_u_count"]] = stats[
        ["observed_u_count", "interpolated_u_count"]
    ].fillna(0).astype(int)
    stats["interpolated_u_ratio"] = np.where(
        stats["valid_u_count"] > 0,
        stats["interpolated_u_count"] / stats["valid_u_count"],
        np.nan,
    )
    return stats.sort_values("pressure_hpa", ascending=False)


def surface_upper_statistics(matched: pd.DataFrame) -> pd.DataFrame:
    """대응 자료의 지표/상층 u 통계와 서풍 비율을 긴 형식으로 반환한다."""
    if matched.empty:
        return pd.DataFrame()
    columns = {
        "surface": "u_surface_ms",
        "850 hPa": "u_850_ms",
        "500 hPa": "u_500_ms",
        "300 hPa": "u_300_ms",
    }
    rows = []
    for label, column in columns.items():
        if column not in matched:
            continue
        values = pd.to_numeric(matched[column], errors="coerce")
        rows.append({
            "level": label,
            "count": int(values.count()),
            "mean_u_ms": values.mean(),
            "std_u_ms": values.std(),
            "westerly_ratio": (values.dropna() > 0).mean() if values.notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def eddy_diagnostics(
    u: np.ndarray,
    v: np.ndarray,
    latitudes_deg: Sequence[float],
) -> dict[str, np.ndarray]:
    """(time, latitude, longitude) u/v에서 에디 운동량 플럭스와 수렴을 계산한다.

    반환 단위는 평균 u와 u'v'가 각각 m/s, m²/s²이며 가속도는 m/s² 및 m/s/day이다.
    """
    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    lat = np.asarray(latitudes_deg, dtype=float)
    if u_arr.shape != v_arr.shape or u_arr.ndim != 3 or u_arr.shape[1] != lat.size:
        raise ValueError("u와 v는 같은 (time, latitude, longitude) 모양이어야 합니다.")
    u_prime = u_arr - np.nanmean(u_arr, axis=0, keepdims=True)
    v_prime = v_arr - np.nanmean(v_arr, axis=0, keepdims=True)
    uv_prime = u_prime * v_prime
    flux = np.nanmean(uv_prime, axis=(0, 2))
    mean_u = np.nanmean(u_arr, axis=(0, 2))
    phi = np.deg2rad(lat)
    cos2 = np.cos(phi) ** 2
    weighted = cos2 * flux
    valid = np.isfinite(weighted) & np.isfinite(phi)
    acceleration = np.full(lat.shape, np.nan, dtype=float)
    if valid.sum() >= 2:
        derivative = np.gradient(weighted[valid], phi[valid])
        safe_cos2 = np.where(cos2[valid] > 1e-8, cos2[valid], np.nan)
        acceleration[valid] = -derivative / (EARTH_RADIUS_M * safe_cos2)
    return {
        "u_prime": u_prime,
        "v_prime": v_prime,
        "uv_prime": uv_prime,
        "mean_u_ms": mean_u,
        "eddy_flux_m2s2": flux,
        "eddy_acceleration_ms2": acceleration,
        "eddy_acceleration_ms_day": acceleration * 86_400.0,
    }
