"""네트워크 없이 실행되는 핵심 바람 계산 테스트."""

import numpy as np
import pandas as pd
import pytest

from wind_analysis import (
    add_season_columns,
    eddy_diagnostics,
    interpolate_standard_levels,
    is_westerly,
    match_nearest_times,
    paired_thermal_wind,
    regression_metrics,
    surface_upper_statistics,
    thermal_wind_shear,
    vertical_wind_statistics,
    wind_components,
)


def test_wind_components_and_westerly_easterly() -> None:
    u, v = wind_components([10.0, 10.0], [270.0, 90.0])
    assert u[0] == pytest.approx(10.0, abs=1e-8)
    assert u[1] == pytest.approx(-10.0, abs=1e-8)
    assert np.allclose(v, 0.0, atol=1e-8)
    assert is_westerly(u).tolist() == [True, False]


def test_log_pressure_interpolation_without_extrapolation() -> None:
    time = pd.Timestamp("2024-01-01T00:00:00Z")
    levels = pd.DataFrame({
        "station_id": ["TEST", "TEST"],
        "station_name": ["TEST", "TEST"],
        "datetime_utc": [time, time],
        "datetime_kst": [time.tz_convert("Asia/Seoul")] * 2,
        "pressure_hpa": [900.0, 800.0],
        "height_m": [1000.0, 2000.0],
        "temperature_c": [10.0, 0.0],
        "u_ms": [5.0, 15.0],
        "v_ms": [0.0, 0.0],
        "level_type": ["20", "20"],
    })
    result = interpolate_standard_levels(levels, (850, 1000), max_gap_hpa=150)
    middle = result[result["pressure_hpa"].eq(850)].iloc[0]
    expected_fraction = (np.log(850) - np.log(800)) / (np.log(900) - np.log(800))
    assert middle["u_ms"] == pytest.approx(15 + expected_fraction * (5 - 15))
    assert bool(middle["is_interpolated"])
    assert np.isnan(result.loc[result["pressure_hpa"].eq(1000), "u_ms"].iloc[0])


def test_season_and_winter_season_year() -> None:
    data = pd.DataFrame({"datetime_utc": pd.to_datetime([
        "2023-12-15T00:00:00Z", "2024-01-15T00:00:00Z", "2024-07-01T00:00:00Z"
    ], utc=True)})
    result = add_season_columns(data)
    assert result["season"].tolist() == ["겨울", "겨울", "여름"]
    assert result["season_year"].tolist() == [2024, 2024, 2024]


def test_nearest_surface_match_respects_tolerance_and_quality() -> None:
    sounding = pd.DataFrame({"datetime_utc": pd.to_datetime(["2024-01-01T00:00Z", "2024-01-01T12:00Z"])})
    surface = pd.DataFrame({
        "datetime_utc": pd.to_datetime(["2024-01-01T00:20Z", "2024-01-01T00:20Z", "2024-01-01T15:30Z"]),
        "u_surface_ms": [2.0, 5.0, 9.0],
        "quality_rank": [2, 0, 0],
    })
    result = match_nearest_times(sounding, surface, tolerance_minutes=90)
    assert len(result) == 1
    assert result.iloc[0]["u_surface_ms"] == 5.0
    assert result.iloc[0]["time_difference_minutes"] == 20.0


def test_thermal_wind_sign_for_colder_north() -> None:
    shear = thermal_wind_shear(
        temp_north_c=-10.0,
        temp_south_c=0.0,
        distance_m=1_000_000.0,
        midpoint_latitude_deg=35.0,
        pressure_lower_hpa=850,
        pressure_upper_hpa=300,
    )
    assert shear > 0  # 북반구에서 북쪽이 차가우면 상층 서풍 강화 방향


def test_surface_upper_statistics() -> None:
    matched = pd.DataFrame({
        "u_surface_ms": [-1.0, 3.0, 5.0],
        "u_850_ms": [2.0, 4.0, 8.0],
        "u_500_ms": [4.0, 8.0, 12.0],
        "u_300_ms": [8.0, 16.0, 20.0],
    })
    stats = surface_upper_statistics(matched).set_index("level")
    assert stats.loc["surface", "mean_u_ms"] == pytest.approx(7 / 3)
    assert stats.loc["surface", "westerly_ratio"] == pytest.approx(2 / 3)
    assert stats.loc["850 hPa", "westerly_ratio"] == 1.0


def test_eddy_anomalies_flux_and_convergence_shape_with_nan() -> None:
    u = np.array([
        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
        [[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]],
        [[3.0, 4.0], [6.0, np.nan], [9.0, 10.0]],
    ])
    v = np.array([
        [[-1.0, 0.0], [0.0, 1.0], [1.0, 2.0]],
        [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]],
        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
    ])
    result = eddy_diagnostics(u, v, [30.0, 40.0, 50.0])
    assert result["u_prime"].shape == u.shape
    assert result["uv_prime"].shape == u.shape
    assert result["eddy_flux_m2s2"].shape == (3,)
    assert result["eddy_acceleration_ms2"].shape == (3,)
    assert np.isfinite(result["eddy_acceleration_ms_day"]).sum() >= 2


def test_exact_standard_level_is_not_marked_interpolated() -> None:
    time = pd.Timestamp("2024-02-01T00:00:00Z")
    levels = pd.DataFrame({
        "station_id": ["TEST"], "station_name": ["TEST"], "datetime_utc": [time],
        "datetime_kst": [time.tz_convert("Asia/Seoul")], "pressure_hpa": [850.0],
        "height_m": [1500.0], "temperature_c": [3.0], "u_ms": [8.0], "v_ms": [1.0],
        "level_type": ["10"],
    })
    result = interpolate_standard_levels(levels, (850,))
    assert not bool(result.iloc[0]["is_interpolated"])
    assert result.iloc[0]["u_ms"] == 8.0


def test_paired_thermal_wind_uses_three_hour_window_and_mean_observed_shear() -> None:
    north_time = pd.Timestamp("2024-01-01T00:00:00Z")
    south_time = pd.Timestamp("2024-01-01T02:00:00Z")

    def frame(time: pd.Timestamp, temps: tuple[float, float], winds: tuple[float, float]) -> pd.DataFrame:
        return pd.DataFrame({
            "datetime_utc": [time, time], "pressure_hpa": [850, 300],
            "temperature_c": list(temps), "u_ms": list(winds),
        })

    result = paired_thermal_wind(
        frame(north_time, (-5.0, -45.0), (5.0, 25.0)),
        frame(south_time, (5.0, -35.0), (4.0, 20.0)),
        north_lat=38.0, south_lat=34.0,
    )
    assert len(result) == 1
    assert result.iloc[0]["predicted_shear_ms"] > 0
    assert result.iloc[0]["observed_shear_ms"] == pytest.approx(18.0)
    assert bool(result.iloc[0]["sign_match"])


def test_regression_metrics_values() -> None:
    metrics = regression_metrics(pd.Series([1.0, 2.0, 3.0]), pd.Series([2.0, 4.0, 6.0]))
    assert metrics.count == 3
    assert metrics.correlation == pytest.approx(1.0)
    assert metrics.slope == pytest.approx(2.0)
    assert metrics.sign_match_rate == 1.0


def test_vertical_statistics_preserve_means_and_add_sample_provenance() -> None:
    standard = pd.DataFrame({
        "pressure_hpa": [850.0, 850.0, 850.0],
        "datetime_utc": pd.to_datetime([
            "2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z",
        ], utc=True),
        "u_ms": [4.0, 8.0, np.nan],
        "v_ms": [1.0, 3.0, 5.0],
        "wind_speed_ms": [5.0, 10.0, np.nan],
        "wind_direction_deg": [270.0, 260.0, np.nan],
        "u_ms_is_interpolated": [False, True, False],
    })
    stats = vertical_wind_statistics(standard).iloc[0]
    assert stats["mean_u_ms"] == pytest.approx(6.0)
    assert stats["mean_speed_ms"] == pytest.approx(7.5)
    assert stats["westerly_ratio"] == pytest.approx(2 / 3)
    assert stats["valid_u_count"] == 2
    assert stats["observed_u_count"] == 1
    assert stats["interpolated_u_count"] == 1
    assert stats["interpolated_u_ratio"] == pytest.approx(0.5)
