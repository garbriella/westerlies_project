"""ISD 결합 필드와 품질 플래그 테스트."""

import numpy as np
import pandas as pd
import pytest

from isd_utils import parse_isd_observations, parse_wnd


def test_parse_wnd_sample() -> None:
    result = parse_wnd("270,1,N,0123,1")
    assert result["wind_direction_deg"] == 270.0
    assert result["wind_speed_ms"] == 12.3
    assert result["u_surface_ms"] == pytest.approx(12.3, abs=1e-8)
    assert result["quality_rank"] == 0


def test_parse_wnd_missing_and_bad_quality() -> None:
    assert np.isnan(parse_wnd("999,9,N,9999,9")["u_surface_ms"])
    assert np.isnan(parse_wnd("270,3,N,0123,1")["wind_direction_deg"])
    assert np.isnan(parse_wnd(None)["wind_speed_ms"])


def test_parse_isd_core_fields_and_missing_values() -> None:
    source = pd.DataFrame({
        "DATE": ["2024-01-01T00:00:00", "2024-01-01T01:00:00"],
        "WND": ["270,1,N,0050,1", "999,9,N,9999,9"],
        "TMP": ["+0123,1", "+9999,9"],
        "DEW": ["+0045,1", "+9999,9"],
        "SLP": ["10132,1", "99999,9"],
        "STATION": ["47138099999", "47138099999"],
        "NAME": ["POHANG", "POHANG"],
        "LATITUDE": [36.032, 36.032],
        "LONGITUDE": [129.38, 129.38],
        "ELEVATION": [4.0, 4.0],
    })
    result = parse_isd_observations(source)
    assert result.iloc[0]["temperature_c"] == 12.3
    assert result.iloc[0]["dewpoint_c"] == 4.5
    assert result.iloc[0]["sea_level_pressure_hpa"] == 1013.2
    assert np.isnan(result.iloc[1]["temperature_c"])
    assert np.isnan(result.iloc[1]["u_surface_ms"])
