"""작은 합성 IGRA 고정폭 문자열 파서 테스트."""

import numpy as np
import pytest

from igra_utils import parse_igra_soundings


def _put(chars: list[str], start: int, end: int, value: str) -> None:
    chars[start:end] = list(value.rjust(end - start))


def _header(level_count: int = 2) -> str:
    chars = [" "] * 80
    chars[0] = "#"
    chars[1:12] = list("KSM00047138")
    _put(chars, 13, 17, "2024")
    _put(chars, 18, 20, "01")
    _put(chars, 21, 23, "15")
    _put(chars, 24, 26, "00")
    _put(chars, 32, 36, str(level_count))
    return "".join(chars)


def _level(pressure_pa: int, height_m: int, temp_tenths: int, direction: int, speed_tenths: int) -> str:
    chars = [" "] * 55
    chars[0:2] = list("10")
    _put(chars, 9, 15, str(pressure_pa))
    chars[15] = "B"
    _put(chars, 16, 21, str(height_m))
    chars[21] = "B"
    _put(chars, 22, 27, str(temp_tenths))
    chars[27] = "B"
    _put(chars, 40, 45, str(direction))
    _put(chars, 46, 51, str(speed_tenths))
    return "".join(chars)


def test_parse_igra_sample_string_and_units() -> None:
    sample = "\n".join([
        _header(),
        _level(85000, 1500, 25, 270, 123),
        _level(50000, 5500, -9999, -9999, -8888),
    ])
    result = parse_igra_soundings(sample, "POHANG")
    assert len(result) == 2
    first = result.iloc[0]
    assert first["pressure_hpa"] == 850.0
    assert first["temperature_c"] == 2.5
    assert first["wind_speed_ms"] == 12.3
    assert first["u_ms"] == pytest.approx(12.3, abs=1e-8)
    assert first["pressure_quality_flag"] == "B"
    second = result.iloc[1]
    assert np.isnan(second["temperature_c"])
    assert np.isnan(second["u_ms"])


def test_invalid_header_and_missing_pressure_are_safe() -> None:
    sample = "\n".join([_header(1), _level(-9999, -9999, -9999, -9999, -9999)])
    assert parse_igra_soundings(sample).empty

