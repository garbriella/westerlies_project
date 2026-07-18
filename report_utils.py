"""실제 계산 결과를 바탕으로 한 자동 해석과 다운로드 자료 생성."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from wind_analysis import regression_metrics


def dataframe_csv(data: pd.DataFrame) -> bytes:
    """Excel에서도 한글이 보이도록 UTF-8 BOM CSV bytes를 반환한다."""
    if data is None or data.empty:
        return b""
    return data.to_csv(index=False).encode("utf-8-sig")


def observational_interpretation(vertical: pd.DataFrame, matched: pd.DataFrame) -> list[str]:
    """고도별 및 지표-상층 결과를 조건부 문장으로 해석한다."""
    messages: list[str] = []
    if not vertical.empty:
        indexed = vertical.set_index("pressure_hpa")
        if 850 in indexed.index and 300 in indexed.index:
            delta = float(indexed.loc[300, "mean_u_ms"] - indexed.loc[850, "mean_u_ms"])
            if np.isfinite(delta):
                direction = "강해졌습니다" if delta > 0 else "강해지지 않았습니다"
                messages.append(f"850→300 hPa에서 평균 u 변화는 {delta:+.2f} m/s로, 선택 자료에서는 상층 서풍 성분이 {direction}.")
    if not matched.empty and {"u_surface_ms", "u_850_ms"}.issubset(matched):
        pair = matched[["u_surface_ms", "u_850_ms"]].dropna()
        if not pair.empty:
            difference = (pair["u_850_ms"] - pair["u_surface_ms"]).mean()
            surface_ratio = (pair["u_surface_ms"] > 0).mean()
            variability = pair["u_surface_ms"].std()
            messages.append(
                f"대응 자료에서 850 hPa u−지표 u의 평균은 {difference:+.2f} m/s이고, "
                f"지표 서풍 성분 발생 비율은 {surface_ratio:.1%}, 지표 u 표준편차는 {variability:.2f} m/s입니다."
            )
    if not messages:
        messages.append("관측 자료가 아직 없어 고도별·지표 비교를 해석하지 않았습니다.")
    return messages


def thermal_interpretation(results: pd.DataFrame) -> list[str]:
    """열풍 예측-관측 일치 정도와 계절 차이를 자동 해석한다."""
    if results.empty:
        return ["열풍 계산 결과가 없어 이론과 관측의 일치 정도를 판단하지 않았습니다."]
    metrics = regression_metrics(results["predicted_shear_ms"], results["observed_shear_ms"])
    messages = [
        f"유효 {metrics.count}쌍에서 상관계수는 {metrics.correlation:.2f}, RMSE는 {metrics.rmse:.2f} m/s, "
        f"부호 일치율은 {metrics.sign_match_rate:.1%}입니다."
    ]
    if np.isfinite(metrics.correlation):
        if metrics.correlation >= 0.5 and metrics.sign_match_rate >= 0.7:
            messages.append("선택 자료는 남북 기온 경도와 연직 서풍 시어가 열풍 관계와 대체로 같은 방향으로 변하는 패턴을 보입니다.")
        else:
            messages.append("일치가 강하지 않습니다. 비지균 성분, 관측소의 경도 차이, 시간 대응 및 두 지점만 사용한 기온 경도 근사가 영향을 줄 수 있습니다.")
    seasonal = results.groupby("season", observed=True)[["temperature_difference_c", "observed_shear_ms"]].mean()
    if {"겨울", "여름"}.issubset(seasonal.index):
        winter_gradient = abs(float(seasonal.loc["겨울", "temperature_difference_c"]))
        summer_gradient = abs(float(seasonal.loc["여름", "temperature_difference_c"]))
        winter_shear = float(seasonal.loc["겨울", "observed_shear_ms"])
        summer_shear = float(seasonal.loc["여름", "observed_shear_ms"])
        if winter_gradient > summer_gradient and winter_shear > summer_shear:
            messages.append("겨울의 남북 기온 차와 관측 상층 시어가 모두 여름보다 커 열풍 관계와 일치하는 계절 패턴입니다.")
        else:
            messages.append("겨울-여름 비교는 두 지표가 함께 강해지는 전형적 패턴을 명확히 보이지 않았습니다.")
    return messages


def eddy_interpretation(data: pd.DataFrame, selected_level: int = 850) -> list[str]:
    """양의/음의 에디 가속도 위치를 실제 배열에서 요약한다."""
    if data.empty or not {"pressure_hpa", "eddy_acceleration_ms_day"}.issubset(data.columns):
        return ["에디 자료가 없어 평균 흐름에 대한 방향을 진단하지 않았습니다."]
    selected = data[data["pressure_hpa"].eq(selected_level)].dropna(subset=["eddy_acceleration_ms_day"])
    if selected.empty:
        return ["에디 자료가 없어 평균 흐름에 대한 방향을 진단하지 않았습니다."]
    maximum = selected.loc[selected["eddy_acceleration_ms_day"].idxmax()]
    minimum = selected.loc[selected["eddy_acceleration_ms_day"].idxmin()]
    return [
        f"{selected_level} hPa에서 가장 큰 양의 진단값은 {maximum['latitude_deg']:.1f}°N의 "
        f"{maximum['eddy_acceleration_ms_day']:+.3f} m/s/day입니다. 양수는 평균 동쪽 방향 흐름을 강화하는 방향입니다.",
        f"가장 큰 음의 진단값은 {minimum['latitude_deg']:.1f}°N의 {minimum['eddy_acceleration_ms_day']:+.3f} m/s/day입니다.",
        "이는 에디 운동량 플럭스 수렴 항만의 진단이며 지상 마찰, 산악 토크, 평균 자오면 순환 등 전체 운동량 수지는 포함하지 않습니다.",
    ]


def build_summary_markdown(
    vertical: pd.DataFrame | None = None,
    matched: pd.DataFrame | None = None,
    thermal: pd.DataFrame | None = None,
    eddy: pd.DataFrame | None = None,
) -> str:
    """보고서에 붙일 수 있는 결과 기반 Markdown 요약문을 만든다."""
    vertical = pd.DataFrame() if vertical is None else vertical
    matched = pd.DataFrame() if matched is None else matched
    thermal = pd.DataFrame() if thermal is None else thermal
    eddy = pd.DataFrame() if eddy is None else eddy
    lines = [
        "# WesterlyScope 분석 결과 요약",
        "",
        "## 1. 상층 편서풍의 연직 구조와 지표 비교",
        *[f"- {message}" for message in observational_interpretation(vertical, matched)],
        "",
        "## 2. 열풍 관계",
        *[f"- {message}" for message in thermal_interpretation(thermal)],
        "",
        "## 3. 에디 운동량 수송",
        *[f"- {message}" for message in eddy_interpretation(eddy)],
        "",
        "## 종합 결론",
    ]
    vertical_support = False
    if not vertical.empty and {"pressure_hpa", "mean_u_ms"}.issubset(vertical.columns):
        indexed = vertical.set_index("pressure_hpa")
        vertical_support = {850, 300}.issubset(indexed.index) and float(indexed.loc[300, "mean_u_ms"]) > float(indexed.loc[850, "mean_u_ms"])
    thermal_metrics = regression_metrics(thermal.get("predicted_shear_ms", pd.Series(dtype=float)), thermal.get("observed_shear_ms", pd.Series(dtype=float)))
    thermal_support = (
        np.isfinite(thermal_metrics.correlation)
        and thermal_metrics.correlation >= 0.3
        and thermal_metrics.sign_match_rate >= 0.6
    )
    support = vertical_support and thermal_support and not matched.empty and not eddy.empty
    if support:
        lines.append(
            "상층 편서풍과 지상 서풍대는 서로 관련되어 있지만 같은 단일 과정의 결과로 볼 수 없다. "
            "상층에서는 남북 기온 경도와 열풍 관계가 편서풍의 연직 강화와 연결되고, 지표에서는 "
            "마찰과 이동성 교란 및 에디 운동량 수송을 포함한 운동량 수지가 중요하다."
        )
    else:
        lines.append(
            "이론적으로는 상층의 열풍 관계와 지표의 복합 운동량 수지가 서로 다른 역할을 하지만, "
            "현재 선택 자료에서는 모든 증거가 계산되지 않아 명확히 확인되지 않음."
        )
    lines.extend([
        "",
        "이 분석은 관측·재분석의 통계적 진단이며 단일 관측소나 특정 한 해 결과만으로 전 지구 순환의 인과관계를 확정하지 않는다.",
    ])
    return "\n".join(lines)


def combined_metrics(
    vertical: pd.DataFrame | None,
    matched: pd.DataFrame | None,
    thermal: pd.DataFrame | None,
    eddy: pd.DataFrame | None,
) -> pd.DataFrame:
    """각 탭의 핵심 수치를 이름-값 표로 결합한다."""
    metrics: list[Mapping[str, object]] = []
    if vertical is not None and not vertical.empty:
        indexed = vertical.set_index("pressure_hpa")
        if {850, 300}.issubset(indexed.index):
            metrics.append({"metric": "mean_u_300_minus_850_ms", "value": indexed.loc[300, "mean_u_ms"] - indexed.loc[850, "mean_u_ms"]})
    if matched is not None and not matched.empty:
        metrics.extend([
            {"metric": "surface_westerly_ratio", "value": (matched["u_surface_ms"].dropna() > 0).mean()},
            {"metric": "surface_850_u_correlation", "value": matched["u_surface_ms"].corr(matched["u_850_ms"])},
        ])
    if thermal is not None and not thermal.empty:
        values = regression_metrics(thermal["predicted_shear_ms"], thermal["observed_shear_ms"])
        metrics.extend([
            {"metric": "thermal_wind_correlation", "value": values.correlation},
            {"metric": "thermal_wind_rmse_ms", "value": values.rmse},
            {"metric": "thermal_wind_sign_match_rate", "value": values.sign_match_rate},
        ])
    if eddy is not None and not eddy.empty:
        maximum = eddy.loc[eddy["eddy_acceleration_ms_day"].idxmax()]
        metrics.extend([
            {"metric": "max_eddy_acceleration_ms_day", "value": maximum["eddy_acceleration_ms_day"]},
            {"metric": "max_eddy_acceleration_latitude_deg", "value": maximum["latitude_deg"]},
        ])
    return pd.DataFrame(metrics)
