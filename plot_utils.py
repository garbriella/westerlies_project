"""WesterlyScope의 Plotly 기반 시각화 함수."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

NAVY = "#15324b"
BLUE = "#2878b5"
TEAL = "#1c9a91"
ORANGE = "#e58b3a"
PALE = "#eef5f9"

PLOTLY_CONFIG = {
    "responsive": True,
    "displaylogo": False,
    "scrollZoom": False,
}


def style_figure(figure: go.Figure, height: int = 430) -> go.Figure:
    """모든 그래프에 일관된 반응형 스타일을 적용한다."""
    figure.update_layout(
        height=height,
        margin=dict(l=45, r=25, t=65, b=45),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Arial, Apple SD Gothic Neo, Malgun Gothic, sans-serif", color=NAVY),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="closest",
    )
    figure.update_xaxes(showgrid=True, gridcolor="#e8eef2", zerolinecolor="#aabac6")
    figure.update_yaxes(showgrid=True, gridcolor="#e8eef2", zerolinecolor="#aabac6")
    return figure


def concept_figure() -> go.Figure:
    """열풍, 지표 변동, 에디 수송을 연결한 설명용 개념 도식."""
    figure = go.Figure()
    figure.add_shape(type="rect", x0=0, x1=10, y0=0, y1=2, fillcolor="#d8eee8", line_width=0)
    figure.add_shape(type="rect", x0=0, x1=4.8, y0=2, y1=8, fillcolor="#fff1df", line_width=0)
    figure.add_shape(type="rect", x0=5.2, x1=10, y0=2, y1=8, fillcolor="#deedf8", line_width=0)
    figure.add_annotation(x=2, y=7.4, text="남쪽: 따뜻함", showarrow=False, font=dict(size=16, color=ORANGE))
    figure.add_annotation(x=8, y=7.4, text="북쪽: 차가움", showarrow=False, font=dict(size=16, color=BLUE))
    for y, length in ((2.7, 1.6), (4.2, 2.7), (5.8, 4.0)):
        figure.add_annotation(x=5 - length / 2, y=y, ax=-(length * 30), ay=0, text="", showarrow=True,
                              arrowhead=3, arrowsize=1.2, arrowwidth=5, arrowcolor=BLUE)
    figure.add_annotation(x=5, y=6.5, text="고도가 높을수록 강한 동쪽 방향(u>0)", showarrow=False)
    figure.add_annotation(x=2.1, y=1.1, text="지표: 마찰·지형·이동성 교란으로 약하고 변동", showarrow=False, font=dict(color=TEAL))
    figure.add_annotation(x=7.8, y=1.1, text="에디의 동서 운동량 수송", showarrow=True, ax=-80, ay=0,
                          arrowcolor=ORANGE, font=dict(color=ORANGE))
    figure.update_xaxes(visible=False, range=[0, 10])
    figure.update_yaxes(visible=False, range=[0, 8])
    figure.update_layout(title="상층 편서풍과 지상 서풍대의 연결(설명용 도식)")
    return style_figure(figure, 460)


def pressure_profile(data: pd.DataFrame, x: str, title: str, x_title: str, color: str = BLUE) -> go.Figure:
    """기압축을 뒤집은 수직 프로파일을 그린다."""
    tooltip_columns = ["valid_u_count", "observed_u_count", "interpolated_u_count", "interpolated_u_ratio"]
    has_sample_metadata = set(tooltip_columns).issubset(data.columns)
    customdata = data[tooltip_columns].to_numpy() if has_sample_metadata else None
    hovertemplate = None
    if has_sample_metadata:
        hovertemplate = (
            f"{x_title}: %{{x:.2f}}<br>기압: %{{y:.0f}} hPa"
            "<br>유효 표본 수: %{customdata[0]:.0f}"
            "<br>실제 관측값 수: %{customdata[1]:.0f}"
            "<br>보간값 수: %{customdata[2]:.0f}"
            "<br>보간값 비율: %{customdata[3]:.1%}<extra></extra>"
        )
    figure = go.Figure(go.Scatter(
        x=data[x],
        y=data["pressure_hpa"],
        mode="lines+markers",
        line=dict(color=color, width=3),
        customdata=customdata,
        hovertemplate=hovertemplate,
    ))
    figure.update_layout(title=title, xaxis_title=x_title, yaxis_title="기압 (hPa)")
    figure.update_yaxes(autorange="reversed", type="log")
    return style_figure(figure)


def seasonal_profiles(standard: pd.DataFrame) -> go.Figure:
    """계절별 평균 u 수직 프로파일."""
    grouped = standard.groupby(["season", "pressure_hpa"], as_index=False, observed=True)["u_ms"].mean()
    figure = px.line(grouped, x="u_ms", y="pressure_hpa", color="season", markers=True,
                     category_orders={"season": ["겨울", "봄", "여름", "가을"]},
                     title="계절별 평균 서풍 성분", labels={"u_ms": "u (m/s)", "pressure_hpa": "기압 (hPa)", "season": "계절"})
    figure.update_yaxes(autorange="reversed", type="log")
    return style_figure(figure)


def sounding_temperature_wind(profile: pd.DataFrame) -> go.Figure:
    """개별 sounding의 기온과 u를 병렬 표시한다."""
    figure = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("기온 수직 분포", "u 수직 분포"))
    figure.add_trace(go.Scatter(x=profile["temperature_c"], y=profile["pressure_hpa"], mode="lines+markers", name="기온", line=dict(color=ORANGE)), 1, 1)
    figure.add_trace(go.Scatter(x=profile["u_ms"], y=profile["pressure_hpa"], mode="lines+markers", name="u", line=dict(color=BLUE)), 1, 2)
    figure.update_xaxes(title_text="기온 (°C)", row=1, col=1)
    figure.update_xaxes(title_text="u (m/s)", row=1, col=2)
    figure.update_yaxes(title_text="기압 (hPa)", autorange="reversed", type="log", row=1, col=1)
    return style_figure(figure, 500)


def wind_arrow_profile(profile: pd.DataFrame) -> go.Figure:
    """기압면별 u/v 화살표로 바람 방향을 표현한다."""
    valid = profile.dropna(subset=["u_ms", "v_ms", "pressure_hpa"])
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=np.zeros(len(valid)), y=valid["pressure_hpa"], mode="markers", marker=dict(color=NAVY), name="기압면"))
    for _, row in valid.iterrows():
        figure.add_annotation(x=float(row["u_ms"]), y=float(row["pressure_hpa"]), ax=0, ay=float(row["pressure_hpa"]),
                              xref="x", yref="y", axref="x", ayref="y", text="", showarrow=True,
                              arrowhead=3, arrowwidth=2, arrowcolor=TEAL)
    figure.update_layout(title="기압면별 바람 성분 화살표", xaxis_title="u 방향 길이 (m/s)", yaxis_title="기압 (hPa)")
    figure.update_yaxes(autorange="reversed", type="log")
    return style_figure(figure)


def matched_time_series(matched: pd.DataFrame) -> go.Figure:
    """대응 시각별 지표 및 상층 u 선그래프."""
    columns = [column for column in ("u_surface_ms", "u_850_ms", "u_500_ms", "u_300_ms") if column in matched]
    long = matched.melt(id_vars="datetime_utc", value_vars=columns, var_name="고도", value_name="u_ms")
    labels = {"u_surface_ms": "지표", "u_850_ms": "850 hPa", "u_500_ms": "500 hPa", "u_300_ms": "300 hPa"}
    long["고도"] = long["고도"].map(labels)
    figure = px.line(long, x="datetime_utc", y="u_ms", color="고도", title="대응 시각의 고도별 u",
                     labels={"datetime_utc": "UTC 시각", "u_ms": "u (m/s)"})
    return style_figure(figure, 470)


def surface_scatter(matched: pd.DataFrame, upper_level: int) -> go.Figure:
    """지표 u와 선택 상층 u 산점도."""
    y_column = f"u_{upper_level}_ms"
    figure = px.scatter(matched, x="u_surface_ms", y=y_column, color="season",
                        title=f"지표 u와 {upper_level} hPa u", labels={"u_surface_ms": "지표 u (m/s)", y_column: f"{upper_level} hPa u (m/s)", "season": "계절"})
    return style_figure(figure)


def seasonal_box(matched: pd.DataFrame) -> go.Figure:
    """계절별 지표·상층 u 상자그림."""
    columns = [column for column in ("u_surface_ms", "u_850_ms", "u_500_ms", "u_300_ms") if column in matched]
    long = matched.melt(id_vars="season", value_vars=columns, var_name="고도", value_name="u_ms")
    figure = px.box(long, x="season", y="u_ms", color="고도", points=False, title="계절별 u 분포",
                    category_orders={"season": ["겨울", "봄", "여름", "가을"]}, labels={"season": "계절", "u_ms": "u (m/s)"})
    return style_figure(figure)


def direction_frequency(matched: pd.DataFrame) -> go.Figure:
    """지표 풍향을 16방위 빈도로 집계한 극좌표 그래프."""
    directions = pd.to_numeric(matched.get("wind_direction_deg"), errors="coerce").dropna()
    bins = np.arange(-11.25, 371.25, 22.5)
    labels = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    shifted = (directions + 11.25) % 360
    indexes = (shifted // 22.5).astype(int) % 16
    counts = indexes.value_counts().reindex(range(16), fill_value=0)
    figure = go.Figure(go.Barpolar(theta=labels, r=counts.values, marker_color=TEAL))
    figure.update_layout(title="지표 풍향 빈도(관측 횟수)", polar=dict(angularaxis=dict(direction="clockwise", rotation=90)))
    return style_figure(figure)


def westerly_easterly_stack(stats: pd.DataFrame) -> go.Figure:
    """고도별 서풍/동풍 발생 비율 누적 막대."""
    frame = stats.copy()
    frame["동풍 또는 무풍 비율"] = 1 - frame["westerly_ratio"]
    figure = go.Figure()
    figure.add_bar(x=frame["level"], y=frame["westerly_ratio"] * 100, name="서풍 성분(u>0)", marker_color=BLUE)
    figure.add_bar(x=frame["level"], y=frame["동풍 또는 무풍 비율"] * 100, name="u≤0", marker_color="#c8d5dd")
    figure.update_layout(barmode="stack", title="고도별 서풍·동풍 성분 발생 비율", xaxis_title="고도", yaxis_title="비율 (%)")
    return style_figure(figure)


def station_map(stations: pd.DataFrame) -> go.Figure:
    """북쪽·남쪽 관측소 위치 지도."""
    figure = px.scatter_map(stations, lat="latitude", lon="longitude", color="role", hover_name="station_name",
                            hover_data=["station_id"], zoom=5, height=430, map_style="open-street-map",
                            title="열풍 분석 관측소 위치")
    figure.update_layout(margin=dict(l=0, r=0, t=55, b=0))
    return figure


def thermal_wind_charts(results: pd.DataFrame) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    """열풍 시계열, 산점도, 계절 비교, 기온경도 관계 그래프 묶음."""
    time_long = results.melt(id_vars="datetime_utc", value_vars=["predicted_shear_ms", "observed_shear_ms"],
                             var_name="구분", value_name="shear_ms")
    time_long["구분"] = time_long["구분"].map({"predicted_shear_ms": "열풍 예측", "observed_shear_ms": "관측"})
    timeline = px.line(time_long, x="datetime_utc", y="shear_ms", color="구분", title="예측·관측 시어 시계열",
                       labels={"datetime_utc": "UTC 시각", "shear_ms": "Δu (m/s)"})
    scatter = px.scatter(results, x="predicted_shear_ms", y="observed_shear_ms", color="season",
                         title="열풍 예측값 대 관측값", labels={"predicted_shear_ms": "예측 Δu (m/s)", "observed_shear_ms": "관측 Δu (m/s)"})
    seasonal = results.groupby("season", as_index=False, observed=True)[
        ["predicted_shear_ms", "observed_shear_ms"]
    ].mean().melt(id_vars="season", var_name="구분", value_name="shear_ms")
    season_plot = px.bar(seasonal, x="season", y="shear_ms", color="구분", barmode="group", title="계절별 예측·관측 시어",
                         category_orders={"season": ["겨울", "봄", "여름", "가을"]}, labels={"season": "계절", "shear_ms": "Δu (m/s)"})
    gradient = px.scatter(results, x="temperature_gradient_k_per_1000km", y="observed_shear_ms", color="season",
                          title="남북 기온 경도와 관측 시어", labels={"temperature_gradient_k_per_1000km": "기온 경도 (K/1000 km)", "observed_shear_ms": "관측 Δu (m/s)"})
    return tuple(style_figure(fig) for fig in (timeline, scatter, season_plot, gradient))  # type: ignore[return-value]


def eddy_charts(data: pd.DataFrame) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure]:
    """기압면별 평균 u, 에디 플럭스, 가속도와 병렬 비교 그래프."""
    color = data["pressure_hpa"].astype(int).astype(str) + " hPa"
    figures = []
    for y, title, label in (
        ("mean_u_ms", "위도별 평균 u", "평균 u (m/s)"),
        ("eddy_flux_m2s2", "위도별 에디 운동량 플럭스", "u'v' (m²/s²)"),
        ("eddy_acceleration_ms_day", "위도별 에디 가속도 진단", "A_eddy (m/s/day)"),
    ):
        fig = px.line(data.assign(기압면=color), x="latitude_deg", y=y, color="기압면", title=title,
                      labels={"latitude_deg": "위도 (°N)", y: label})
        figures.append(style_figure(fig))
    comparison = make_subplots(rows=1, cols=2, subplot_titles=("평균 u", "에디 가속도"))
    for level, group in data.groupby("pressure_hpa"):
        comparison.add_trace(go.Scatter(x=group["latitude_deg"], y=group["mean_u_ms"], name=f"{int(level)} hPa"), 1, 1)
        comparison.add_trace(go.Scatter(x=group["latitude_deg"], y=group["eddy_acceleration_ms_day"], name=f"{int(level)} hPa", showlegend=False), 1, 2)
    comparison.update_xaxes(title_text="위도 (°N)")
    comparison.update_yaxes(title_text="u (m/s)", row=1, col=1)
    comparison.update_yaxes(title_text="m/s/day", row=1, col=2)
    comparison.update_layout(title="평균 흐름과 에디 가속도 진단 병렬 비교")
    figures.append(style_figure(comparison, 450))
    return tuple(figures)  # type: ignore[return-value]
