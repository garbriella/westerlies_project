"""WesterlyScope Streamlit 애플리케이션 진입점."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import traceback

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from igra_utils import (
    cached_standard_levels,
    discover_igra_files,
    download_igra_archive,
    download_igra_station_list,
    korean_igra_stations,
    parse_igra_archive,
)
from isd_utils import (
    download_isd_history,
    download_isd_period,
    find_isd_candidates,
    match_surface_to_standard_levels,
)
from plot_utils import (
    PLOTLY_CONFIG,
    concept_figure,
    direction_frequency,
    eddy_charts,
    matched_time_series,
    pressure_profile,
    seasonal_box,
    seasonal_profiles,
    sounding_temperature_wind,
    station_map,
    surface_scatter,
    thermal_wind_charts,
    wind_arrow_profile,
    westerly_easterly_stack,
)
from reanalysis_utils import (
    available_reanalysis_years,
    discover_reanalysis_files,
    latest_complete_year,
    load_and_calculate_eddy,
)
from report_utils import (
    build_summary_markdown,
    combined_metrics,
    dataframe_csv,
    eddy_interpretation,
    observational_interpretation,
    thermal_interpretation,
)
from wind_analysis import (
    STANDARD_PRESSURES,
    paired_thermal_wind,
    regression_metrics,
    surface_upper_statistics,
    vertical_wind_statistics,
)


st.set_page_config(
    page_title="WesterlyScope | 상층 편서풍과 지상 서풍대 분석기",
    page_icon="〰",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --navy:#15324b; --blue:#2878b5; --teal:#1c9a91; --pale:#eef5f9; }
    .stApp { background: linear-gradient(180deg, #f5f9fc 0%, #ffffff 420px); color: var(--navy); }
    h1, h2, h3 { color: var(--navy); letter-spacing: -0.02em; }
    div[data-testid="stMetric"] { background:#fff; border:1px solid #dbe7ee; border-radius:16px; padding:16px; box-shadow:0 5px 18px rgba(21,50,75,.05); }
    div[data-testid="stAlert"] { border-radius:14px; }
    .hero { padding:24px 28px; border-radius:22px; color:white; margin-bottom:18px;
            background:linear-gradient(125deg,#15324b,#245f87 58%,#1c9a91); box-shadow:0 12px 34px rgba(21,50,75,.18); }
    .hero h1 { color:white; margin:0 0 8px; font-size:2.05rem; }
    .hero p { color:#eaf4f8; margin:0; font-size:1.04rem; line-height:1.7; }
    .note-card { background:white; border:1px solid #dbe7ee; border-left:5px solid #2878b5; border-radius:14px; padding:15px 18px; margin:8px 0 14px; }
    .small-muted { color:#5d7383; font-size:.88rem; }
    @media(max-width:700px) { .hero { padding:20px; } .hero h1 { font-size:1.6rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def show_exception(message: str, exc: Exception) -> None:
    """사용자 메시지와 분리된 기술 오류를 표시한다."""
    st.error(message)
    with st.expander("기술적 오류 내용"):
        st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


def station_label(station_id: str, stations: pd.DataFrame) -> str:
    """관측소 선택 상자용 이름을 만든다."""
    match = stations[stations["station_id"].eq(station_id)]
    if match.empty:
        return station_id
    row = match.iloc[0]
    return f"{row['station_name']} · {station_id}"


def load_igra_station(row: pd.Series, start_date: date, end_date: date) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """POR 우선·Y2D 대체 순서로 선택 IGRA 관측소 자료를 불러온다."""
    listed_start = date(int(row["first_year"]), 1, 1)
    listed_end = date(int(row["last_year"]), 12, 31)
    if end_date < listed_start or start_date > listed_end:
        raise ValueError(
            f"{row['station_name']}({row['station_id']})의 IGRA 목록상 자료기간은 "
            f"{row['first_year']}–{row['last_year']}년이며, 요청 기간 "
            f"{start_date}–{end_date}과 겹치지 않습니다."
        )
    candidates = discover_igra_files(str(row["station_id"]))
    errors: list[str] = []
    for candidate in candidates:
        try:
            archive = download_igra_archive(candidate["url"])
            levels = parse_igra_archive(
                archive, candidate["filename"], str(row["station_name"]), str(start_date), str(end_date)
            )
            if levels.empty:
                errors.append(f"{candidate['source']}: 선택 기간 sounding 없음")
                continue
            standard = cached_standard_levels(levels, STANDARD_PRESSURES)
            return levels, standard, f"{candidate['source']} · {candidate['filename']}"
        except Exception as exc:  # 개별 후보 실패 후 다음 공식 디렉터리 후보 시도
            errors.append(f"{candidate['source']}: {exc}")
    raise RuntimeError(
        f"{row['station_name']}({row['station_id']}) 자료를 읽지 못했습니다. "
        f"요청 기간: {start_date}–{end_date}, IGRA 목록상 기간: "
        f"{row['first_year']}–{row['last_year']}년. " + " | ".join(errors)
    )


def thermal_station_defaults(stations: pd.DataFrame, start_date: date, end_date: date) -> tuple[str, str]:
    """선택 기간을 덮는 대한민국 IGRA 관측소 중 최북·최남 기본 ID를 고른다."""
    full_period = stations[
        (stations["first_year"] <= start_date.year)
        & (stations["last_year"] >= end_date.year)
        & (stations["observation_count"] > 0)
    ]
    candidates = full_period
    if len(candidates) < 2:
        candidates = stations[
            (stations["first_year"] <= end_date.year)
            & (stations["last_year"] >= start_date.year)
            & (stations["observation_count"] > 0)
        ]
    if len(candidates) < 2:
        candidates = stations
    north_id = str(candidates.loc[candidates["latitude"].idxmax(), "station_id"])
    south_id = str(candidates.loc[candidates["latitude"].idxmin(), "station_id"])
    return north_id, south_id


def filter_season(data: pd.DataFrame, season: str) -> pd.DataFrame:
    """전체 또는 선택 계절만 남긴다."""
    if data.empty or season == "전체":
        return data
    return data[data["season"].eq(season)].copy()


def metric_value(value: float, unit: str = "", digits: int = 2) -> str:
    """결측에 안전한 지표 문자열."""
    return "자료 없음" if not np.isfinite(value) else f"{value:.{digits}f}{unit}"


for key, default in {
    "igra_levels": pd.DataFrame(),
    "standard_levels": pd.DataFrame(),
    "isd_candidates": pd.DataFrame(),
    "surface_data": pd.DataFrame(),
    "matched_data": pd.DataFrame(),
    "thermal_results": pd.DataFrame(),
    "eddy_results": pd.DataFrame(),
    "comments": [],
    "igra_source": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


station_error: Exception | None = None
try:
    all_stations = download_igra_station_list()
    ks_stations = korean_igra_stations(all_stations)
except Exception as exc:
    station_error = exc
    ks_stations = pd.DataFrame()

st.markdown(
    """
    <div class="hero">
      <h1>WesterlyScope | 상층 편서풍과 지상 서풍대 분석기</h1>
      <p>NOAA 고층·지상 관측과 재분석 자료를 이용하여 상층 편서풍의 연직 강화와 중위도 지상 서풍대의 유지 원리를 분석합니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if station_error is not None:
    show_exception("NOAA IGRA 관측소 목록에 연결하지 못했습니다. 연구 개요와 방법 탭은 계속 볼 수 있습니다.", station_error)


with st.sidebar:
    st.header("분석 설정")
    if not ks_stations.empty:
        ids = ks_stations["station_id"].tolist()
        default_id = "KSM00047138" if "KSM00047138" in ids else ids[0]
        selected_id = st.selectbox(
            "IGRA 관측소",
            ids,
            index=ids.index(default_id),
            format_func=lambda value: station_label(value, ks_stations),
        )
        station_row = ks_stations[ks_stations["station_id"].eq(selected_id)].iloc[0]
        st.caption(f"{station_row['station_name']} · {station_row['station_id']}")
        st.write(f"위도 {station_row['latitude']:.4f}°, 경도 {station_row['longitude']:.4f}°, 고도 {station_row['elevation_m']:.1f} m")
        st.map(pd.DataFrame({"lat": [station_row["latitude"]], "lon": [station_row["longitude"]]}), zoom=5, height=180)

        last_available = min(date.today(), date(int(station_row["last_year"]), 12, 31))
        first_available = date(int(station_row["first_year"]), 1, 1)
        default_start = max(first_available, date(last_available.year - 3, last_available.month, min(last_available.day, 28)))
        start_date = st.date_input("시작일", value=default_start, min_value=first_available, max_value=last_available)
        end_date = st.date_input("종료일", value=last_available, min_value=first_available, max_value=last_available)
        season = st.selectbox("계절", ["전체", "겨울", "봄", "여름", "가을"])
        selected_pressures = st.multiselect("표준기압면 (hPa)", STANDARD_PRESSURES, default=list(STANDARD_PRESSURES))
        tolerance = st.slider("ISD 대응 허용 범위 (분)", 30, 180, 90, 15)

        st.subheader("열풍 관계 관측소")
        north_default, south_default = thermal_station_defaults(ks_stations, start_date, end_date)
        north_id = st.selectbox(
            "북쪽 관측소",
            ids,
            index=ids.index(north_default),
            format_func=lambda value: station_label(value, ks_stations),
            key=f"thermal_north_{start_date}_{end_date}",
        )
        south_id = st.selectbox(
            "남쪽 관측소",
            ids,
            index=ids.index(south_default),
            format_func=lambda value: station_label(value, ks_stations),
            key=f"thermal_south_{start_date}_{end_date}",
        )
        lower_options = STANDARD_PRESSURES[:-1]
        lower_pressure = st.selectbox("열풍 하층 기압", lower_options, index=list(lower_options).index(850))
        upper_options = [pressure for pressure in STANDARD_PRESSURES if pressure < lower_pressure]
        upper_default = 300 if 300 in upper_options else upper_options[0]
        upper_pressure = st.selectbox("열풍 상층 기압", upper_options, index=upper_options.index(upper_default))

        if st.button("IGRA 고층자료 불러오기", type="primary", width="stretch"):
            if start_date > end_date:
                st.error("시작일은 종료일보다 늦을 수 없습니다.")
            else:
                try:
                    with st.spinner("IGRA 전체기간 자료를 우선 확인하고 있습니다..."):
                        levels, standard, source = load_igra_station(station_row, start_date, end_date)
                        history = download_isd_history()
                        candidates = find_isd_candidates(
                            history, float(station_row["latitude"]), float(station_row["longitude"]), str(station_row["station_name"])
                        )
                    st.session_state.igra_levels = levels
                    st.session_state.standard_levels = standard
                    st.session_state.igra_source = source
                    st.session_state.isd_candidates = candidates
                    st.session_state.matched_data = pd.DataFrame()
                    st.success(f"sounding {levels['datetime_utc'].nunique():,}개를 불러왔습니다.")
                except Exception as exc:
                    show_exception("IGRA 자료를 불러오지 못했습니다. 기간과 관측소를 확인해 주세요.", exc)

        candidates = st.session_state.isd_candidates
        if isinstance(candidates, pd.DataFrame) and not candidates.empty:
            st.subheader("ISD 지상 관측소")
            candidate_keys = candidates["station_key"].tolist()
            isd_key = st.selectbox(
                "대응 후보",
                candidate_keys,
                format_func=lambda key: (
                    f"{candidates.loc[candidates['station_key'].eq(key), 'station_name'].iloc[0]} · "
                    f"{candidates.loc[candidates['station_key'].eq(key), 'distance_km'].iloc[0]:.1f} km"
                ),
            )
            isd_row = candidates[candidates["station_key"].eq(isd_key)].iloc[0]
            if float(isd_row["distance_km"]) > 30:
                st.warning("30 km 이내의 자동 대응 후보가 아닙니다. 거리와 관측소명을 확인한 뒤 직접 선택하세요.")
            else:
                st.caption(f"30 km 기준 충족 · 거리 {isd_row['distance_km']:.1f} km")
            if st.button("ISD 지상자료 불러오기", width="stretch"):
                try:
                    with st.spinner("선택 기간의 ISD 연도별 파일만 불러오고 있습니다..."):
                        surface, missing_years = download_isd_period(isd_row["USAF"], isd_row["WBAN"], str(start_date), str(end_date))
                        matched = match_surface_to_standard_levels(st.session_state.standard_levels, surface, tolerance)
                    st.session_state.surface_data = surface
                    st.session_state.matched_data = matched
                    if missing_years:
                        st.warning("ISD 파일이 없었던 연도: " + ", ".join(map(str, missing_years)))
                    if matched.empty:
                        st.warning("허용 시간 범위 안에서 대응되는 지상·상층 관측이 없습니다.")
                    else:
                        st.success(f"지상·상층 관측 {len(matched):,}쌍을 대응했습니다.")
                except Exception as exc:
                    show_exception("ISD 지상 자료를 불러오거나 시간 대응하지 못했습니다.", exc)

        if st.button("자료 새로고침", width="stretch"):
            st.cache_data.clear()
            cache_dir = Path("data_cache")
            if cache_dir.is_dir():
                for cached_file in cache_dir.glob("KSM*-data*.txt.zip"):
                    cached_file.unlink(missing_ok=True)
            for key in ("igra_levels", "standard_levels", "isd_candidates", "surface_data", "matched_data", "thermal_results", "eddy_results"):
                st.session_state[key] = pd.DataFrame()
            st.rerun()

        st.divider()
        st.caption("UTC는 국제 표준시이며 KST는 UTC+9입니다. 자료 대응은 UTC로 계산하고 화면에는 두 시각을 함께 표시합니다.")
        with st.expander("분석 기준 도움말"):
            st.write("u > 0은 공기가 동쪽으로 이동하는 서풍 성분입니다. 풍향만으로 편서풍 여부를 판정하지 않습니다.")
            st.write("장기간을 선택하면 IGRA 파싱과 여러 ISD 연도 파일 처리 시간이 늘어납니다. 기본값은 최근 약 3년입니다.")
    else:
        station_row = pd.Series(dtype=object)
        selected_id = ""
        start_date = end_date = date.today()
        season = "전체"
        selected_pressures = list(STANDARD_PRESSURES)
        tolerance = 90
        north_id = south_id = ""
        lower_pressure, upper_pressure = 850, 300
        st.info("관측소 목록을 사용할 수 없어 관측 설정을 표시할 수 없습니다.")


standard_all = st.session_state.standard_levels
standard_view = filter_season(standard_all, season)
if not selected_pressures:
    standard_view = pd.DataFrame()
elif not standard_view.empty:
    standard_view = standard_view[standard_view["pressure_hpa"].isin(selected_pressures)]
vertical_stats = vertical_wind_statistics(standard_view)
matched_all = st.session_state.matched_data
matched_view = filter_season(matched_all, season)

tabs = st.tabs([
    "1. 연구 개요",
    "2. 고도별 서풍 구조",
    "3. 지상과 상층 비교",
    "4. 열풍 관계 검증",
    "5. 에디 운동량 수송",
    "6. 종합 분석",
    "7. 자료와 방법",
])


with tabs[0]:
    st.subheader("연구 질문")
    questions = [
        "중위도에서는 고도가 높아질수록 서풍 성분이 실제로 강해지는가?",
        "남북 기온 차로 계산한 열풍 관계의 예측값은 관측된 연직 서풍 시어와 일치하는가?",
        "지표의 서풍은 상층 서풍보다 약하고 변동성이 큰가?",
        "중위도 에디의 동서 방향 운동량 수송은 평균 서풍 유지에 어떤 방향으로 작용하는가?",
        "상층 편서풍과 지상 서풍대를 같은 형성 원리로 설명할 수 있는가?",
    ]
    for number, question in enumerate(questions, 1):
        st.write(f"{number}. {question}")
    st.plotly_chart(concept_figure(), width="stretch", config=PLOTLY_CONFIG)
    st.caption("이 그림은 실제 관측 결과가 아니라 분석 개념을 설명하기 위한 도식입니다.")
    overview_cols = st.columns(3)
    with overview_cols[0]:
        st.markdown("### 상층 편서풍")
        st.write("남북 기온 경도는 열풍 관계를 통해 기압면 사이의 지균풍 차이와 연결됩니다. 열풍은 새로운 독립 바람이 아니라 연직 지균풍 시어입니다.")
    with overview_cols[1]:
        st.markdown("### 지상 서풍대")
        st.write("지표에서는 마찰, 지형, 해륙풍과 이동성 고·저기압 때문에 순간 풍향이 크게 변합니다. 서풍대는 장기 평균적인 u>0 성분을 뜻합니다.")
    with overview_cols[2]:
        st.markdown("### 에디 운동량 수송")
        st.write("이동성 교란의 u'v' 수렴은 평균 동서풍을 강화하거나 약화하는 한 항입니다. 전체 운동량 수지나 지표 풍향을 단독으로 결정하지 않습니다.")
    st.info("u > 0: 동쪽으로 이동하는 서풍 성분 · u < 0: 서쪽으로 이동하는 동풍 성분 · v > 0: 북쪽으로 이동하는 남풍 성분 · v < 0: 남쪽으로 이동하는 북풍 성분")
    st.markdown("**자료와 분석 흐름:** IGRA 라디오존데 → 표준기압면 보간·연직 구조 → ISD 최근접 시간 대응 → 두 관측소 열풍 검증 → NCEP/DOE R2 에디 진단 → 조건부 종합 해석")


with tabs[1]:
    st.subheader("IGRA 고도별 서풍 구조")
    if standard_view.empty:
        st.info("사이드바에서 기간과 관측소를 정한 뒤 ‘IGRA 고층자료 불러오기’를 누르세요. 자료가 없을 때 임의 결과를 만들지 않습니다.")
    else:
        raw_view = filter_season(st.session_state.igra_levels, season)
        metrics = st.columns(5)
        metrics[0].metric("선택 라디오존데 관측 횟수", f"{standard_view['datetime_utc'].nunique():,}개")
        metrics[1].metric("유효 기압면별 u 자료 수", f"{standard_view['u_ms'].count():,}개")
        for column, pressure in zip(metrics[2:], (850, 500, 300)):
            value = vertical_stats.loc[vertical_stats["pressure_hpa"].eq(pressure), "mean_u_ms"]
            column.metric(f"{pressure} hPa 평균 u", metric_value(float(value.iloc[0]) if not value.empty else np.nan, " m/s"))
        st.caption(f"자료 출처: {st.session_state.igra_source} · 실제 레벨과 보간 레벨은 다운로드 표의 is_interpolated 열로 구분됩니다.")
        col1, col2, col3 = st.columns(3)
        col1.plotly_chart(pressure_profile(vertical_stats, "mean_u_ms", "기압면별 평균 u", "u (m/s)"), width="stretch", config=PLOTLY_CONFIG)
        col2.plotly_chart(pressure_profile(vertical_stats, "mean_speed_ms", "기압면별 평균 풍속", "풍속 (m/s)", "#1c9a91"), width="stretch", config=PLOTLY_CONFIG)
        col3.plotly_chart(pressure_profile(vertical_stats, "westerly_ratio", "기압면별 서풍 발생 비율", "u>0 비율", "#e58b3a"), width="stretch", config=PLOTLY_CONFIG)
        st.plotly_chart(seasonal_profiles(standard_all), width="stretch", config=PLOTLY_CONFIG)
        st.dataframe(
            vertical_stats[[
                "pressure_hpa", "valid_u_count", "observed_u_count",
                "interpolated_u_count", "interpolated_u_ratio",
            ]].rename(columns={
                "pressure_hpa": "기압면 (hPa)",
                "valid_u_count": "유효 표본 수",
                "observed_u_count": "실제 관측값 수",
                "interpolated_u_count": "보간값 수",
                "interpolated_u_ratio": "보간값 비율",
            }),
            width="stretch",
            hide_index=True,
            column_config={"보간값 비율": st.column_config.NumberColumn(format="percent")},
        )

        available_times = sorted(standard_view["datetime_utc"].dropna().unique())
        selected_time = st.selectbox(
            "개별 sounding 선택 (UTC)", available_times,
            format_func=lambda value: f"UTC {pd.Timestamp(value).strftime('%Y-%m-%d %H:%M')} · KST {pd.Timestamp(value).tz_convert('Asia/Seoul').strftime('%Y-%m-%d %H:%M')}",
        )
        profile = standard_view[standard_view["datetime_utc"].eq(selected_time)].sort_values("pressure_hpa", ascending=False)
        left, right = st.columns([2, 1])
        left.plotly_chart(sounding_temperature_wind(profile), width="stretch", config=PLOTLY_CONFIG)
        right.plotly_chart(wind_arrow_profile(profile), width="stretch", config=PLOTLY_CONFIG)
        with st.expander("개별 sounding 표와 전체 수직분포 보기"):
            st.dataframe(profile, width="stretch", hide_index=True)
            sample_times = available_times[-min(80, len(available_times)):]
            sample = standard_view[standard_view["datetime_utc"].isin(sample_times)]
            fig = px.line(sample, x="u_ms", y="pressure_hpa", color="datetime_utc", title="최근 개별 sounding u 수직 분포")
            fig.update_yaxes(autorange="reversed", type="log", title="기압 (hPa)")
            fig.update_xaxes(title="u (m/s)")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)


with tabs[2]:
    st.subheader("ISD 지표와 IGRA 상층 비교")
    if matched_view.empty:
        st.info("IGRA를 먼저 불러온 뒤 사이드바에서 ISD 후보를 확인하고 ‘ISD 지상자료 불러오기’를 누르세요.")
    else:
        upper_stats = surface_upper_statistics(matched_view)
        cards = st.columns(4)
        for card, (_, row) in zip(cards, upper_stats.iterrows()):
            card.metric(f"{row['level']} 평균 u", metric_value(float(row["mean_u_ms"]), " m/s"),
                        help=f"서풍 발생 비율 {row['westerly_ratio']:.1%}, 표준편차 {row['std_u_ms']:.2f} m/s")
        correlation_850 = matched_view["u_surface_ms"].corr(matched_view["u_850_ms"])
        correlation_300 = matched_view["u_surface_ms"].corr(matched_view["u_300_ms"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("지표–850 hPa 상관", metric_value(correlation_850))
        c2.metric("지표–300 hPa 상관", metric_value(correlation_300))
        c3.metric("평균 시간 차", metric_value(matched_view["time_difference_minutes"].mean(), "분", 1))
        c4.metric("대응 관측", f"{len(matched_view):,}쌍")
        st.plotly_chart(matched_time_series(matched_view), width="stretch", config=PLOTLY_CONFIG)
        col1, col2 = st.columns(2)
        col1.plotly_chart(surface_scatter(matched_view, 850), width="stretch", config=PLOTLY_CONFIG)
        col2.plotly_chart(surface_scatter(matched_view, 300), width="stretch", config=PLOTLY_CONFIG)
        col3, col4 = st.columns(2)
        col3.plotly_chart(seasonal_box(matched_view), width="stretch", config=PLOTLY_CONFIG)
        col4.plotly_chart(direction_frequency(matched_view), width="stretch", config=PLOTLY_CONFIG)
        st.plotly_chart(westerly_easterly_stack(upper_stats), width="stretch", config=PLOTLY_CONFIG)
        for message in observational_interpretation(vertical_stats, matched_view):
            st.write("- " + message)
        st.warning("지표풍 감소나 큰 변동성은 마찰과 국지 영향에 부합할 수 있지만 마찰력의 크기를 직접 계산한 것은 아닙니다. 지형·해륙풍·고저기압 위치·계절풍도 영향을 줍니다.")
        display_cols = [column for column in ("datetime_utc", "surface_datetime_utc", "datetime_kst", "time_difference_minutes", "u_surface_ms", "u_850_ms", "u_500_ms", "u_300_ms") if column in matched_view]
        with st.expander("UTC·KST 대응 자료 확인"):
            st.dataframe(matched_view[display_cols], width="stretch", hide_index=True)


with tabs[3]:
    st.subheader("두 관측소를 이용한 열풍 관계 근사 검증")
    if not ks_stations.empty and north_id and south_id:
        north_row = ks_stations[ks_stations["station_id"].eq(north_id)].iloc[0]
        south_row = ks_stations[ks_stations["station_id"].eq(south_id)].iloc[0]
        map_data = pd.DataFrame([
            {**north_row.to_dict(), "role": "북쪽"},
            {**south_row.to_dict(), "role": "남쪽"},
        ])
        st.plotly_chart(station_map(map_data), width="stretch", config=PLOTLY_CONFIG)
        st.caption(
            f"북쪽 {north_row['station_name']}: {north_row['first_year']}–{north_row['last_year']}년 · "
            f"남쪽 {south_row['station_name']}: {south_row['first_year']}–{south_row['last_year']}년 · "
            f"요청 기간: {start_date}–{end_date}"
        )
        if st.button("열풍 관계 계산 실행", key="thermal_run", type="primary"):
            if float(north_row["latitude"]) <= float(south_row["latitude"]):
                st.error("북쪽 관측소의 위도가 남쪽 관측소보다 높도록 선택하세요.")
            elif (
                int(north_row["last_year"]) < start_date.year
                or int(north_row["first_year"]) > end_date.year
                or int(south_row["last_year"]) < start_date.year
                or int(south_row["first_year"]) > end_date.year
            ):
                st.error("선택한 관측소 중 요청 기간과 자료기간이 겹치지 않는 곳이 있습니다. 위 지도 아래의 자료기간을 확인해 주세요.")
            else:
                try:
                    with st.spinner("두 관측소의 sounding을 불러와 3시간 이내 자료를 대응하고 있습니다..."):
                        _, north_standard, _ = load_igra_station(north_row, start_date, end_date)
                        _, south_standard, _ = load_igra_station(south_row, start_date, end_date)
                        results = paired_thermal_wind(
                            north_standard, south_standard,
                            float(north_row["latitude"]), float(south_row["latitude"]),
                            int(lower_pressure), int(upper_pressure), 3,
                        )
                    st.session_state.thermal_results = results
                    if results.empty:
                        st.warning("선택 기간과 기압층에서 기온·바람이 모두 유효한 대응 sounding이 없습니다.")
                    else:
                        st.success(f"열풍 비교 {len(results):,}쌍을 계산했습니다.")
                except Exception as exc:
                    show_exception("열풍 관계 계산을 완료하지 못했습니다.", exc)
    thermal = filter_season(st.session_state.thermal_results, season)
    if not thermal.empty:
        scores = regression_metrics(thermal["predicted_shear_ms"], thermal["observed_shear_ms"])
        cols = st.columns(6)
        cols[0].metric("남북 거리", metric_value(thermal["meridional_distance_km"].mean(), " km", 0))
        cols[1].metric("평균 남북 기온 차", metric_value(thermal["temperature_difference_c"].mean(), " °C"))
        cols[2].metric("상관계수", metric_value(scores.correlation))
        cols[3].metric("회귀 기울기", metric_value(scores.slope))
        cols[4].metric("RMSE", metric_value(scores.rmse, " m/s"))
        cols[5].metric("부호 일치율", metric_value(scores.sign_match_rate * 100, "%", 1))
        figures = thermal_wind_charts(thermal)
        for left_fig, right_fig in ((figures[0], figures[1]), (figures[2], figures[3])):
            left, right = st.columns(2)
            left.plotly_chart(left_fig, width="stretch", config=PLOTLY_CONFIG)
            right.plotly_chart(right_fig, width="stretch", config=PLOTLY_CONFIG)
        for message in thermal_interpretation(thermal):
            st.write("- " + message)
        st.warning("이 계산은 정확한 지균풍 검증이 아니라 두 지점 근사입니다. 관측소는 같은 경도선에 있지 않고, 실제 바람의 비지균 성분·시각/고도 차이·지형과 국지순환을 포함합니다.")
    else:
        st.info("북쪽·남쪽 관측소와 기압층을 정한 뒤 실행 버튼을 누르세요. sounding은 기본 3시간 이내에서만 대응합니다.")


with tabs[4]:
    st.subheader("NCEP/DOE Reanalysis II 에디 운동량 수송")
    st.info("재분석 자료는 관측과 수치모형이 결합된 자료입니다. OPeNDAP 처리는 수 분 걸릴 수 있으며 실행 버튼을 누르기 전에는 대규모 자료를 열지 않습니다.")
    catalog_error: Exception | None = None
    try:
        files = discover_reanalysis_files()
        years = available_reanalysis_years(files)
        default_year = latest_complete_year(years)
    except Exception as exc:
        catalog_error = exc
        years, default_year = [], None
    if catalog_error:
        show_exception("THREDDS 파일 목록을 읽지 못했습니다. 관측소 기반 탭은 계속 사용할 수 있습니다.", catalog_error)
    elif years:
        settings = st.columns(4)
        eddy_year = settings[0].selectbox("분석 연도", years, index=years.index(default_year))
        eddy_season = settings[1].selectbox("재분석 계절", ["겨울", "봄", "여름", "가을"], key="eddy_season")
        eddy_level = settings[2].selectbox("주요 표시 기압면", [850, 300], format_func=lambda value: f"{value} hPa")
        lat_range = settings[3].slider("위도 범위 (°N)", 0, 90, (20, 70))
        if st.button("에디 분석 실행", key="eddy_run", type="primary"):
            progress = st.progress(5, text="OPeNDAP 연결을 준비합니다...")
            try:
                progress.progress(20, text="선택 기간·위도·850/300 hPa의 u와 v를 불러옵니다...")
                result = load_and_calculate_eddy(int(eddy_year), eddy_season, float(lat_range[0]), float(lat_range[1]), (850, 300))
                progress.progress(90, text="u'v'와 구면 수렴 진단을 정리합니다...")
                st.session_state.eddy_results = result
                progress.progress(100, text="완료")
                st.success("에디 운동량 진단을 완료했습니다.")
            except Exception as exc:
                progress.empty()
                show_exception("OPeNDAP 또는 netCDF 처리에 실패했습니다. 다른 연도나 계절을 시도해 주세요.", exc)
        eddy = st.session_state.eddy_results
        if not eddy.empty:
            selected_eddy = eddy[eddy["pressure_hpa"].eq(eddy_level)]
            maximum = selected_eddy.loc[selected_eddy["eddy_acceleration_ms_day"].idxmax()]
            cards = st.columns(4)
            cards[0].metric("처리 날짜", f"{int(selected_eddy['processed_days'].max())}일")
            cards[1].metric("최대 양의 진단 위도", f"{maximum['latitude_deg']:.1f}°N")
            cards[2].metric("가속도", f"{maximum['eddy_acceleration_ms2']:.2e} m/s²")
            cards[3].metric("일 단위 환산", f"{maximum['eddy_acceleration_ms_day']:+.3f} m/s/day")
            figures = eddy_charts(eddy)
            for left_fig, right_fig in ((figures[0], figures[1]), (figures[2], figures[3])):
                left, right = st.columns(2)
                left.plotly_chart(left_fig, width="stretch", config=PLOTLY_CONFIG)
                right.plotly_chart(right_fig, width="stretch", config=PLOTLY_CONFIG)
            for message in eddy_interpretation(eddy, eddy_level):
                st.write("- " + message)
            st.warning("특정 한 해를 모든 해의 기후학적 특성으로 일반화하지 마세요. 이 값은 전체 대기 운동량 방정식이 아니며 마찰·산악 토크·평균 자오면 순환 등은 포함하지 않습니다.")


with tabs[5]:
    st.subheader("세 증거의 종합")
    thermal = st.session_state.thermal_results
    eddy = st.session_state.eddy_results
    evidence = st.columns(3)
    with evidence[0]:
        st.markdown("### 1. 상층 연직 강화")
        for message in observational_interpretation(vertical_stats, pd.DataFrame())[:1]:
            st.write(message)
    with evidence[1]:
        st.markdown("### 2. 지표 평균과 변동")
        for message in observational_interpretation(pd.DataFrame(), matched_view):
            st.write(message)
    with evidence[2]:
        st.markdown("### 3. 에디 수송 방향")
        for message in eddy_interpretation(eddy)[:1]:
            st.write(message)
    summary = build_summary_markdown(vertical_stats, matched_view, thermal, eddy)
    st.markdown(summary)
    st.subheader("분석 자료 다운로드")
    downloads = st.columns(3)
    datasets = [
        ("고도별 통계", vertical_stats, "vertical_statistics.csv"),
        ("표준기압면 자료", standard_view, "standard_pressure_levels.csv"),
        ("지상·상층 대응자료", matched_view, "surface_upper_matches.csv"),
        ("열풍 계산 결과", thermal, "thermal_wind_results.csv"),
        ("에디 진단 결과", eddy, "eddy_momentum_results.csv"),
        ("종합 지표", combined_metrics(vertical_stats, matched_view, thermal, eddy), "combined_metrics.csv"),
    ]
    for index, (label, data, filename) in enumerate(datasets):
        downloads[index % 3].download_button(
            f"{label} CSV", dataframe_csv(data), filename, "text/csv",
            disabled=data.empty, width="stretch", key=f"download_{filename}",
        )
    st.download_button("분석 결과 요약문 Markdown", summary.encode("utf-8"), "westerlies_summary.md", "text/markdown", width="stretch")


with tabs[6]:
    st.subheader("자료와 방법")
    method_items = {
        "편서풍과 서풍 성분": "편서풍대는 장기 평균 u>0인 영역입니다. 한 시각·한 관측소의 서풍만으로 전 지구 편서풍대를 증명하지 않습니다.",
        "지균풍과 열풍 관계": "기압경도력과 코리올리 힘이 균형을 이룬 바람이 지균풍입니다. 열풍은 두 기압면 지균풍의 차이이며 Δu_tw = −(R_d/f) ln(p_lower/p_upper) ∂T̄/∂y로 근사합니다.",
        "지표 마찰과 이동성 교란": "마찰은 지표풍을 바꾸지만 이 앱은 마찰력을 직접 산출하지 않습니다. 지형, 해륙풍, 계절풍, 이동성 고·저기압도 순간 바람에 작용합니다.",
        "에디 운동량 플럭스": "시간평균에서 벗어난 u'와 v'의 곱을 시간·경도 평균합니다. A_eddy = −[a cos²φ]⁻¹ ∂[cos²φ overline(u'v')]/∂φ는 평균 동서풍에 대한 한 진단항입니다.",
        "IGRA 2.2": "라디오존데 고정폭 자료에서 기압, 고도, 기온, 풍향과 풍속을 읽습니다. -8888/-9999는 결측으로 처리하고 기압 중복을 제거합니다.",
        "ISD": "Global Hourly의 WND 결합 필드를 방향·속도·품질 플래그로 분리하고, IGRA 위치에서 이름과 거리로 대한민국 후보를 제시합니다.",
        "NCEP/DOE Reanalysis II": "관측과 수치모형을 결합한 재분석입니다. 선택 기간·위도·기압면을 OPeNDAP에서 먼저 잘라 전체 netCDF 다운로드를 피합니다.",
        "기압면 보간": "로그 기압에 선형 보간하고 외삽하지 않습니다. 양옆 기압 차가 250 hPa를 넘으면 NaN이며 풍향 대신 u와 v를 각각 보간합니다.",
        "시간 대응": "UTC를 기준으로 sounding 전후 30~180분 범위의 가장 가까운 유효 ISD 관측을 고릅니다. 없으면 강제 대응하지 않습니다.",
        "상관과 인과": "상관계수와 회귀는 함께 변하는 정도이지 인과관계의 증명이 아닙니다. 단일 관측소, 두 지점 근사, 특정 한 해 재분석에는 대표성 한계가 있습니다.",
    }
    for title, body in method_items.items():
        with st.expander(title):
            st.write(body)
    st.markdown(
        """
        ### NOAA 자료 출처

        - [IGRA 2.2 관측소 목록](https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/doc/igra2-station-list.txt) · [자료 형식](https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/doc/igra2-data-format.txt) · [전체기간 자료](https://www.ncei.noaa.gov/data/integrated-global-radiosonde-archive/access/data-por/)
        - [Integrated Surface Database 안내](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) · [관측소 이력](https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv)
        - [NCEP/DOE Reanalysis II 안내](https://psl.noaa.gov/data/gridded/data.ncep.reanalysis2.html) · [THREDDS 카탈로그](https://psl.noaa.gov/thredds/catalog/Datasets/ncep.reanalysis2/Dailies/pressure/catalog.html)

        NOAA 공개자료는 API 키 없이 접근합니다. 자료 이용 시 각 NOAA 페이지의 인용·이용 안내와 갱신 상태를 함께 확인하세요.
        """
    )


st.divider()
st.subheader("현재 세션 의견")
st.caption("댓글은 현재 브라우저 세션에만 저장되며 앱 재시작 또는 세션 종료 시 사라질 수 있습니다.")
comment_cols = st.columns([1, 3])
comment_name = comment_cols[0].text_input("이름", key="comment_name")
comment_text = comment_cols[1].text_area("의견", key="comment_text", height=90)
if st.button("댓글 남기기"):
    if comment_name.strip() and comment_text.strip():
        st.session_state.comments.append({
            "name": comment_name.strip(),
            "text": comment_text.strip(),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
        st.success("현재 세션에 댓글을 저장했습니다.")
    else:
        st.warning("이름과 의견을 모두 입력해 주세요.")
for comment in reversed(st.session_state.comments):
    st.markdown(f"**{comment['name']}** · {comment['time']}")
    st.write(comment["text"])

st.caption("WesterlyScope · NOAA 공개자료 기반 교육·탐구용 분석기 · UTC/KST 병기")
