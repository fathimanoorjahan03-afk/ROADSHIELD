
from flask import Flask, render_template, request, url_for
import pandas as pd
from pathlib import Path
from functools import lru_cache

from services.prediction_service import (
    get_form_fields,
    predict_severity
)

from services.historical_service import (
    generate_historical_analysis
)

from services.hotspot_service import (
    REQUIRED_COLUMNS,
    build_hotspot_map,
    save_hotspot_map
)


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR / "data" / "road_collision_2023_2025 (2).csv"
)

DAY_LABELS = {
    1: "Sun",
    2: "Mon",
    3: "Tue",
    4: "Wed",
    5: "Thu",
    6: "Fri",
    7: "Sat"
}

DAY_ORDER = [2, 3, 4, 5, 6, 7, 1]


# =====================================================
# LOAD DASHBOARD DATA
# =====================================================

@lru_cache(maxsize=1)
def load_data():

    df = pd.read_csv(
        DATA_PATH,
        usecols=[
            "collision_year",
            "date",
            "time",
            "police_force",
            "day_of_week",
            "collision_severity",
            "speed_limit"
        ],
        low_memory=False
    )

    df["date"] = pd.to_datetime(
        df["date"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    df["month"] = df["date"].dt.month

    df["hour"] = pd.to_numeric(
        df["time"].astype("string").str[:2],
        errors="coerce"
    )

    df.loc[
        ~df["hour"].between(0, 23),
        "hour"
    ] = float("nan")

    for column in [
        "collision_year",
        "collision_severity",
        "speed_limit",
        "day_of_week"
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


# =====================================================
# HOMEPAGE
# =====================================================

@app.route("/")
def home():
    return render_template("home.html")


# =====================================================
# DASHBOARD
# =====================================================

@app.route("/dashboard")
def dashboard():

    df = load_data()

    total_collisions = len(df)

    yearly = (
        df.groupby("collision_year")
        .size()
        .sort_index()
    )

    monthly = (
        df.groupby("month")
        .size()
        .reindex(range(1, 13), fill_value=0)
    )

    hourly = (
        df.groupby("hour")
        .size()
        .reindex(range(24), fill_value=0)
    )

    years = [int(y) for y in yearly.index]

    yearly_counts = [
        int(x) for x in yearly.values
    ]

    latest_year = years[-1]
    previous_year = years[-2]

    previous_count = yearly.loc[previous_year]
    latest_count = yearly.loc[latest_year]

    yearly_change = (
        (latest_count - previous_count)
        / previous_count
        * 100
    )

    peak_hour = int(hourly.idxmax())

    # -----------------------------------------------
    # COLLISION TIMING HEATMAP
    # -----------------------------------------------

    heatmap_table = pd.crosstab(
        df["day_of_week"],
        df["hour"]
    ).reindex(
        index=DAY_ORDER,
        columns=range(24),
        fill_value=0
    )

    max_heat_value = int(
        heatmap_table.to_numpy().max()
    )

    heatmap_rows = []

    for day_code in DAY_ORDER:

        row_cells = []

        for hour in range(24):

            count = int(
                heatmap_table.loc[day_code, hour]
            )

            intensity = (
                count / max_heat_value
                if max_heat_value > 0
                else 0
            )

            alpha = 0.10 + 0.85 * intensity

            row_cells.append({
                "hour": f"{hour:02d}:00",
                "count": count,
                "color": (
                    f"rgba(56, 189, 248, "
                    f"{alpha:.3f})"
                )
            })

        heatmap_rows.append({
            "day": DAY_LABELS[day_code],
            "cells": row_cells
        })

    peak_day_code, peak_day_hour = (
        heatmap_table.stack().idxmax()
    )

    peak_day_name = DAY_LABELS[
        int(peak_day_code)
    ]

    peak_day_hour_label = (
        f"{int(peak_day_hour):02d}:00"
    )

    peak_day_hour_count = int(
        heatmap_table.loc[
            peak_day_code,
            peak_day_hour
        ]
    )

    timing_insight = (
        f"The highest concentration of recorded collisions "
        f"appears on {peak_day_name} at "
        f"{peak_day_hour_label}, "
        f"with {peak_day_hour_count:,} collisions "
        f"in the historical dataset."
    )

    # -----------------------------------------------
    # SEVERITY DISTRIBUTION
    # -----------------------------------------------

    severity_labels = {
        1: "Fatal",
        2: "Serious",
        3: "Slight"
    }

    severity_counts = (
        df["collision_severity"]
        .value_counts()
        .reindex([3, 2, 1], fill_value=0)
    )

    severity_chart = {
        "labels": [
            "Slight",
            "Serious",
            "Fatal"
        ],
        "counts": [
            int(severity_counts[3]),
            int(severity_counts[2]),
            int(severity_counts[1])
        ]
    }

    # -----------------------------------------------
    # SEVERITY BY SPEED LIMIT
    # -----------------------------------------------

    valid_speeds = [
        20, 30, 40, 50, 60, 70
    ]

    speed_data = df[
        df["speed_limit"].isin(valid_speeds)
        & df["collision_severity"].isin([1, 2, 3])
    ].copy()

    speed_counts = pd.crosstab(
        speed_data["speed_limit"],
        speed_data["collision_severity"]
    ).reindex(
        index=valid_speeds,
        columns=[3, 2, 1],
        fill_value=0
    )

    speed_totals = speed_counts.sum(axis=1)

    speed_percentages = (
        speed_counts
        .div(
            speed_totals.replace(
                0, float("nan")
            ),
            axis=0
        )
        .fillna(0)
        * 100
    )

    speed_chart = {
        "labels": [
            f"{speed} mph"
            for speed in valid_speeds
        ],
        "slight": (
            speed_percentages[3]
            .round(2)
            .tolist()
        ),
        "serious": (
            speed_percentages[2]
            .round(2)
            .tolist()
        ),
        "fatal": (
            speed_percentages[1]
            .round(2)
            .tolist()
        )
    }

    # -----------------------------------------------
    # HISTORICAL SEVERITY INSIGHTS
    # -----------------------------------------------

    most_common_code = int(
        severity_counts.idxmax()
    )

    most_common_percentage = (
        severity_counts.max()
        / severity_counts.sum()
        * 100
    )

    severe_counts = (
        speed_counts[1] + speed_counts[2]
    )

    severe_percentages = (
        severe_counts
        .div(
            speed_totals.replace(
                0, float("nan")
            )
        )
        .mul(100)
    )

    eligible_speeds = severe_percentages[
        speed_totals >= 100
    ]

    insights = [
        (
            f"{severity_labels[most_common_code]} "
            f"collisions account for "
            f"{most_common_percentage:.1f}% "
            f"of recorded collisions."
        )
    ]

    if not eligible_speeds.empty:

        highest_speed = int(
            eligible_speeds.idxmax()
        )

        highest_percentage = float(
            eligible_speeds.max()
        )

        highest_count = int(
            speed_totals.loc[
                highest_speed
            ]
        )

        insights.append(
            f"Among the displayed speed-limit groups, "
            f"{highest_speed} mph has the highest "
            f"proportion of Serious or Fatal collisions "
            f"({highest_percentage:.1f}%, based on "
            f"{highest_count:,} recorded collisions)."
        )

    insights.append(
        "These percentages describe severity among "
        "recorded collisions. They do not measure "
        "the probability of an accident occurring."
    )

    # -----------------------------------------------
    # DASHBOARD CARDS AND CHART DATA
    # -----------------------------------------------

    stats = {
        "total": f"{total_collisions:,}",
        "yearly_change": f"{yearly_change:+.2f}%",
        "areas": int(
            df["police_force"].nunique()
        ),
        "peak_hour": f"{peak_hour:02d}:00",
        "latest_year": latest_year,
        "previous_year": previous_year
    }

    chart_data = {
        "years": years,
        "yearly_counts": yearly_counts,
        "months": [
            "Jan", "Feb", "Mar", "Apr",
            "May", "Jun", "Jul", "Aug",
            "Sep", "Oct", "Nov", "Dec"
        ],
        "monthly_counts": [
            int(x) for x in monthly.values
        ]
    }

    return render_template(
        "dashboard.html",
        stats=stats,
        chart_data=chart_data,
        heatmap_rows=heatmap_rows,
        timing_insight=timing_insight,
        severity_chart=severity_chart,
        speed_chart=speed_chart,
        insights=insights
    )


# =====================================================
# SEVERITY PREDICTION + HISTORICAL ANALYSIS
# =====================================================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    fields = get_form_fields()

    result = None
    historical = None
    error = None
    values = {}

    if request.method == "POST":

        values = request.form.to_dict()

        try:

            # Run the saved prediction model.
            result = predict_severity(values)

            # Generate historical analysis only
            # when the user clicks its button.
            if values.get("intent") == "historical":

                historical = (
                    generate_historical_analysis(
                        values
                    )
                )

        except ValueError as exc:

            error = str(exc)

        except Exception:

            app.logger.exception(
                "Prediction or historical "
                "analysis failed"
            )

            error = (
                "The operation could not be completed. "
                "Check the Flask terminal for details."
            )

    return render_template(
        "predict.html",
        fields=fields,
        values=values,
        result=result,
        historical=historical,
        error=error
    )


# =====================================================
# BLACK SPOT ANALYSIS
# =====================================================

@lru_cache(maxsize=1)
def prepare_hotspot_page():
    """
    Generate the hotspot map only once
    for the current Flask process.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Collision dataset not found: {DATA_PATH}"
        )

    # Read only the columns needed for the map.
    df = pd.read_csv(
        DATA_PATH,
        usecols=REQUIRED_COLUMNS,
        low_memory=False
    )

    # Use the teammate's original hotspot algorithm.
    map_object, grid_data, hot_cells, summary = (
        build_hotspot_map(
            df,
            percentile=0.99
        )
    )

    # Save the Folium map where Flask can serve it.
    output_path = (
        Path(app.static_folder)
        / "generated"
        / "hotspots_map.html"
    )

    save_hotspot_map(
        map_object,
        output_path
    )

    # Keep project branding consistent.
    map_html = output_path.read_text(
        encoding="utf-8"
    )

    map_html = map_html.replace(
        "ROADSHIELD AI",
        "ROADSHIELD"
    )

    output_path.write_text(
        map_html,
        encoding="utf-8"
    )

    return summary


@app.route("/hotspots")
def hotspots():

    try:

        summary = prepare_hotspot_page()

        map_url = url_for(
            "static",
            filename="generated/hotspots_map.html"
        )

        return render_template(
            "hotspots.html",
            summary=summary,
            map_url=map_url,
            error=None
        )

    except Exception:

        app.logger.exception(
            "Black Spot Analysis failed"
        )

        return render_template(
            "hotspots.html",
            summary=None,
            map_url=None,
            error=(
                "The hotspot map could not be generated. "
                "Please check the Flask terminal "
                "for the detailed error."
            )
        ), 500


# =====================================================
# RUN FLASK
# =====================================================

if __name__ == "__main__":
    app.run(debug=True)
