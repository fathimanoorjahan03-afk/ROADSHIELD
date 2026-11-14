
from pathlib import Path
from functools import lru_cache

import pandas as pd

from services.prediction_service import get_code_meaning


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "road_collision_2023_2025 (2).csv"

DAY_NAMES = {
    1: "Sunday",
    2: "Monday",
    3: "Tuesday",
    4: "Wednesday",
    5: "Thursday",
    6: "Friday",
    7: "Saturday"
}

MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}


@lru_cache(maxsize=1)
def load_historical_data():
    columns = [
        "collision_year",
        "collision_severity",
        "date",
        "time",
        "day_of_week",
        "weather_conditions",
        "light_conditions",
        "road_surface_conditions",
        "road_type",
        "speed_limit",
        "urban_or_rural_area"
    ]

    df = pd.read_csv(
        DATA_PATH,
        usecols=columns,
        low_memory=False
    )

    for column in columns:
        if column not in ["date", "time"]:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df["date"] = pd.to_datetime(
        df["date"],
        dayfirst=True,
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

    df = df[
        df["collision_severity"].isin([1, 2, 3])
    ].copy()

    return df


def count_text(number):
    return f"{int(number):,}"


def percentage(part, total):
    if total == 0:
        return 0.0

    return part / total * 100


def severity_percentage(df):
    if df.empty:
        return 0.0

    severe = df["collision_severity"].isin([1, 2]).sum()
    return percentage(severe, len(df))


def filter_conditions(df, selected):
    result = df

    for column, value in selected.items():
        result = result[result[column] == value]

    return result


def peak_value(df, column):
    valid = df[column].dropna()

    if valid.empty:
        return None, 0

    counts = valid.value_counts()

    return int(counts.idxmax()), int(counts.max())


def readable_condition(field, code):
    description = get_code_meaning(field, str(code))

    if description:
        return description

    return f"code {code}"


def sample_note(count):
    if count == 0:
        return "No records matched this combination."

    if count < 30:
        return (
            "This comparison contains fewer than 30 records, "
            "so its percentages may be unstable."
        )

    if count < 100:
        return (
            "This comparison has a limited sample size; "
            "interpret differences cautiously."
        )

    return ""


def generate_historical_analysis(form_values):

    df = load_historical_data()

    selected_columns = [
        "weather_conditions",
        "light_conditions",
        "road_surface_conditions",
        "road_type",
        "speed_limit",
        "urban_or_rural_area",
        "day_of_week",
        "month",
        "hour"
    ]

    selected = {}

    for column in selected_columns:
        raw_value = form_values.get(column, "")

        try:
            value = float(raw_value)

            if not value.is_integer():
                raise ValueError()

            selected[column] = int(value)

        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid or missing historical input: {column}"
            )

    day = selected["day_of_week"]
    month = selected["month"]
    hour = selected["hour"]

    if day not in DAY_NAMES:
        raise ValueError("Invalid day of week.")

    if month not in MONTH_NAMES:
        raise ValueError("Invalid month.")

    if hour not in range(24):
        raise ValueError("Invalid collision hour.")

    weather = readable_condition(
        "weather_conditions",
        selected["weather_conditions"]
    )

    lighting = readable_condition(
        "light_conditions",
        selected["light_conditions"]
    )

    surface = readable_condition(
        "road_surface_conditions",
        selected["road_surface_conditions"]
    )

    road = readable_condition(
        "road_type",
        selected["road_type"]
    )

    day_name = DAY_NAMES[day]
    month_name = MONTH_NAMES[month]
    time_label = f"{hour:02d}:00–{hour:02d}:59"

    total = len(df)
    overall_severe = severity_percentage(df)

    # Individual condition groups
    weather_df = filter_conditions(
        df, {"weather_conditions": selected["weather_conditions"]}
    )

    day_df = filter_conditions(
        df, {"day_of_week": day}
    )

    hour_df = filter_conditions(
        df, {"hour": hour}
    )

    month_df = filter_conditions(
        df, {"month": month}
    )

    surface_df = filter_conditions(
        df,
        {"road_surface_conditions": selected["road_surface_conditions"]}
    )

    speed_df = filter_conditions(
        df, {"speed_limit": selected["speed_limit"]}
    )

    road_df = filter_conditions(
        df, {"road_type": selected["road_type"]}
    )

    lighting_df = filter_conditions(
        df, {"light_conditions": selected["light_conditions"]}
    )

    # Two-condition and multi-condition groups
    day_hour_df = filter_conditions(
        df,
        {
            "day_of_week": day,
            "hour": hour
        }
    )

    weather_hour_df = filter_conditions(
        df,
        {
            "weather_conditions": selected["weather_conditions"],
            "hour": hour
        }
    )

    weather_surface_df = filter_conditions(
        df,
        {
            "weather_conditions": selected["weather_conditions"],
            "road_surface_conditions":
                selected["road_surface_conditions"]
        }
    )

    speed_road_df = filter_conditions(
        df,
        {
            "speed_limit": selected["speed_limit"],
            "road_type": selected["road_type"]
        }
    )

    # Selected combination of main environmental conditions
    environmental_df = filter_conditions(
        df,
        {
            "weather_conditions": selected["weather_conditions"],
            "light_conditions": selected["light_conditions"],
            "road_surface_conditions":
                selected["road_surface_conditions"]
        }
    )

    # Strict match across the nine selected conditions
    exact_df = filter_conditions(df, selected)

    sections = []

    # ==================================================
    # WEATHER INSIGHTS
    # ==================================================

    weather_insights = []

    weather_insights.append(
        f"Between 2023 and 2025, "
        f"{count_text(len(weather_df))} collisions were "
        f"recorded under {weather.lower()} conditions. "
        f"This represents {percentage(len(weather_df), total):.1f}% "
        f"of all historical collision records."
    )

    if not weather_df.empty:
        peak_hour, peak_count = peak_value(
            weather_df, "hour"
        )

        if peak_hour is not None:
            weather_insights.append(
                f"For this weather condition, the highest "
                f"number of collisions was recorded during "
                f"{peak_hour:02d}:00–{peak_hour:02d}:59, "
                f"with {count_text(peak_count)} records. "
                f"The selected hour, {time_label}, had "
                f"{count_text(len(weather_hour_df))} records "
                f"under the same weather condition."
            )

        yearly_weather = (
            weather_df.groupby("collision_year")
            .size()
        )

        if not yearly_weather.empty:
            peak_year = int(yearly_weather.idxmax())
            peak_year_count = int(yearly_weather.max())

            weather_insights.append(
                f"Among the available years, {peak_year} had "
                f"the highest number of recorded collisions "
                f"under this weather condition "
                f"({count_text(peak_year_count)} records)."
            )

    sections.append({
        "title": "Weather-Related Insights",
        "icon": "☁",
        "insights": weather_insights
    })

    # ==================================================
    # DAY AND TIME INSIGHTS
    # ==================================================

    time_insights = [
        (
            f"Across the three-year dataset, "
            f"{count_text(len(day_df))} collisions were "
            f"recorded on {day_name}s, while "
            f"{count_text(len(hour_df))} collisions occurred "
            f"during the selected hour ({time_label}) "
            f"across all days."
        ),
        (
            f"Specifically, {count_text(len(day_hour_df))} "
            f"collisions occurred on {day_name}s between "
            f"{time_label}. This accounts for "
            f"{percentage(len(day_hour_df), len(day_df)):.1f}% "
            f"of collisions reported on that weekday "
            f"where an hour was recorded."
        )
    ]

    if not day_df.empty:
        peak_hour, peak_count = peak_value(
            day_df, "hour"
        )

        if peak_hour is not None:
            time_insights.append(
                f"The most frequently recorded collision "
                f"hour on {day_name}s was "
                f"{peak_hour:02d}:00–{peak_hour:02d}:59, "
                f"with {count_text(peak_count)} records."
            )

    time_insights.append(
        f"In {month_name}, the dataset contains "
        f"{count_text(len(month_df))} recorded collisions "
        f"across 2023–2025."
    )

    sections.append({
        "title": "Day and Time Insights",
        "icon": "◷",
        "insights": time_insights
    })

    # ==================================================
    # ROAD AND ENVIRONMENT INSIGHTS
    # ==================================================

    road_insights = [
        (
            f"For roads with a {selected['speed_limit']} mph "
            f"speed limit, {count_text(len(speed_df))} "
            f"collisions were reported. "
            f"Of these, {count_text(len(speed_road_df))} "
            f"occurred on roads classified as "
            f"{road.lower()}."
        ),
        (
            f"Under {surface.lower()} road-surface "
            f"conditions, {count_text(len(surface_df))} "
            f"collisions were recorded. "
            f"Of these, {count_text(len(weather_surface_df))} "
            f"also had the selected weather condition "
            f"({weather.lower()})."
        ),
        (
            f"Under {lighting.lower()} lighting conditions, "
            f"{count_text(len(lighting_df))} collisions "
            f"were recorded. "
            f"When the selected weather, lighting and road "
            f"surface were combined, "
            f"{count_text(len(environmental_df))} records matched."
        )
    ]

    if len(speed_road_df) > 0:
        severe_rate = severity_percentage(speed_road_df)

        road_insights.append(
            f"Among the {count_text(len(speed_road_df))} "
            f"collisions matching the selected road type "
            f"and speed limit, {severe_rate:.1f}% were "
            f"classified as Serious or Fatal, compared with "
            f"{overall_severe:.1f}% in the entire dataset. "
            f"{sample_note(len(speed_road_df))}"
        )

    sections.append({
        "title": "Road and Environmental Insights",
        "icon": "◇",
        "insights": road_insights
    })

    # ==================================================
    # COMBINED CONDITION INSIGHTS
    # ==================================================

    combined_insights = [
        (
            f"When the selected weather and hour were "
            f"considered together, "
            f"{count_text(len(weather_hour_df))} "
            f"historical collision records matched."
        ),
        (
            f"Matching both {day_name} and the selected "
            f"hour produced {count_text(len(day_hour_df))} "
            f"records. Adding other conditions creates "
            f"more specific historical comparison groups."
        )
    ]

    exact_count = len(exact_df)

    if exact_count > 0:
        exact_severe = severity_percentage(exact_df)

        combined_insights.append(
            f"Across all nine selected conditions "
            f"(weather, lighting, road surface, road type, "
            f"speed limit, urban/rural classification, "
            f"day, month and hour), "
            f"{count_text(exact_count)} collision records "
            f"matched exactly. Among those records, "
            f"{exact_severe:.1f}% were Serious or Fatal. "
            f"{sample_note(exact_count)}"
        )

        year_counts = (
            exact_df.groupby("collision_year")
            .size()
            .sort_index()
        )

        year_parts = [
            f"{int(year)}: {count_text(count)}"
            for year, count in year_counts.items()
        ]

        combined_insights.append(
            "The matching records were distributed by year "
            "as follows: " + "; ".join(year_parts) + "."
        )

    else:
        combined_insights.append(
            "No historical records matched all nine "
            "selected conditions simultaneously. "
            "This does not mean such a collision is "
            "impossible. The broader weather-hour, "
            "day-hour, weather-surface and road-speed "
            "comparisons above provide historical context."
        )

    sections.append({
        "title": "Combined Condition Insights",
        "icon": "✦",
        "insights": combined_insights
    })

    # ==================================================
    # OVERALL INTERPRETATION
    # ==================================================

    interpretation = [
        (
            f"The overall Serious/Fatal proportion in the "
            f"historical dataset is {overall_severe:.1f}%. "
            "The figures above describe reported collisions "
            "rather than the chance of a collision occurring."
        ),
        (
            "Differences in recorded collision counts can "
            "reflect traffic volume, road exposure, reporting "
            "patterns and other factors. They should not be "
            "interpreted as proof that a selected condition "
            "causes more collisions."
        )
    ]

    sections.append({
        "title": "Interpretation and Limitations",
        "icon": "ⓘ",
        "insights": interpretation
    })

    return {
        "total_records": total,
        "exact_matches": exact_count,
        "sections": sections,
        "selected_summary": (
            f"{day_name}, {month_name}, {time_label} | "
            f"{weather} | {selected['speed_limit']} mph | "
            f"{road}"
        )
    }
