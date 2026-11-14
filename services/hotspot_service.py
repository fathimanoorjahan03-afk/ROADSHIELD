"""RoadShield AI hotspot analysis service for Flask integration.

This module extracts the final combined Fatal + Serious hotspot workflow from
RoadShield_Hotspot_Satellite_Fixed (1).ipynb and makes it reusable outside Colab.

Input: a pandas DataFrame loaded from the UK STATS19 collision CSV.
Output: 1 km severe-collision grid summary, candidate hotspot cells, summary KPIs,
        and a Folium satellite map.

Severity codes follow the notebook: 1 = Fatal, 2 = Serious, 3 = Slight.
Slight collisions are excluded from this hotspot analysis.
"""
from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import folium
import numpy as np
import pandas as pd
import branca.colormap as cm
from branca.element import MacroElement, Element
from jinja2 import Template
from folium.plugins import Fullscreen


REQUIRED_COLUMNS = [
    "longitude",
    "latitude",
    "location_easting_osgr",
    "location_northing_osgr",
    "collision_severity",
    "first_road_class",
    "first_road_number",
    "local_authority_highway_current",
]


class ClickToZoom(MacroElement):
    """Attach a smooth map zoom action to an individual hotspot marker."""

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        {{ this._parent.get_name() }}.on('click', function(event) {
            {{ this._parent._parent._parent.get_name() }}.flyTo(
                event.latlng,
                {{ this.zoom }},
                {duration: 1.2}
            );
        });
        {% endmacro %}
        """
    )

    def __init__(self, zoom: int = 15):
        super().__init__()
        self._name = "ClickToZoom"
        self.zoom = int(zoom)


def _most_common_value(values: pd.Series) -> str:
    values = values.dropna().astype(str).str.strip()
    values = values[~values.isin(["", "-1", "nan", "None", "Unknown"])]
    return str(values.value_counts().index[0]) if not values.empty else "Unknown"


def _road_labels(frame: pd.DataFrame) -> pd.Series:
    road_class = pd.to_numeric(frame["first_road_class"], errors="coerce")
    road_number = pd.to_numeric(frame["first_road_number"], errors="coerce").fillna(0)
    road_prefix = road_class.map({1: "M", 2: "A", 3: "A", 4: "B", 5: "C", 6: "Unclassified"})
    labels = pd.Series("Unknown road", index=frame.index, dtype="object")

    numbered_roads = road_class.isin([1, 2, 3, 4, 5]) & road_number.gt(0)
    labels.loc[numbered_roads] = (
        road_prefix.loc[numbered_roads].astype(str)
        + road_number.loc[numbered_roads].astype(int).astype(str)
    )
    labels.loc[road_class.eq(6)] = "Unclassified road"
    return labels


def prepare_hotspot_data(
    df: pd.DataFrame,
    percentile: float = 0.99,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Prepare grid counts and candidate hotspots using the notebook's method.

    The percentile is calculated across occupied 1 km cells containing at least
    one Fatal or Serious collision, matching the final notebook cell.
    """
    if not 0 < percentile < 1:
        raise ValueError("percentile must be between 0 and 1, e.g. 0.99")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    work = df.copy()
    numeric_columns = [
        "longitude", "latitude", "location_easting_osgr",
        "location_northing_osgr", "collision_severity",
        "first_road_class", "first_road_number",
    ]
    for col in numeric_columns:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    valid_geo = (
        work["longitude"].between(-8.5, 2.5)
        & work["latitude"].between(49.5, 61.5)
        & work["location_easting_osgr"].between(0, 700000)
        & work["location_northing_osgr"].between(0, 1300000)
    )

    # Only severity 1 (Fatal) and 2 (Serious) enter the hotspot calculations.
    severe = work.loc[
        valid_geo & work["collision_severity"].isin([1, 2])
    ].copy()
    if severe.empty:
        raise ValueError(
            "No valid geocoded Fatal or Serious collision records were found. "
            "Check the severity and coordinate columns."
        )

    severe["grid_easting_1km"] = (
        np.floor(severe["location_easting_osgr"] / 1000) * 1000
    ).astype("int64")
    severe["grid_northing_1km"] = (
        np.floor(severe["location_northing_osgr"] / 1000) * 1000
    ).astype("int64")
    severe["road_label_for_map"] = _road_labels(severe)

    grid_keys = ["grid_easting_1km", "grid_northing_1km"]
    grid_data = (
        severe.groupby(grid_keys)
        .agg(
            fatal=("collision_severity", lambda values: int((values == 1).sum())),
            serious=("collision_severity", lambda values: int((values == 2).sum())),
            fatal_serious_total=("collision_severity", "size"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
        )
        .reset_index()
    )

    threshold = float(grid_data["fatal_serious_total"].quantile(percentile))
    hotspots = grid_data.loc[
        grid_data["fatal_serious_total"] >= threshold
    ].copy()
    if hotspots.empty:
        raise ValueError("No candidate hotspot cells were identified.")

    grid_labels = (
        severe.groupby(grid_keys)
        .agg(
            road_name=("road_label_for_map", _most_common_value),
            authority_code=("local_authority_highway_current", _most_common_value),
        )
        .reset_index()
    )
    hotspots = hotspots.merge(grid_labels, on=grid_keys, how="left")
    hotspots["location_name"] = (
        hotspots["road_name"].fillna("Unknown road").astype(str)
        + " • "
        + hotspots["authority_code"].fillna("Authority unavailable").astype(str)
    )

    summary = {
        "valid_fatal_serious_records": int(len(severe)),
        "total_fatal": int(grid_data["fatal"].sum()),
        "total_serious": int(grid_data["serious"].sum()),
        "total_fatal_serious": int(grid_data["fatal_serious_total"].sum()),
        "occupied_severe_grid_cells": int(len(grid_data)),
        "candidate_hotspot_cells": int(len(hotspots)),
        "percentile": float(percentile),
        "hotspot_threshold": threshold,
    }
    return grid_data, hotspots, summary


def build_hotspot_map(
    df: pd.DataFrame,
    percentile: float = 0.99,
    marker_zoom: int = 15,
) -> tuple[folium.Map, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build the interactive Folium map and return map + analysis tables.

    Note: authority codes are shown in labels to avoid making the Flask app
    dependent on an external geocoding/lookup request. The original notebook
    optionally looked up full authority names over the network.
    """
    grid_data, hot_cells, summary = prepare_hotspot_data(df, percentile)

    map_center = [float(hot_cells["latitude"].mean()), float(hot_cells["longitude"].mean())]
    roadshield_map = folium.Map(
        location=map_center,
        zoom_start=6,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    folium.TileLayer(
        tiles=(
            "https://services.arcgisonline.com/ArcGIS/rest/"
            "services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri, Vantor, Earthstar Geographics, and GIS User Community",
        name="Satellite Imagery",
        overlay=False,
        control=True,
        show=True,
        max_zoom=19,
    ).add_to(roadshield_map)

    minimum_count = int(hot_cells["fatal_serious_total"].min())
    maximum_count = int(hot_cells["fatal_serious_total"].max())
    if minimum_count == maximum_count:
        maximum_count = minimum_count + 1

    colour_scale = cm.LinearColormap(
        colors=["#FDE047", "#FB923C", "#EF4444", "#7F1D1D"],
        vmin=minimum_count,
        vmax=maximum_count,
        caption="Fatal + Serious collisions per 1 km hotspot cell",
    )

    hotspot_layer = folium.FeatureGroup(name="Fatal + Serious Hotspots", show=True)

    for row in hot_cells.itertuples(index=False):
        fatal_count = int(row.fatal)
        serious_count = int(row.serious)
        severe_total = int(row.fatal_serious_total)
        spot_colour = colour_scale(severe_total)
        safe_location = html.escape(str(row.location_name))
        fatal_pct = (fatal_count / severe_total) * 100 if severe_total else 0
        serious_pct = (serious_count / severe_total) * 100 if severe_total else 0

        popup_html = f"""
        <div style="font-family:Arial,sans-serif;min-width:235px;">
          <div style="background:#111827;color:white;padding:11px;border-radius:7px 7px 0 0;">
            <div style="font-size:11px;letter-spacing:1px;color:#FCA5A5;font-weight:bold;">ROADSHIELD AI</div>
            <div style="font-size:16px;font-weight:bold;margin-top:4px;">{safe_location}</div>
          </div>
          <div style="padding:12px;background:white;">
            <div style="font-size:13px;color:#374151;margin-bottom:9px;">Fatal + Serious hotspot</div>
            <div style="display:flex;gap:8px;margin-bottom:10px;">
              <div style="flex:1;background:#FEE2E2;padding:10px;border-radius:6px;">
                <div style="font-size:11px;color:#991B1B;">FATAL</div>
                <div style="font-size:23px;font-weight:bold;color:#991B1B;">{fatal_count:,}</div>
                <div style="font-size:10px;color:#7F1D1D;">{fatal_pct:.1f}% of severe cases</div>
              </div>
              <div style="flex:1;background:#FFEDD5;padding:10px;border-radius:6px;">
                <div style="font-size:11px;color:#9A3412;">SERIOUS</div>
                <div style="font-size:23px;font-weight:bold;color:#9A3412;">{serious_count:,}</div>
                <div style="font-size:10px;color:#9A3412;">{serious_pct:.1f}% of severe cases</div>
              </div>
            </div>
            <div style="border-top:1px solid #E5E7EB;padding-top:9px;font-size:12px;color:#374151;">
              <b>Total Fatal + Serious:</b> {severe_total:,}<br>
              <b>Grid size:</b> 1 km × 1 km<br>
              <b>Grid easting:</b> {int(row.grid_easting_1km):,}<br>
              <b>Grid northing:</b> {int(row.grid_northing_1km):,}<br>
              <b>Latitude:</b> {float(row.latitude):.5f}<br>
              <b>Longitude:</b> {float(row.longitude):.5f}
            </div>
          </div>
        </div>
        """

        popup = folium.Popup(popup_html, max_width=340)
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=13,
            color=spot_colour,
            weight=5,
            opacity=0.30,
            fill=True,
            fill_color=spot_colour,
            fill_opacity=0.12,
            interactive=False,
        ).add_to(hotspot_layer)

        marker_radius = float(np.clip(6 + np.sqrt(severe_total), 7, 17))
        marker = folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=marker_radius,
            color="white",
            weight=2,
            opacity=1,
            fill=True,
            fill_color=spot_colour,
            fill_opacity=0.95,
            tooltip=(
                f"{safe_location} | Fatal: {fatal_count} | "
                f"Serious: {serious_count} | Total: {severe_total}"
            ),
            popup=popup,
        )
        marker.add_to(hotspot_layer)
        ClickToZoom(zoom=marker_zoom).add_to(marker)

    hotspot_layer.add_to(roadshield_map)
    colour_scale.add_to(roadshield_map)

    total_fatal = summary["total_fatal"]
    total_serious = summary["total_serious"]
    summary_html = f"""
    <div style="position:fixed;top:15px;left:60px;z-index:9999;width:285px;
    background:rgba(15,23,42,0.94);color:white;padding:14px;border-radius:9px;
    border-left:5px solid #EF4444;box-shadow:0 3px 14px rgba(0,0,0,.35);font-family:Arial;">
      <div style="font-size:11px;letter-spacing:2px;color:#FCA5A5;font-weight:bold;">ROADSHIELD AI</div>
      <div style="font-size:19px;font-weight:bold;margin-top:4px;">Fatal + Serious Hotspots</div>
      <div style="font-size:11px;color:#CBD5E1;margin-top:4px;">Great Britain | 2023–2025 | 1 km grid</div>
      <hr style="border-color:#475569;margin:10px 0;">
      <div style="display:flex;gap:14px;">
        <div><div style="font-size:20px;font-weight:bold;color:#FCA5A5;">{total_fatal:,}</div><div style="font-size:10px;color:#CBD5E1;">Fatal</div></div>
        <div><div style="font-size:20px;font-weight:bold;color:#FDBA74;">{total_serious:,}</div><div style="font-size:10px;color:#CBD5E1;">Serious</div></div>
        <div><div style="font-size:20px;font-weight:bold;">{len(hot_cells):,}</div><div style="font-size:10px;color:#CBD5E1;">Hotspots</div></div>
      </div>
      <div style="font-size:10px;color:#CBD5E1;margin-top:10px;">Hotspots are selected using the {percentile:.0%} percentile of occupied Fatal + Serious grid cells.</div>
    </div>
    """
    roadshield_map.get_root().html.add_child(Element(summary_html))

    folium.LayerControl(position="topright", collapsed=False).add_to(roadshield_map)
    Fullscreen(position="topleft", force_separate_button=True).add_to(roadshield_map)
    hotspot_points = hot_cells[["latitude", "longitude"]].values.tolist()
    roadshield_map.fit_bounds(hotspot_points, padding=(30, 30), max_zoom=8)

    hot_cells = hot_cells.sort_values(
        ["fatal_serious_total", "fatal", "serious"], ascending=False
    ).reset_index(drop=True)

    return roadshield_map, grid_data, hot_cells, summary


def save_hotspot_map(map_object: folium.Map, output_path: str | Path) -> Path:
    """Save a generated map to an HTML file and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    map_object.save(str(path))
    return path
