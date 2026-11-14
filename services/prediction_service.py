
from pathlib import Path
from functools import lru_cache
import calendar
import csv

import joblib
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "road_collision_2023_2025 (2).csv"
LABELS_PATH = BASE_DIR / "data" / "code_labels.csv"

SEVERITY_LABELS = {
    1: "Fatal",
    2: "Serious",
    3: "Slight"
}

NUMERIC_FIELDS = [
    "number_of_vehicles",
    "speed_limit",
    "month",
    "hour"
]

SEARCHABLE_FIELDS = [
    "police_force",
    "local_authority_ons_district",
    "local_authority_highway_current"
]

ESSENTIAL_FIELDS = [
    "number_of_vehicles",
    "day_of_week",
    "road_type",
    "speed_limit",
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",
    "urban_or_rural_area",
    "month",
    "hour"
]

MISSING_CATEGORY_FIELDS = [
    "first_road_class",
    "junction_detail_historic",
    "junction_detail",
    "junction_control",
    "second_road_class",
    "pedestrian_crossing_human_control_historic",
    "pedestrian_crossing_physical_facilities_historic",
    "pedestrian_crossing",
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",
    "special_conditions_at_site",
    "carriageway_hazards_historic",
    "carriageway_hazards",
    "urban_or_rural_area",
    "trunk_road_flag"
]

FIELD_NAMES = {
    "police_force": "Police Force",
    "number_of_vehicles": "Number of Vehicles",
    "day_of_week": "Day of Week",
    "local_authority_ons_district": "Local Authority District",
    "local_authority_highway_current": "Highway Authority",
    "first_road_class": "First Road Class",
    "road_type": "Road Type",
    "speed_limit": "Speed Limit (mph)",
    "junction_detail_historic": "Historic Junction Detail",
    "junction_detail": "Junction Detail",
    "junction_control": "Junction Control",
    "second_road_class": "Second Road Class",
    "pedestrian_crossing_human_control_historic":
        "Historic Crossing Human Control",
    "pedestrian_crossing_physical_facilities_historic":
        "Historic Crossing Facilities",
    "pedestrian_crossing": "Pedestrian Crossing",
    "light_conditions": "Lighting Conditions",
    "weather_conditions": "Weather Conditions",
    "road_surface_conditions": "Road Surface Conditions",
    "special_conditions_at_site": "Special Conditions at Site",
    "carriageway_hazards_historic": "Historic Carriageway Hazards",
    "carriageway_hazards": "Carriageway Hazards",
    "urban_or_rural_area": "Urban or Rural Area",
    "trunk_road_flag": "Trunk Road Flag",
    "month": "Month",
    "hour": "Collision Time"
}

DAY_NAMES = {
    "1": "Sunday",
    "2": "Monday",
    "3": "Tuesday",
    "4": "Wednesday",
    "5": "Thursday",
    "6": "Friday",
    "7": "Saturday"
}

# Verified/common STATS19 descriptions.
# Newer and historic code systems are kept separate.
# Unmapped codes remain selectable using their original values.
CODE_LABELS = {
    "police_force": {
    "1": "Metropolitan Police",
    "3": "Cumbria",
    "4": "Lancashire",
    "5": "Merseyside",
    "6": "Greater Manchester",
    "7": "Cheshire",
    "10": "Northumbria",
    "11": "Durham",
    "12": "North Yorkshire",
    "13": "West Yorkshire",
    "14": "South Yorkshire",
    "16": "Humberside",
    "17": "Cleveland",
    "20": "West Midlands",
    "21": "Staffordshire",
    "22": "West Mercia",
    "23": "Warwickshire",
    "30": "Derbyshire",
    "31": "Nottinghamshire",
    "32": "Lincolnshire",
    "33": "Leicestershire",
    "34": "Northamptonshire",
    "35": "Cambridgeshire",
    "36": "Norfolk",
    "37": "Suffolk",
    "40": "Bedfordshire",
    "41": "Hertfordshire",
    "42": "Essex",
    "43": "Thames Valley",
    "44": "Hampshire",
    "45": "Surrey",
    "46": "Kent",
    "47": "Sussex",
    "48": "City of London",
    "50": "Devon and Cornwall",
    "52": "Avon and Somerset",
    "53": "Gloucestershire",
    "54": "Wiltshire",
    "55": "Dorset",
    "60": "North Wales",
    "61": "Gwent",
    "62": "South Wales",
    "63": "Dyfed-Powys",
    "99": "Police Scotland"
    },
    "first_road_class": {
        "1": "Motorway",
        "2": "A(M) road",
        "3": "A road",
        "4": "B road",
        "5": "C road",
        "6": "Unclassified road"
    },
    "second_road_class": {
        "0": "Not at a junction",
        "1": "Motorway",
        "2": "A(M) road",
        "3": "A road",
        "4": "B road",
        "5": "C road",
        "6": "Unclassified road"
    },
    "road_type": {
        "1": "Roundabout",
        "2": "One-way street",
        "3": "Dual carriageway",
        "6": "Single carriageway",
        "7": "Slip road",
        "9": "Unknown road type"
    },
    "junction_detail_historic": {
        "0": "Not at or within 20 metres of junction",
        "1": "Roundabout",
        "2": "Mini-roundabout",
        "3": "T or staggered junction",
        "5": "Slip road",
        "6": "Crossroads",
        "7": "More than four junction arms",
        "8": "Private drive or entrance",
        "9": "Other junction"
    },
    "junction_control": {
        "1": "Authorised person",
        "2": "Automatic traffic signal",
        "3": "Stop sign",
        "4": "Give way or uncontrolled"
    },
    "light_conditions": {
        "1": "Daylight",
        "4": "Darkness — street lights lit",
        "5": "Darkness — street lights unlit",
        "6": "Darkness — no street lighting",
        "7": "Darkness — lighting unknown"
    },
    "weather_conditions": {
        "1": "Fine, no high winds",
        "2": "Raining, no high winds",
        "3": "Snowing, no high winds",
        "4": "Fine, high winds",
        "5": "Raining, high winds",
        "6": "Snowing, high winds",
        "7": "Fog or mist",
        "8": "Other weather",
        "9": "Unknown weather"
    },
    "road_surface_conditions": {
        "1": "Dry",
        "2": "Wet or damp",
        "3": "Snow",
        "4": "Frost or ice",
        "5": "Flood over 3 cm deep"
    },
    "urban_or_rural_area": {
        "1": "Urban",
        "2": "Rural",
        "3": "Unallocated"
    },
    "trunk_road_flag": {
        "1": "Trunk road",
        "2": "Non-trunk road"
    },
    "special_conditions_at_site": {
        "0": "None",
        "1": "Automatic traffic signal out",
        "2": "Automatic traffic signal defective",
        "3": "Road sign or marking defective",
        "4": "Roadworks",
        "5": "Road surface defective",
        "6": "Oil or diesel",
        "7": "Mud"
    },
    "pedestrian_crossing_human_control_historic": {
        "0": "None within 50 metres",
        "1": "Control by school crossing patrol",
        "2": "Control by other authorised person"
    },
    "pedestrian_crossing_physical_facilities_historic": {
        "0": "No physical crossing within 50 metres",
        "1": "Zebra crossing",
        "4": "Pelican, puffin, toucan or similar",
        "5": "Pedestrian phase at traffic signal",
        "7": "Footbridge or subway",
        "8": "Central refuge"
    },
"carriageway_hazards_historic": {
    "-1": "Missing / Unknown",
    "0": "None",
    "1": "Dislodged vehicle load in carriageway",
    "2": "Other object in carriageway",
    "3": "Involvement with previous collision",
    "6": "Pedestrian in carriageway, not injured",
    "7": "Animal in carriageway (except ridden horse)",
    "9": "Unknown"
},
"carriageway_hazards": {
    "-1": "Missing / Unknown",
    "0": "None",
    "11": "Defective traffic signals",
    "12": "Defective or obscured road signs or markings",
    "13": "Roadworks",
    "14": "Oil or diesel",
    "15": "Mud",
    "16": "Dislodged vehicle load in carriageway",
    "17": "Other object in carriageway",
    "18": "Involvement with previous collision",
    "19": "Pedestrian in carriageway, not injured",
    "20": "Animal in carriageway (except ridden horse)",
    "21": "Poor or defective road surface",
    "99": "Unknown (self-reported)"
},
    "junction_detail": {
    "-1": "Missing / Unknown",
    "0": "Not at or within 20 metres of junction",
    "13": "T or staggered junction",
    "16": "Crossroads",
    "17": "Junction with more than 4 arms",
    "18": "Using private drive or entrance",
    "19": "Other junction",
    "99": "Unknown (self-reported)"
    },
"pedestrian_crossing": {
    "-1": "Missing / Unknown",
    "0": "No physical crossing facility within 50 metres",
    "11": "School crossing patrol",
    "12": "Other authorised person controlling crossing",
    "13": "Zebra crossing",
    "14": "Pedestrian light crossing (Pelican, Puffin, Toucan)",
    "15": "Pedestrian phase at traffic signal junction",
    "16": "Footbridge or subway",
    "17": "Central refuge without other controls",
    "99": "Unknown (self-reported)"
},
}


def sort_code(value):
    """Sort numeric codes numerically and area codes alphabetically."""
    value = str(value)

    try:
        return (0, int(value), value)
    except ValueError:
        return (1, value.casefold(), value)


@lru_cache(maxsize=1)
def get_extra_labels():
    """
    Optional verified label dictionary.

    Expected CSV columns:
    field,code,meaning

    This allows additional official lookup descriptions
    without changing model inputs.
    """
    result = {}

    if not LABELS_PATH.exists():
        return result

    with open(LABELS_PATH, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            field = row["field"].strip()
            code = row["code"].strip()
            meaning = row["meaning"].strip()

            if field and code and meaning:
                result.setdefault(field, {})[code] = meaning

    return result


def get_code_meaning(field, code):
    code = str(code)

    if code == "-1" and field in MISSING_CATEGORY_FIELDS:
        return "Missing / Unknown"

    extra = get_extra_labels().get(field, {})

    if code in extra:
        return extra[code]

    return CODE_LABELS.get(field, {}).get(code, "")


@lru_cache(maxsize=1)
def load_artifacts():
    model = joblib.load(
        MODEL_DIR / "final_random_forest_model.pkl"
    )

    preprocessor = joblib.load(
        MODEL_DIR / "fitted_preprocessor.pkl"
    )

    frequency_maps = joblib.load(
        MODEL_DIR / "frequency_maps.pkl"
    )

    info = joblib.load(
        MODEL_DIR / "preprocessing_info.pkl"
    )

    if not hasattr(preprocessor, "transformers_"):
        raise ValueError("The preprocessor is not fitted.")

    if model.n_features_in_ != len(info["feature_names"]):
        raise ValueError("Model and preprocessing feature counts differ.")

    return model, preprocessor, frequency_maps, info


@lru_cache(maxsize=1)
def get_form_fields():
    info = joblib.load(
        MODEL_DIR / "preprocessing_info.pkl"
    )

    categorical = list(dict.fromkeys(
        info["low_cardinality_features"]
        + info["high_cardinality_features"]
    ))

    source = pd.read_csv(
        DATA_PATH,
        usecols=categorical,
        dtype={column: "string" for column in categorical},
        low_memory=False
    )

    fields = []

    for column in info["input_columns"]:
        if column == "is_weekend":
            continue

        field = {
            "name": column,
            "label": FIELD_NAMES.get(
                column,
                column.replace("_", " ").title()
            ),
            "group": (
                "essential"
                if column in ESSENTIAL_FIELDS
                else "advanced"
            ),
            "type": "select",
            "options": []
        }

        if column in ["number_of_vehicles", "speed_limit"]:
            field["type"] = "number"
            field["min"] = 1

        elif column == "month":
            field["type"] = "select"
            field["options"] = [
                {
                    "value": str(month),
                    "label": f"{month} — {calendar.month_name[month]}"
                }
                for month in range(1, 13)
            ]

        elif column == "hour":
            field["type"] = "time"

        else:
            if column == "day_of_week":
                values = [str(day) for day in range(1, 8)]
            else:
                values = (
                    source[column]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
                values = sorted(values, key=sort_code)

            if column in SEARCHABLE_FIELDS:
                field["type"] = "search"

            for value in values:
                if column == "day_of_week":
                    meaning = DAY_NAMES.get(value, "")
                else:
                    meaning = get_code_meaning(column, value)

                label = (
                    f"{value} — {meaning}"
                    if meaning else value
                )

                field["options"].append({
                    "value": value,
                    "label": label
                })

        fields.append(field)

    return fields


def predict_severity(form_data):
    model, preprocessor, frequency_maps, info = load_artifacts()

    raw_values = {}

    for field in info["input_columns"]:
        if field == "is_weekend":
            continue

        value = str(form_data.get(field, "")).strip()

        if not value:
            raise ValueError(
                f"Please provide {FIELD_NAMES.get(field, field)}."
            )

        raw_values[field] = value

    day = int(raw_values["day_of_week"])

    if day not in range(1, 8):
        raise ValueError("Invalid day of week.")

    raw_values["is_weekend"] = int(day in [1, 7])

    for field in NUMERIC_FIELDS:
        try:
            value = float(raw_values[field])
        except ValueError:
            raise ValueError(
                f"Invalid numerical value for {FIELD_NAMES[field]}."
            )

        if not np.isfinite(value):
            raise ValueError(f"Invalid value for {FIELD_NAMES[field]}.")

        if value != int(value):
            raise ValueError(
                f"{FIELD_NAMES[field]} must be a whole number."
            )

        raw_values[field] = int(value)

    if not 1 <= raw_values["month"] <= 12:
        raise ValueError("Month must be between 1 and 12.")

    if not 0 <= raw_values["hour"] <= 23:
        raise ValueError("Hour must be between 0 and 23.")

    if raw_values["number_of_vehicles"] < 1:
        raise ValueError("Number of vehicles must be at least 1.")

    if raw_values["speed_limit"] <= 0:
        raise ValueError("Speed limit must be positive.")

    allowed_options = {
        field["name"]: {
            option["value"] for option in field["options"]
        }
        for field in get_form_fields()
        if field["type"] in ("select", "search")
    }

    for field, options in allowed_options.items():
        if str(raw_values[field]) not in options:
            raise ValueError(
                f"Invalid selection for {FIELD_NAMES.get(field, field)}."
            )

    # Preserve the original input column order.
    row = pd.DataFrame(
        [raw_values],
        columns=info["input_columns"]
    )

    # Preserve existing missing-category handling.
    for field in MISSING_CATEGORY_FIELDS:
        if field in row.columns:
            row[field] = row[field].replace("-1", "Missing")

    for field in info["low_cardinality_features"]:
        row[field] = (
            row[field]
            .fillna("Missing")
            .astype(str)
        )

    # Use the saved training-derived frequency mappings.
    for field in info["high_cardinality_features"]:
        row[field] = (
            row[field]
            .fillna("Missing")
            .astype(str)
            .map(frequency_maps[field])
            .fillna(0.0)
        )

    row["is_weekend"] = row["is_weekend"].astype(int)

    processed = preprocessor.transform(row)

    if processed.shape[1] != model.n_features_in_:
        raise ValueError(
            "Processed features do not match the saved model."
        )

    probabilities = model.predict_proba(processed)[0]

    class_probabilities = {
        int(label): float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }

    predicted_class = int(model.predict(processed)[0])

    return {
        "prediction": SEVERITY_LABELS[predicted_class],
        "prediction_code": predicted_class,
        "probabilities": [
            {
                "name": SEVERITY_LABELS[class_code],
                "percentage": round(
                    class_probabilities.get(class_code, 0) * 100,
                    2
                )
            }
            for class_code in [3, 2, 1]
        ]
    }
