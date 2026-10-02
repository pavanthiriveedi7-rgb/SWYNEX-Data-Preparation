# %% [markdown]
# # SWYNEX Task 1 - Data Preparation: Indian Railways (Trains + Stations)
#
# **Goal:** turn two raw, messy GeoJSON files into clean, typed, documented,
# analysis-ready tables, and record every assumption made on the way.
#
# **Dataset:** [DataMeet Indian Railways](https://github.com/datameet/railways) -
# a public, community-built dataset of Indian trains (5,208) and stations (8,990),
# including route geometry.
#
# **Why this dataset:** it is nested (GeoJSON), India-specific, and full of
# real-world problems: placeholder strings (`"?"`, `"None"`, empty), dummy
# stations, inconsistent spellings, impossible values and mixed casing.

# %%
import json
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/datameet/railways/master/"
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
REPORT_DIR = Path("reports")
for d in (RAW_DIR, OUT_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

for fname in ("trains.json", "stations.json"):
    path = RAW_DIR / fname
    if not path.exists():
        print("Downloading", fname)
        urllib.request.urlretrieve(BASE_URL + fname, path)

# %% [markdown]
# ## 1. Load the nested GeoJSON into flat tables
# Each GeoJSON *feature* has `properties` (the attributes) and a `geometry`
# (route line for trains, point for stations). I flatten both.
#
# **Derived from geometry:** number of route points and the route length in km
# (haversine distance along the polyline). Later I use this to cross-check the
# stated distance.

# %%
def haversine_km(lon1, lat1, lon2, lat2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlmb = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def polyline_km(coords):
    c = np.asarray(coords, dtype=float)
    if len(c) < 2:
        return np.nan
    return float(haversine_km(c[:-1, 0], c[:-1, 1], c[1:, 0], c[1:, 1]).sum())


with open(RAW_DIR / "trains.json") as f:
    train_feats = json.load(f)["features"]
with open(RAW_DIR / "stations.json") as f:
    station_feats = json.load(f)["features"]

trains_raw = pd.json_normalize([f["properties"] for f in train_feats])
trains_raw["route_points"] = [len(f["geometry"]["coordinates"]) for f in train_feats]
trains_raw["route_km_geo"] = [polyline_km(f["geometry"]["coordinates"]) for f in train_feats]

stations_raw = pd.json_normalize([f["properties"] for f in station_feats])
stations_raw["lon"] = [f["geometry"]["coordinates"][0] if f["geometry"] else np.nan for f in station_feats]
stations_raw["lat"] = [f["geometry"]["coordinates"][1] if f["geometry"] else np.nan for f in station_feats]

print("Trains  :", trains_raw.shape)
print("Stations:", stations_raw.shape)
trains_raw.head(3).T

# %% [markdown]
# ## 2. Audit the raw data
# In this dataset "missing" does **not** only mean `NaN`. I count real nulls
# *and* placeholder strings (`""`, `"?"`, `"None"`), because `df.isna()` alone
# would hide most of the problem.

# %%
PLACEHOLDERS = {"", "?", "none", "null", "nan", "n/a", "na", "-"}

# Identifier columns are NEVER treated as placeholders. Real example found while
# auditing: station code "NAN" (Nandgaon Road, Maharashtra) matches the placeholder
# "nan" when comparing case-insensitively, and would have been wiped out.
ID_COLS = {"code", "number", "return_train", "from_station_code", "to_station_code"}


def is_missing_raw(s: pd.Series) -> pd.Series:
    """True for real nulls and for placeholder strings (identifier columns: real nulls only)."""
    if pd.api.types.is_numeric_dtype(s) or s.name in ID_COLS:
        return s.isna()
    return s.isna() | s.astype(str).str.strip().str.lower().isin(PLACEHOLDERS)


def missing_table(df: pd.DataFrame, dataset: str, col_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dataset": dataset,
            "column": df.columns,
            col_name: [int(is_missing_raw(df[c]).sum()) for c in df.columns],
        }
    )


raw_audit = pd.concat(
    [missing_table(trains_raw, "trains", "raw_missing"),
     missing_table(stations_raw, "stations", "raw_missing")],
    ignore_index=True,
)
print("Hidden + real missing values in RAW data (only columns with problems):")
print(raw_audit[raw_audit.raw_missing > 0].to_string(index=False))

print("\nNaN-only view of trains (what a naive isna() would show):")
print(trains_raw.isna().sum()[trains_raw.isna().sum() > 0])

print("\nOther raw issues:")
print("  Duplicate train numbers :", trains_raw["number"].duplicated().sum())
print("  Duplicate station codes :", stations_raw["code"].duplicated().sum())
print("  Zero distances          :", (trains_raw["distance"] == 0).sum())
print("  Train numbers not 5 digits:", (~trains_raw["number"].str.fullmatch(r"\d{5}")).sum())
print("  'XX-' dummy stations    :", stations_raw["code"].str.startswith("XX-").sum())
print("  State spellings         :", sorted(stations_raw["state"].dropna().str.strip().unique())[:0] or "see below")
print(stations_raw["state"].value_counts().tail(12))

# %% [markdown]
# ## 3. Standardise missing values
# **Assumption 1:** every placeholder (`""`, `"?"`, `"None"`) in a *descriptive* column
# means "unknown", so all of them become a real null. Strings are stripped of
# surrounding whitespace.
#
# **Exception - identifier columns.** Station code `NAN` (Nandgaon Road) is a valid
# code, not a missing value, yet it matches the placeholder `nan`. Codes and train
# numbers are only stripped, never nulled.

# %%
trains = trains_raw.copy()
stations = stations_raw.copy()

for df in (trains, stations):
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            cleaned = df[c].astype("str").str.strip()
            if c in ID_COLS:
                df[c] = cleaned.where(df[c].notna(), np.nan)
            else:
                df[c] = cleaned.where(~cleaned.str.lower().isin(PLACEHOLDERS), np.nan)

# %% [markdown]
# ## 4. Fix data types and parse structured fields
#
# | Column | Raw | Clean | Note |
# |---|---|---|---|
# | `number` | text, e.g. `11039-Slip`, `17603-Slip2`, `14887S` | `train_id` (original, unique), `train_no` (5-digit text), `is_slip_coach`, `slip_no` | A *slip coach* is a coach detached from a running train. The number is an ID, so it stays text (keeps leading zeros) |
# | `departure`, `arrival` | `"10:40:00"` | `departure_time`, `arrival_time` (`HH:MM`) + minutes after midnight | Easier to compute with |
# | `duration_h`, `duration_m` | two floats | `duration_min` (nullable int) | One number instead of two |
# | `first_ac`... `first_class` | 0/1 ints | booleans | They are yes/no flags |
# | `type`, `zone` | text | categories | Few distinct values |

# %%
# --- Train number ----------------------------------------------------------
trains["train_id"] = trains["number"]  # original ID, unique ("11039" and "11039-Slip" are different records)
# Slip coaches appear as "11039-Slip", "17603-Slip2" and "14887S" (three spellings).
trains["is_slip_coach"] = (trains["number"].str.contains("slip", case=False, na=False)
                           | trains["number"].str.fullmatch(r"\d{5}S", na=False))
# slip_no = sequence number among the slip records of the same base train (0 = not a slip).
# Needed because e.g. 14887 has two different slips ("14887-Slip" and "14887S").
trains = trains.sort_values("train_id").reset_index(drop=True)
trains["slip_no"] = (trains[trains["is_slip_coach"]].groupby(trains["number"].str.extract(r"^(\d+)", expand=False))
                     .cumcount() + 1).reindex(trains.index).fillna(0).astype("int8")
trains["train_no"] = trains["number"].str.extract(r"^(\d+)", expand=False).str.zfill(5)
# One raw return_train value has HTML leaked into it by the original scraper:
#   'Trivandrum Jan Shatabdi" href=/train/13981/1242/59> 12081'
# Rule: use the leading digits; if there are none, use the LAST 5-digit number.
lead = trains["return_train"].str.extract(r"^(\d+)", expand=False)
tail = trains["return_train"].str.extract(r"(\d{5})\D*$", expand=False)
trains["return_train_no"] = lead.fillna(tail).str.zfill(5)
print("return_train values repaired from scraper HTML:", int((lead.isna() & tail.notna()).sum()))

# --- Time columns ----------------------------------------------------------
def parse_minutes(s: pd.Series) -> pd.Series:
    h = pd.to_numeric(s.str[:2], errors="coerce")
    m = pd.to_numeric(s.str[3:5], errors="coerce")
    return (h * 60 + m).astype("Int64")


trains["departure_min"] = parse_minutes(trains["departure"])
trains["arrival_min"] = parse_minutes(trains["arrival"])
trains["departure_time"] = trains["departure"].str[:5]
trains["arrival_time"] = trains["arrival"].str[:5]

# --- Duration and distance -------------------------------------------------
trains["duration_min"] = (trains["duration_h"] * 60 + trains["duration_m"]).astype("Int64")
trains["distance_km"] = trains["distance"].astype("Float64")

# --- Class availability ----------------------------------------------------
CLASS_COLS = {"first_ac": "1A", "second_ac": "2A", "third_ac": "3A",
              "sleeper": "SL", "chair_car": "CC", "first_class": "FC"}
for col in CLASS_COLS:
    trains[col] = trains[col].astype(bool)
trains["classes_available"] = trains[list(CLASS_COLS)].apply(
    lambda r: ",".join(code for col, code in CLASS_COLS.items() if r[col]) or "None listed", axis=1
)
trains["n_classes"] = trains[list(CLASS_COLS)].sum(axis=1).astype("int8")

print("Raw `classes` column - non-empty values:", trains["classes"].notna().sum(),
      "(it is entirely empty, so it is dropped; the flag columns carry the information)")

# %% [markdown]
# ## 5. Handle invalid values and incomplete records
#
# **Assumption 2 - quarantine, don't delete.** 15 trains have *no* times,
# distance, duration, type or zone. They carry no usable schedule, so they are
# moved to `trains_quarantine.csv` instead of being imputed. Nothing is silently lost.
#
# **Assumption 3 - zero distance is invalid.** A train cannot travel 0 km, so
# `distance_km == 0` becomes null (and a flag records it). Same for a 0-minute duration.
#
# **Assumption 4 - suspicious values are flagged, not changed.** Implausible speeds
# are kept but flagged, because I cannot tell whether distance or duration is wrong.

# %%
incomplete = trains["distance_km"].isna() & trains["duration_min"].isna() & trains["departure_min"].isna()
trains_quarantine = trains[incomplete].copy()
trains = trains[~incomplete].copy()
print(f"Quarantined {len(trains_quarantine)} incomplete trains; {len(trains)} remain")

trains["distance_was_zero"] = (trains["distance_km"] == 0).fillna(False)
trains.loc[trains["distance_km"] == 0, "distance_km"] = pd.NA
trains.loc[trains["duration_min"] == 0, "duration_min"] = pd.NA

trains["avg_speed_kmph"] = (trains["distance_km"].astype(float) / (trains["duration_min"].astype(float) / 60)).round(1)
# Indian passenger trains rarely exceed ~130 km/h; <5 km/h is not a real timetable.
trains["speed_suspect"] = ((trains["avg_speed_kmph"] > 130) | (trains["avg_speed_kmph"] < 5)).fillna(False)
print("Zero-distance rows nulled :", int(trains["distance_was_zero"].sum()))
print("Suspect-speed rows flagged:", int(trains["speed_suspect"].sum()))
trains.loc[trains["speed_suspect"], ["train_no", "name", "distance_km", "duration_min", "avg_speed_kmph"]].head(8)

# %% [markdown]
# ## 6. Consistency checks between fields
# Two independent checks that the numbers agree with each other:
# 1. **Clock check:** `arrival - departure` (mod 24h) must equal `duration` (mod 24h).
# 2. **Geometry check:** the stated distance should be close to the length of the route drawn on the map.

# %%
clock_gap = (trains["arrival_min"] - trains["departure_min"]) % 1440
dur_gap = trains["duration_min"].astype(float) % 1440
mismatch = (clock_gap.astype(float) - dur_gap).abs() > 1
print("Clock vs duration mismatches:", int(mismatch.sum()),
      "of", int(trains["duration_min"].notna().sum()))

trains["days_en_route"] = (trains["duration_min"] // 1440).astype("Int64")
trains["is_overnight"] = (trains["arrival_min"] < trains["departure_min"]) | (trains["days_en_route"] > 0)

trains["geo_ratio"] = (trains["route_km_geo"] / trains["distance_km"].astype(float)).round(3)
print("\nRoute-length / stated-distance ratio:")
print(trains["geo_ratio"].describe().round(3))

# %% [markdown]
# **Reading the ratio:** the typical train sits at ~0.97 (middle 50% between 0.95 and
# 0.99), so stated distance and map geometry agree well. The long tails are the
# interesting part. I flag only extreme disagreement (route more than 20% longer than
# the stated distance, or less than a third of it) as `distance_geo_mismatch`.
# They are flagged, not corrected, because either field could be the wrong one.

# %%
trains["distance_geo_mismatch"] = ((trains["geo_ratio"] > 1.2) | (trains["geo_ratio"] < 0.33)).fillna(False)
print("Distance/geometry mismatches flagged:", int(trains["distance_geo_mismatch"].sum()))

# %% [markdown]
# ## 7. Text standardisation
# **Assumption 5:** station names arrive in mixed case (about 88% ALL-CAPS), so
# they are converted to Title Case for consistency. Train type codes get a readable
# label; only codes I am confident about are expanded, the rest keep their raw code.

# %%
for col in ("from_station_name", "to_station_name"):
    trains[col] = trains[col].str.replace(r"\s+", " ", regex=True).str.title()
trains["name"] = trains["name"].str.replace(r"\s+", " ", regex=True)
# One train has a blank name. Names in this dataset follow "ORIGIN - DESTINATION ...",
# so I rebuild it from the (cleaned) station names and mark it as derived.
blank_name = trains["name"].isna()
trains.loc[blank_name, "name"] = trains["from_station_name"] + " - " + trains["to_station_name"]
print("Train names rebuilt from stations:", int(blank_name.sum()))

TYPE_LABELS = {
    "Pass": "Passenger", "Exp": "Express", "SF": "Superfast", "Mail": "Mail",
    "MEMU": "MEMU (electric multiple unit)", "DEMU": "DEMU (diesel multiple unit)",
    "Raj": "Rajdhani", "Drnt": "Duronto", "Shtb": "Shatabdi", "JShtb": "Jan Shatabdi",
    "GR": "Garib Rath", "SKr": "Sampark Kranti", "Toy": "Toy train",
}
trains["train_type_code"] = trains["type"]
trains["train_type"] = trains["type"].map(TYPE_LABELS).fillna(trains["type"])

# %% [markdown]
# ## 8. Missing zones: impute only if it can be validated
# 275 trains have an unknown railway `zone`. A tempting fix is "use the zone of the
# origin station", but I do not assume it works - I **test** it.
#
# **Method:** for every train with a known zone, predict its zone as the most
# common zone among *other* trains leaving from the same origin station
# (leave-one-out), then measure accuracy. I only fill the unknowns if accuracy is >= 90%.

# %%
known = trains[trains["zone"].notna()]
counts = known.groupby(["from_station_code", "zone"]).size().rename("n").reset_index()

correct = total = 0
for _, row in known.iterrows():
    sub = counts[counts["from_station_code"] == row["from_station_code"]].copy()
    sub.loc[sub["zone"] == row["zone"], "n"] -= 1
    sub = sub[sub["n"] > 0]
    if sub.empty:
        continue  # no other train from this station -> cannot predict
    total += 1
    correct += int(sub.sort_values("n", ascending=False).iloc[0]["zone"] == row["zone"])

coverage = total / len(known)
accuracy = correct / total
print(f"Leave-one-out accuracy: {accuracy:.1%}  (predictable for {coverage:.1%} of known trains)")

THRESHOLD = 0.90
mode_zone = counts.sort_values("n", ascending=False).drop_duplicates("from_station_code").set_index("from_station_code")["zone"]
trains["zone_imputed"] = False
if accuracy >= THRESHOLD:
    fill = trains["from_station_code"].map(mode_zone)
    mask = trains["zone"].isna() & fill.notna()
    trains.loc[mask, "zone"] = fill[mask]
    trains.loc[mask, "zone_imputed"] = True
    print(f"Accuracy >= {THRESHOLD:.0%}: imputed {int(mask.sum())} zones")
else:
    print(f"Accuracy < {THRESHOLD:.0%}: NOT imputing")
trains["zone"] = trains["zone"].fillna("Unknown")
print("Zones still unknown:", int((trains["zone"] == "Unknown").sum()))

# %% [markdown]
# ## 9. Clean the stations table
#
# | Issue | Decision |
# |---|---|
# | `XX-...` dummy stations (no name, no location) | Not real stations: moved to quarantine if no train uses them |
# | Blank state strings | Became nulls in step 3 |
# | `Orissa` and `Odisha` both present | **Assumption 6:** same state (renamed 2011) -> `Odisha` |
# | `Delhi NCT` | Shortened to `Delhi` |
# | `Bangladesh` as a "state" | Moved to a `country` column; those stations are kept but marked non-Indian |
# | ~half of stations have no state, zone, address or coordinates | Left null (no safe way to invent them); flagged with `has_metadata` |
# | Coordinates | Validated against India's bounding box (lat 6-38, lon 68-98) |

# %%
stations["is_dummy"] = stations["code"].str.startswith("XX-")
used_codes = set(trains["from_station_code"]) | set(trains["to_station_code"])
dummy_used = stations.loc[stations["is_dummy"], "code"].isin(used_codes).sum()
print("Dummy stations used by any train:", int(dummy_used))

stations_quarantine = stations[stations["is_dummy"] & ~stations["code"].isin(used_codes)].copy()
stations = stations.drop(stations_quarantine.index).copy()

stations["country"] = np.where(stations["state"] == "Bangladesh", "Bangladesh", "India")
stations.loc[stations["state"] == "Bangladesh", "state"] = np.nan
stations["state"] = stations["state"].replace({"Orissa": "Odisha", "Delhi NCT": "Delhi"})

stations["has_metadata"] = stations[["state", "zone", "address"]].notna().any(axis=1)
in_india = stations["lat"].between(6, 38) & stations["lon"].between(68, 98)
print("Coordinates outside India's bounding box:", int((stations["lat"].notna() & ~in_india).sum()))
stations.loc[stations["lat"].notna() & ~in_india, ["lat", "lon"]] = np.nan

stations["name"] = stations["name"].str.strip()
for c in ("state", "zone", "country"):
    stations[c] = stations[c].astype("category")

# %% [markdown]
# ## 10. Join stations onto trains
# Adds origin and destination state so trains can be analysed geographically.
# `is_interstate` is only set when both states are known.

# %%
lookup = stations.set_index("code")["state"]
trains["origin_state"] = trains["from_station_code"].map(lookup).astype("object")
trains["dest_state"] = trains["to_station_code"].map(lookup).astype("object")
both = trains["origin_state"].notna() & trains["dest_state"].notna()
trains["is_interstate"] = pd.Series(pd.NA, index=trains.index, dtype="boolean")
trains.loc[both, "is_interstate"] = trains.loc[both, "origin_state"] != trains.loc[both, "dest_state"]
for c in ("origin_state", "dest_state", "train_type", "zone", "train_type_code"):
    trains[c] = trains[c].astype("category")
print("Trains with both states known:", int(both.sum()), "of", len(trains))

# %% [markdown]
# ## 11. Select final columns and validate
# Every assertion must pass or the script stops.

# %%
FINAL_COLS = [
    "train_id", "train_no", "name", "is_slip_coach", "slip_no", "train_type_code", "train_type", "zone",
    "zone_imputed", "from_station_code", "from_station_name", "origin_state",
    "to_station_code", "to_station_name", "dest_state", "is_interstate",
    "departure_time", "arrival_time", "departure_min", "arrival_min", "duration_min",
    "days_en_route", "is_overnight", "distance_km", "avg_speed_kmph",
    "return_train_no", "classes_available", "n_classes",
    "first_ac", "second_ac", "third_ac", "sleeper", "chair_car", "first_class",
    "route_points", "route_km_geo", "geo_ratio",
    "distance_was_zero", "speed_suspect", "distance_geo_mismatch",
]
trains_clean = trains[FINAL_COLS].reset_index(drop=True)
stations_clean = stations[["code", "name", "state", "zone", "address", "country",
                           "lat", "lon", "has_metadata"]].reset_index(drop=True)

assert trains_clean["train_id"].is_unique, "duplicate train ids"
assert not trains_clean.duplicated(["train_no", "is_slip_coach", "slip_no"]).any(), "duplicate train + slip combo"
assert trains_clean["train_no"].str.fullmatch(r"\d{5}").all(), "bad train number"
assert stations_clean["code"].is_unique, "duplicate station codes"
assert trains_clean[["train_no", "name", "train_type", "zone", "departure_time", "arrival_time"]].notna().all().all()
assert trains_clean["distance_km"].dropna().gt(0).all(), "non-positive distance"
assert trains_clean["duration_min"].dropna().gt(0).all(), "non-positive duration"
assert trains_clean["from_station_code"].isin(stations_clean["code"]).all(), "unknown origin"
assert trains_clean["to_station_code"].isin(stations_clean["code"]).all(), "unknown destination"
assert len(trains_clean) + len(trains_quarantine) == len(trains_raw), "rows lost"
assert len(stations_clean) + len(stations_quarantine) == len(stations_raw), "stations lost"
print("All validation checks passed.")

# %% [markdown]
# ## 12. Before / after quality report

# %%
after_audit = pd.concat(
    [pd.DataFrame({"dataset": "trains", "column": trains_clean.columns,
                   "clean_missing": trains_clean.isna().sum().values}),
     pd.DataFrame({"dataset": "stations", "column": stations_clean.columns,
                   "clean_missing": stations_clean.isna().sum().values})],
    ignore_index=True,
)
report = raw_audit.merge(after_audit, on=["dataset", "column"], how="outer").fillna(0)
report[["raw_missing", "clean_missing"]] = report[["raw_missing", "clean_missing"]].astype(int)
report.to_csv(REPORT_DIR / "data_quality_report.csv", index=False)
print(report[(report.raw_missing > 0) | (report.clean_missing > 0)].to_string(index=False))

# %%
# Chart 1: raw missing (incl. hidden placeholders) vs naive NaN count, trains
t_raw = raw_audit[(raw_audit.dataset == "trains") & (raw_audit.raw_missing > 0)].set_index("column")["raw_missing"]
t_nan = trains_raw.isna().sum().reindex(t_raw.index)
fig, ax = plt.subplots(figsize=(8, 4))
pd.DataFrame({"Real NaN only": t_nan, "NaN + placeholders (?, None, blank)": t_raw}).sort_values(
    "NaN + placeholders (?, None, blank)").plot.barh(ax=ax, color=["#8172B2", "#C44E52"])
ax.set_xscale("symlog", linthresh=10)
ax.set_xlabel("Rows affected (log scale)")
ax.set_title("Hidden missing values in raw trains data")
plt.tight_layout()
plt.savefig(REPORT_DIR / "hidden_missing_values.png", dpi=150)
plt.close()

# Chart 2: average speed by train type (cleaned data is now analysable)
order = trains_clean.groupby("train_type", observed=True)["avg_speed_kmph"].median().sort_values().index
data = [trains_clean.loc[(trains_clean.train_type == t) & ~trains_clean.speed_suspect, "avg_speed_kmph"].dropna() for t in order]
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.boxplot(data, tick_labels=[str(t).split(" (")[0] for t in order], vert=True, showfliers=False)
ax.set_ylabel("Average speed (km/h)")
ax.set_title("Average speed by train type (cleaned, suspect rows excluded)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(REPORT_DIR / "speed_by_train_type.png", dpi=150)
plt.close()

# %% [markdown]
# ## 13. Save outputs

# %%
trains_clean.to_csv(OUT_DIR / "trains_clean.csv", index=False)
stations_clean.to_csv(OUT_DIR / "stations_clean.csv", index=False)
trains_quarantine[["number", "name", "from_station_code", "to_station_code", "type"]].to_csv(
    OUT_DIR / "trains_quarantine.csv", index=False)
stations_quarantine[["code", "name"]].to_csv(OUT_DIR / "stations_quarantine.csv", index=False)

# --- Round-trip test: station code "NAN" must survive being saved and re-read -----
# (pandas keeps uppercase "NAN" by default, but nulls "NA", "nan", "None"; reading with
#  keep_default_na=False is the safe habit for any column that holds codes.)
back = pd.read_csv(OUT_DIR / "stations_clean.csv", keep_default_na=False, na_values=[""])
assert "NAN" in set(back["code"]), "station code NAN was lost"
assert back["code"].notna().all(), "null station code after round trip"
print("Round-trip check passed: station code NAN preserved.")

summary = {
    "trains_raw": len(trains_raw), "trains_clean": len(trains_clean),
    "trains_quarantined": len(trains_quarantine),
    "stations_raw": len(stations_raw), "stations_clean": len(stations_clean),
    "stations_quarantined": len(stations_quarantine),
    "zones_imputed": int(trains_clean["zone_imputed"].sum()),
    "zones_unknown": int((trains_clean["zone"] == "Unknown").sum()),
    "zero_distances_nulled": int(trains_clean["distance_was_zero"].sum()),
    "speed_suspect": int(trains_clean["speed_suspect"].sum()),
    "distance_geo_mismatch": int(trains_clean["distance_geo_mismatch"].sum()),
    "slip_coaches": int(trains_clean["is_slip_coach"].sum()),
}
print(json.dumps(summary, indent=2))
