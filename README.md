# SWYNEX-Indian-Railways-Data-Preparation

**SWYNEX Technologies internship - Task 1: Data Preparation**

Cleaning and documenting a messy real-world dataset: **5,208 Indian trains and 8,990 stations**, delivered as nested GeoJSON with placeholder values, dummy records, scraper leftovers and inconsistent spellings.

## Dataset

- **Source:** [DataMeet Indian Railways](https://github.com/datameet/railways) (`trains.json`, `stations.json`), a public community dataset
- **Format:** GeoJSON (attributes + route geometry), flattened into tables
- The script downloads the raw files automatically on first run (`data/raw/` is not committed)

## Why this dataset is hard (what I found in the raw data)

| Problem found | Count | How I handled it |
|---|---|---|
| Missing values hidden as text (`"?"`, `"None"`, blank). A plain `isna()` shows **0** missing zones, but 290 are really missing | 290 zones, 15 times/types, 5,208 `classes` | Counted placeholders in the audit, converted to real nulls |
| `classes` column completely empty | 5,208 | Dropped; rebuilt from the six class flag columns |
| 15 trains with no times, distance, duration, type or zone | 15 | **Quarantined** to a separate file, not imputed |
| Zero distance (impossible) | 27 | Set to null and flagged `distance_was_zero` |
| Implausible speeds (<5 or >130 km/h) | 3 | Flagged `speed_suspect`, not changed |
| Stated distance disagrees with the route drawn on the map | 121 | Flagged `distance_geo_mismatch` |
| Train numbers like `11039-Slip`, `17603-Slip2`, `14887S` (slip coaches) | 69 | Parsed into `train_no`, `is_slip_coach`, `slip_no`; original kept as unique `train_id` |
| **Real station code `NAN` matches the placeholder "nan"** | 1 | Identifier columns are protected from placeholder replacement; round-trip test added |
| Web-scraper HTML leaked into a `return_train` value | 1 | Repaired to the real number (`12081`) |
| One train with a blank name | 1 | Rebuilt from origin and destination station names |
| `Orissa` and `Odisha` both used; `Delhi NCT` | 65 + 2, 28 | Unified to `Odisha` and `Delhi` |
| `Bangladesh` stored as a state | 2 | Moved to a `country` column |
| `XX-...` dummy stations (no name or location), unused by any train | 15 | Quarantined |
| Station names ~88% ALL CAPS | most | Title Case |

## Key decisions and assumptions

1. **Placeholders mean "unknown"** in descriptive columns; **identifier columns are never nulled** (see station code `NAN`).
2. **Quarantine, don't delete.** Rows removed from the clean tables are saved in `trains_quarantine.csv` and `stations_quarantine.csv`, and a check asserts that `clean + quarantined = raw`.
3. **Imputation must be validated.** For the 275 trains with unknown `zone`, I tested "use the most common zone of other trains from the same origin station" with leave-one-out validation. Accuracy was **72.9%**, below my 90% threshold, so I did **not** impute. They stay `Unknown`.
4. **Suspicious values are flagged, not silently corrected**, because either distance or duration could be the wrong one.
5. **Missing stays missing when there is no safe source.** About half the stations have no state, zone, address or coordinates; I left them null and added `has_metadata`. Consequently `is_interstate` is only known for 1,049 trains.
6. Type codes are expanded only where I am confident (Pass, Exp, SF, Raj, Drnt, Shtb...); uncertain codes (`Del`, `Hyd`, `Klkt`) keep their raw value in `train_type`.

## Checks that passed

- Clock check: `arrival - departure` equals `duration` for **all 5,192** trains (0 mismatches)
- Stated distance vs route geometry: median ratio **0.97**, so the two sources agree for typical trains
- Sanity check on results: median speed is lowest for toy trains (13 km/h), then passenger trains (33 km/h), and highest for Rajdhani and Duronto (about 74 km/h)
- 13 automated assertions: unique IDs, valid train numbers, positive distances and durations, all stations referenced exist, no rows lost, and more

## Results

| | Raw | Clean | Quarantined |
|---|---|---|---|
| Trains | 5,208 | 5,193 | 15 |
| Stations | 8,990 | 8,975 | 15 |

Clean trains table: 40 columns including parsed times, `duration_min`, `avg_speed_kmph`, `is_overnight`, `classes_available`, origin and destination state, and quality flags.

## Project structure

```
SWYNEX-Indian-Railways-Data-Preparation/
├── data/
│   ├── raw/                     # downloaded automatically (not committed)
│   └── processed/
│       ├── trains_clean.csv
│       ├── stations_clean.csv
│       ├── trains_quarantine.csv
│       └── stations_quarantine.csv
├── reports/
│   ├── data_quality_report.csv  # missing values before vs after, per column
│   ├── hidden_missing_values.png
│   └── speed_by_train_type.png
├── data_preparation.ipynb       # executed notebook with outputs
├── data_preparation.py          # same logic as a script
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
python data_preparation.py          # or open data_preparation.ipynb
```

When reading the cleaned CSVs, keep text codes intact:

```python
pd.read_csv("data/processed/stations_clean.csv", keep_default_na=False, na_values=[""])
```

## Limitations

- Station metadata (state, coordinates) is missing for about half of all stations, which limits geographic analysis.
- The dataset is a snapshot from the community project and may not match current railway timetables.
- Zone for 275 trains remains unknown because no reliable rule was found.

## Author

Pavan - R.K. College of Engineering
