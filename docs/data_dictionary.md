# UHPFRC Mix-Design Data Dictionary

This dictionary defines the intended fields for the extracted research dataset. Ranges are screening ranges for data validation, not claims that every concrete mix must fall inside them. Values outside a range should be checked against the source paper rather than automatically deleted.

The CSV files currently contain headers only. No data rows were added as part of this definition.

## Core Dataset Fields

| Field | Meaning | Unit | Valid range / allowed values | Required? |
|---|---|---|---|---|
| `cement` | Portland cement or primary cementitious binder dosage | kg/m3 | > 0; normally 200-1500 | Required |
| `silica_fume` | Silica-fume dosage | kg/m3 | >= 0; normally 0-400 | Optional; required when reported |
| `fly_ash` | Fly-ash dosage | kg/m3 | >= 0; normally 0-500 | Optional; required when reported |
| `sand` | Fine aggregate dosage | kg/m3 | >= 0; normally 0-1800 | Required |
| `coarse_aggregate` | Coarse or recycled coarse aggregate dosage | kg/m3 | >= 0; normally 0-1800 | Required when the mix uses coarse aggregate; otherwise record 0 |
| `water` | Mixing water dosage | kg/m3 | > 0; normally 80-400 | Required |
| `superplasticizer` | Superplasticizer dosage | kg/m3 | >= 0; normally 0-100 | Optional; required when reported |
| `fiber_type` | Fibre material category | categorical | `polypropylene`, `steel`, `hybrid`, `basalt`, `carbon`, `polyethylene`, `none`, or source-specific value | Required for fibre analysis |
| `fiber_content_percent` | Fibre volume fraction; use for PPF when `fiber_type=polypropylene` | % by volume | 0-5% for routine validation; preserve a source value outside this range for review | Required for fibre analysis |
| `water_binder_ratio` | Water-to-total-binder mass ratio | ratio, dimensionless | > 0; normally 0.10-0.60 | Required |
| `curing_age_days` | Age at compressive-strength test | days | > 0; normally 1-365 | Required |
| `curing_temp_celsius` | Curing temperature | deg C | normally 0-100; preserve special curing regimes with a source note | Optional; required when reported |
| `specimen_type` | Geometry used for the compressive-strength test | categorical | `cube`, `cylinder`, `prism`, or source-specific value | Required for comparable strength analysis |
| `compressive_strength_MPa` | Measured compressive strength response | MPa | > 0; normally 0-300 for this dataset | Required target |
| `source_paper` | Identifier for the paper supplying the observation | text identifier | Non-empty paper ID or citation key | Required metadata |

## Requested Research Concepts

| Concept | Recommended field / current mapping | Unit | Valid range / allowed values | Required? |
|---|---|---|---|---|
| RA percentage | `recycled_aggregate_percent` (planned normalized field); currently represented, if reported, through the aggregate fields and source metadata | % of aggregate by mass or volume, as stated by source | 0-100% | Required for recycled-aggregate comparisons; otherwise optional |
| PPF percentage | `fiber_content_percent` when `fiber_type=polypropylene` | % by volume unless the paper reports another basis | 0-5% for routine validation; retain original basis and value in extraction notes | Required for PPF mixes |
| SCM type | `scm_type` (supported by the ML preprocessor but not yet a CSV column) | categorical | `silica_fume`, `fly_ash`, `slag`, `metakaolin`, `rice_husk_ash`, `other`, or `none` | Optional; required when SCM is used |
| SCM dosage | `scm_dosage_percent` (planned normalized field), or individual dosage columns such as `silica_fume` and `fly_ash` | % of binder, unless stored as kg/m3 | 0-100% when expressed as binder replacement | Optional; required when SCM is used |

## Data-Entry Rules

- Preserve the units and definitions used by each source paper in extraction notes.
- Do not infer an RA percentage, SCM dosage percentage, or fibre basis when the paper does not report it.
- Use blank values for unavailable optional measurements; use zero only when the source explicitly reports that the component is absent.
- Keep `source_paper` for traceability, but exclude it from model features to avoid source leakage.
- Record specimen geometry and curing conditions because they affect measured compressive strength.
- Keep the target column numeric and do not mix strength units in the same column.
- Before modelling, reconcile duplicate mixes, inconsistent units, and multiple strength measurements from the same experimental condition.

## Implementation Note

The current application schema contains `fiber_type` and `fiber_content_percent`, rather than separate `ppf_content_percent` and `recycled_aggregate_percent` fields. The ML preprocessing code already recognizes `scm_type` if it is added to incoming data. These normalized fields should be added to the CSV and model input schema only after the extraction format is agreed and populated consistently across all source papers.
