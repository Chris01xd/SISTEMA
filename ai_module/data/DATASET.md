# Dataset card — digital twin data

## Source

| | |
|---|---|
| **Name** | MIMIC-IV Clinical Database Demo **v2.2** + MIMIC-IV-ED Demo **v2.2** |
| **Publisher** | PhysioNet (MIT Laboratory for Computational Physiology) |
| **URLs** | https://physionet.org/content/mimic-iv-demo/2.2/ · https://physionet.org/content/mimic-iv-ed-demo/2.2/ |
| **License** | Open Database License (ODbL) v1.0: open access, **no credentials or data use agreement required** |
| **DOIs** | 10.13026/dp1f-ex47 (MIMIC-IV demo) · 10.13026/jzz5-vs76 (MIMIC-IV-ED demo) |
| **Content** | Real, de-identified records of 100 patients from Beth Israel Deaconess Medical Center (Boston): 275 hospital admissions, 140 ICU stays and 222 ED visits with triage |
| **Integrity** | Every file is verified against PhysioNet's published `SHA256SUMS.txt` before use (`digital_twin/ingest.py`) |

Citations (required by the license terms):

- Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). *MIMIC-IV Clinical Database Demo* (version 2.2). PhysioNet. https://doi.org/10.13026/dp1f-ex47
- Johnson, A., Bulgarelli, L., Pollard, T., Celi, L. A., Mark, R., & Horng, S. (2023). *MIMIC-IV-ED Demo* (version 2.2). PhysioNet. https://doi.org/10.13026/jzz5-vs76
- Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation* 101(23), e215–e220.

**Why this dataset.** The project's domain is hospital resource allocation during mass
casualty events. This dataset is public and downloadable without registration and has
published checksums. It also contains exactly the signals a bed-capacity twin needs:
admission and discharge times, ICU stays, emergency-department arrival mode and **triage
acuity (ESI)** with vital signs recorded at triage. It is also the public dataset the rest
of the platform already uses (`backend/app/datasets/`), so the AI layer reasons over the
same data as the rest of the system. Alternatives considered: UCI *Diabetes 130-US
hospitals* (large, but no timestamps, ICU or triage data, and limited to one population);
Kaggle hospital length-of-stay sets (synthetic, or need a login to download).

## Files used

| File | Used for |
|---|---|
| `hosp/admissions.csv.gz` | admission/discharge times, admission type & location, in-hospital death |
| `hosp/patients.csv.gz` | sex, `anchor_age` / `anchor_year` (age at admission) |
| `icu/icustays.csv.gz` | ICU use and ICU hours per admission |
| `ed/edstays.csv.gz` | which admissions came through the ED, arrival transport |
| `ed/triage.csv.gz` | ESI acuity and triage vital signs |

## Columns kept → processed table (`data/processed/admissions.parquet`, one row per admission)

| Column | Derivation | Role |
|---|---|---|
| `subject_id`, `hadm_id` | as published | keys (split grouping) |
| `age` | `anchor_age + (year(admittime) − anchor_year)`: the documented MIMIC-IV method | feature |
| `sex` | `gender` | feature |
| `admission_type`, `admission_location` | as published | features |
| `admit_hour` | hour of `admittime` | feature (as sin/cos), arrival profile |
| `arrived_via_ed`, `arrival_transport` | from `edstays` (`NOT_ED` when no ED visit) | features |
| `ed_acuity_esi` | triage `acuity` of the most urgent ED visit linked to the admission | feature, surge mix, priority |
| `triage_*` (temperature °F, heart rate, resp. rate, SpO₂, SBP, DBP) | triage vitals of that visit | features |
| `los_hours` | `dischtime − admittime` | **target** (regression), simulation |
| `icu_flag`, `icu_hours` | sum of `icustays.los` × 24, clipped to `los_hours` | **target** (classification), simulation |
| `died_in_hospital` | `hospital_expire_flag` | observed profile only |

## Discarded variables and why

| Variables | Reason |
|---|---|
| diagnoses (`diagnoses_icd`) | coded at discharge → **leakage** for an admission-time model |
| `dischtime`-derived, `discharge_location`, `deathtime`, `edouttime`, `disposition` | outcomes → leakage |
| `race`, `insurance`, `language`, `marital_status` | sensitive attributes with no operational meaning for bed planning (fairness risk) |
| `pain`, `chiefcomplaint` | free text, inconsistent coding (`"13"`, `"UA"`, `"Critical"`...) |
| `admit_provider_id`, `anchor_year_group`, `dod` | identifiers or post-discharge information |
| calendar dates | shifted per patient by PhysioNet for de-identification: only durations and hour of day are meaningful |

## Cleaning and validation (`digital_twin/validation.py`, `digital_twin/preprocess.py`)

1. **Raw validation**: required columns present, primary keys unique, admissions reference existing patients. Any violation stops the pipeline.
2. Admissions with missing/invalid times or `dischtime <= admittime` are dropped (0 in v2.2).
3. Ages outside 0–120 are dropped (0 in v2.2).
4. ICU hours longer than the hospital stay are clipped to the stay (10 admissions in v2.2).
5. Triage temperatures between 30 and 45 are Celsius typed into the Fahrenheit field and are converted. Vitals outside plausible ranges become missing: temperature 90–110 °F, HR 20–250, RR 4–60, SpO₂ 50–100, SBP 50–300, DBP 20–200. A DBP of 879 and a temperature of 36.5 exist in the raw triage table; they belong to ED visits not linked to an admission, so they are not in the processed table.
6. When an admission has several ED visits, the most urgent one (lowest ESI) is kept; ties go to the latest visit.
7. **Processed validation**: unique admissions, positive stays, ESI in 1–5, `icu_flag` consistent with `icu_hours`.

Every count is written to `data/processed/data_report.json`, and the processed table's
SHA-256 goes into `provenance.json` and into every trained model's metadata.

Result for v2.2: **275 admissions, 100 patients**. The SHA-256 of the processed Parquet file
depends on the Parquet writer (pyarrow) version, even when the content is identical, which
is why each environment records its own hash and each model records the hash of the exact
file it was trained on. For the same raw files, the pipeline produced identical content and
identical model metrics on Python 3.14/pyarrow 25 and Python 3.11/pyarrow 17.

## Assumptions and limitations

- **Small sample** (100 patients): statistics and model metrics are illustrative and not representative of any hospital. The test set has 52 admissions, so metrics are reported with bootstrap 95% intervals.
- The demo over-represents ICU patients (46.5% of admissions had an ICU stay).
- There is no real calendar timeline because of per-patient date shifting. The twin therefore models the hospital's **operational profile** (arrival pattern by hour, stays, ICU needs by severity) and not a replay of real days.
- Bed counts and the baseline admission rate **are not in the data**. They are explicit, configurable assumptions of the modeled hospital, labeled `source: "assumption"` everywhere.
- The ESI level is only known for admissions that came through the ED (159 of 275). ESI 4–5 patients were not admitted in the demo, so a surge mix requesting them falls back to the observed ED pool.
