# Data Audit Notes — Customer Churn

Source: IBM Telco Customer Churn dataset (7,043 customers). Findings from the
raw audit (Task 2) before any cleaning. Each finding records the decision and
where it is applied.

## Verified baseline facts
- **Shape:** 7,043 rows × 21 columns.
- **No duplicate rows.** All 7,043 `customerID` values unique.
- **Class balance:** 73.5% `No` (5,174) / 26.5% `Yes` (1,869) churn.
  Moderately imbalanced — accuracy is a misleading metric (see README / Task 7).
- **Gender** is already clean: only `Male` (3,555) / `Female` (3,488).
  No `M`/`Male`/`male` inconsistency to resolve.

## Finding 1 — `TotalCharges` is stored as text (structural defect)
- `TotalCharges` has dtype `str`, not numeric.
- After numeric coercion, **11 rows become null**.
- Every one of those 11 rows has `tenure = 0` → they are brand-new customers
  with no bill generated yet.
- **Missingness mechanism:** NOT missing-at-random. Missing billing correlates
  perfectly with `tenure = 0`. This is a finding, not just an imputation step.
- **Decision (Task 5):** fill `TotalCharges = MonthlyCharges * tenure` for these
  rows (== `MonthlyCharges` at tenure 0, i.e. 0). Dropping 11/7043 (0.16%) is
  the alternative; filling keeps the customers and matches the domain invariant
  `TotalCharges ≈ MonthlyCharges × tenure`.

## Finding 2 — categorical encoding trap
- The 6 internet add-on columns (`OnlineSecurity`, `OnlineBackup`,
  `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`) each
  contain a `"No internet service"` value (1,526 rows) that is semantically
  identical to `"No"` for customers without internet.
- `MultipleLines` contains `"No phone service"` (682 rows).
- **Decision (Task 5):** recode `"No internet service"` → `"No"` for the six
  internet add-ons. For `MultipleLines`, `"No phone service"` is kept as its
  own meaningful value (it encodes the fact the customer has no phone service,
  which is not the same as having a phone but no second line).

## Finding 3 — no label leakage found (checked explicitly)
- No `cancellation_date`, `churn_date`, `days_since_last_login`, or
  `account_status` style columns exist (searched by name).
- `TotalCharges` is cumulative lifetime billing (≤ `MonthlyCharges × tenure`),
  so it is not a post-churn proxy — it is safe to keep.
- **Decision:** no columns dropped for leakage. The hunt is documented so the
  reasoning is auditable.

## Sanity checks that passed
- `tenure` range 0–72 (valid: 0 = new customer, 72 = 6-year tenure).
- `MonthlyCharges` 18.25–118.75 (positive, plausible).
- `TotalCharges` 18.8–8,684.8 (positive).
- No stray whitespace beyond the 11 blank `TotalCharges` cells.

## Pending work (deferred to later tasks)
- Feature engineering: encode categoricals, decide `SeniorCitizen`/binary
  handling, train/test split (Task 6), model + metric (Tasks 7–11).
