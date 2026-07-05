# FraudGuard Datasets

## Auto Insurance Dataset (Primary — for ML training)

Place these two files here before running `ml/train_model.py`:

| File | Size | Source |
|------|------|--------|
| `fraud_oracle.csv` | 15,420 rows | Kaggle — Auto Insurance Fraud Detection |
| `carclaims.csv` | 15,420 rows | Kaggle — Car Insurance Fraud (label cross-check) |

**Download:** https://www.kaggle.com/datasets/roshansharma/insurance-fraud-detection

Both files must be present. The training script verifies label consistency between them.

---

## Healthcare Fraud Dataset (Included)

These files are from the **Medicare Healthcare Provider Fraud Detection** dataset:

| File | Description |
|------|-------------|
| `Train_Claims.csv` | Provider-level fraud labels |
| `Train_Inpatient.csv` | Inpatient claim records |
| `Train_Beneficiary.csv` | Beneficiary demographic data |
| `Test_Claims.csv` | Test set provider labels |

**Source:** https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis

These are used for reference and future model extension to healthcare fraud detection.

---

## Dataset Schema — fraud_oracle.csv

| Column | Type | Description |
|--------|------|-------------|
| Month | string | Month of claim |
| WeekOfMonth | int | Week number (1-5) |
| DayOfWeek | string | Day claim was made |
| Make | string | Vehicle make |
| AccidentArea | string | Urban / Rural |
| DayOfWeekClaimed | string | Day claim was filed |
| MonthClaimed | string | Month claim was filed |
| WeekOfMonthClaimed | int | Week claim was filed |
| Sex | string | M / F |
| MaritalStatus | string | Single / Married / Divorced / Widow |
| Age | int | Policyholder age |
| Fault | string | Policy Holder / Third Party |
| PolicyType | string | Vehicle category + base policy |
| VehicleCategory | string | Sedan / Sport / Utility |
| VehiclePrice | string | Price range |
| FraudFound_P | int | **TARGET** — 1=Fraud, 0=Legitimate |
| PolicyNumber | int | Unique policy ID |
| RepNumber | int | Representative number |
| Deductible | int | Deductible amount |
| DriverRating | int | Driver rating (1-4) |
| Days:Policy-Accident | string | Days between policy start and accident |
| Days:Policy-Claim | string | Days between policy start and claim |
| PastNumberOfClaims | string | none / 1 / 2 to 4 / more than 4 |
| AgeOfVehicle | string | new / 2 years / ... / more than 7 |
| AgeOfPolicyHolder | string | 16 to 17 / 18 to 20 / ... / over 65 |
| PoliceReportFiled | string | Yes / No |
| WitnessPresent | string | Yes / No |
| AgentType | string | Internal / External |
| NumberOfSuppliments | string | none / 1 to 2 / 3 to 5 / more than 5 |
| AddressChange-Claim | string | no change / under 6 months / 1 year / 2 to 3 years / 4 to 8 years |
| NumberOfCars | string | 1 vehicle / 2 vehicles / ... |
| Year | int | Year of policy |
| BasePolicy | string | Liability / Collision / All Perils |
