"""
MatCare CDSS  -  Maternal Health Risk Clinical Decision-Support System (Streamlit)
=================================================================================

Run it from VS Code's terminal (inside the folder that holds these files):

    python -m streamlit run maternal_risk_app.py

Files that must sit in the SAME folder as this script
-----------------------------------------------------
1. finalized_maternal_model.sav            <- saved by the last cell of the notebook (Section 14)
2. Maternal_Health_Risk_RAW_UNCLEANED.csv  <- the dataset used in the notebook

What the system does (clinician workflow)
-----------------------------------------
  1. Register the patient and enter her clinical measurements.
  2. The saved model predicts her risk level (Low / Mid / High) and an early-warning alert is checked.
  3. The system shows WHICH measurements pushed the result, WHY each one matters clinically,
     and a recommended care plan (immediate actions, follow-up, counselling, danger signs).
  4. A printable report is produced and the visit is added to the session's patient records.

How it is linked to the notebook
--------------------------------
* The notebook trains, tunes and selects the best model, then pickles it (with its settings,
  test-set metrics and the chosen XAI technique) into `finalized_maternal_model.sav`.
* This app loads that file, so whichever model the notebook selected is the one the app uses.
* The app re-applies the notebook's cleaning and 80/20 split on the CSV (same random_state)
  to rebuild the explainer's background data, so SHAP / LIME explanations match the notebook.
* The clinical reference ranges and the early-warning rule are the same as Section 13 of the notebook.

Code layout
-----------
  1. Settings & constants          6. Clinical knowledge base (why it matters + recommended actions)
  2. Clinical reference logic      7. Patient assessment & care plan
  3. Input form definition         8. UI building blocks (CSS, cards, charts, report)
  4. Example patients              9. Pages
  5. Cached loaders               10. main()
"""

from __future__ import annotations

import html
import pickle
import warnings
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline as SkPipeline

warnings.filterwarnings("ignore")

# =============================================================================
# 1. SETTINGS & CONSTANTS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "finalized_maternal_model.sav"
DATA_FILE = BASE_DIR / "Maternal_Health_Risk_RAW_UNCLEANED.csv"

APP_NAME = "MatCare CDSS"
APP_TAGLINE = "Maternal Health Risk · Clinical Decision Support"
RANDOM_STATE = 42  # must match the notebook so the same train/test split is rebuilt
LABELS = ["Low", "Mid", "High"]
RISK_COLORS = {"Low": "#1E9E6A", "Mid": "#E8A100", "High": "#D64527"}
RISK_ICONS = {"Low": "🟢", "Mid": "🟡", "High": "🔴"}
NAVY, TEAL = "#0B3C5D", "#1C7C8C"

BINARY_FEATURES = ["SmokingAlcoholExposure", "PreviousComplications"]
YES_NO_MAP = {"yes": 1, "y": 1, "1": 1, "no": 0, "n": 0, "0": 0}

# Plausible clinical limits used by the notebook's cleaning step (anything outside -> missing).
PLAUSIBLE = {"Age": (12, 60), "SystolicBP": (70, 200), "DiastolicBP": (40, 130),
             "BloodSugar": (3, 25), "BodyTemperature": (94, 106), "HeartRate": (40, 180),
             "BMI": (12, 60), "HemoglobinLevel": (5, 18), "GestationalAge": (4, 42),
             "WeightGain": (-5, 30), "PreviousPregnancies": (0, 15), "AntenatalVisitCount": (0, 20)}

FRIENDLY = {"Age": "Age", "SystolicBP": "Systolic blood pressure", "DiastolicBP": "Diastolic blood pressure",
            "BloodSugar": "Blood sugar", "BodyTemperature": "Body temperature", "HeartRate": "Heart rate",
            "BMI": "Body mass index (BMI)", "HemoglobinLevel": "Haemoglobin level",
            "GestationalAge": "Gestational age", "SmokingAlcoholExposure": "Smoking / alcohol exposure",
            "PreviousComplications": "Previous complications", "WeightGain": "Weight gain in pregnancy",
            "PreviousPregnancies": "Previous pregnancies", "AntenatalVisitCount": "Antenatal visits so far"}

# =============================================================================
# 2. CLINICAL REFERENCE LOGIC  (ported unchanged from notebook, Section 13)
# =============================================================================
FIXED_RANGES = {  # feature: (lowest normal, highest normal, text)
    "Age": (18, 34, "18-34 years"),
    "SystolicBP": (90, 139, "90-139 mmHg"),
    "DiastolicBP": (60, 89, "60-89 mmHg"),
    "BloodSugar": (3.9, 7.8, "3.9-7.8 mmol/L (non-fasting reading)"),
    "BodyTemperature": (97.0, 99.0, "97.0-99.0 °F"),
    "HeartRate": (60, 100, "60-100 bpm at rest"),
    "BMI": (18.5, 24.99, "18.5-24.9 kg/m²"),
    "PreviousPregnancies": (0, 4, "0-4 previous pregnancies"),
}
ABOVE_NOTES = {  # checked from the largest threshold down
    "Age": [(35, "advanced maternal age")],
    "SystolicBP": [(160, "severe-range hypertension"), (140, "hypertension range")],
    "DiastolicBP": [(110, "severe-range hypertension"), (90, "hypertension range")],
    "BloodSugar": [(11.1, "diabetes range"), (10.0, "reaches the 1-hour gestational-diabetes cut-off"),
                   (7.8, "raised blood sugar")],
    "BodyTemperature": [(100.4, "fever"), (99.0, "slightly raised temperature")],
    "HeartRate": [(100, "tachycardia")],
    "BMI": [(30, "obese"), (25, "overweight")],
    "PreviousPregnancies": [(5, "grand multiparity")],
}
BELOW_NOTES = {"Age": "adolescent pregnancy", "SystolicBP": "low blood pressure", "DiastolicBP": "low blood pressure",
               "BloodSugar": "low blood sugar", "BodyTemperature": "below normal temperature",
               "HeartRate": "slow heart rate (bradycardia)", "BMI": "underweight"}
BINARY_NOTES = {"SmokingAlcoholExposure": "smoking / alcohol exposure present",
                "PreviousComplications": "previous pregnancy complications"}
WHO_CONTACT_WEEKS = (12, 20, 26, 30, 34, 36, 38, 40)  # WHO 2016: 8 antenatal contacts
IOM_TOTAL = {"underweight": (12.5, 18), "normal weight": (11.5, 16),  # kg over the whole pregnancy (IOM 2009)
             "overweight": (7, 11.5), "obese": (5, 9)}
IOM_RATE = {"underweight": (0.44, 0.58), "normal weight": (0.35, 0.50),  # kg / week, trimesters 2-3 (IOM 2009)
            "overweight": (0.23, 0.33), "obese": (0.17, 0.27)}


def bmi_group(bmi: float) -> str:
    """BMI category used by the IOM weight-gain guideline."""
    return "underweight" if bmi < 18.5 else "normal weight" if bmi < 25 else "overweight" if bmi < 30 else "obese"


def expected_weight_gain(bmi: float, gest_age: float) -> tuple[float, float]:
    """IOM: 0.5-2.0 kg in the first 13 weeks, then a BMI-specific weekly rate."""
    lo_rate, hi_rate = IOM_RATE[bmi_group(bmi)]
    first, extra = min(gest_age, 13) / 13, max(gest_age - 13, 0)
    return 0.5 * first + extra * lo_rate, 2.0 * first + extra * hi_rate


def clinical_status(feature: str, value: float, patient: dict) -> tuple[str, str, str]:
    """Return (status, normal-range text, note). status = 'below normal' | 'normal' | 'above normal'."""
    v = float(value)
    if feature in FIXED_RANGES:
        lo, hi, txt = FIXED_RANGES[feature]
        if v < lo:
            return "below normal", txt, BELOW_NOTES.get(feature, "")
        if v > hi:
            return "above normal", txt, next((n for t, n in ABOVE_NOTES.get(feature, []) if v >= t), "")
        return "normal", txt, ""

    ga = float(patient["GestationalAge"])
    if feature == "HemoglobinLevel":
        lo = 10.5 if 14 <= ga < 28 else 11.0  # 2nd-trimester cut-off is 10.5, otherwise 11.0
        txt = f"{lo:.1f} g/dL or more (anaemia below this)"
        if v < lo:
            return "below normal", txt, "severe anaemia" if v < 7 else "moderate anaemia" if v < 10 else "mild anaemia"
        return "normal", txt, ""
    if feature == "GestationalAge":
        txt = "up to 41 weeks (term = 37-41 weeks)"
        return ("above normal", txt, "post-term pregnancy") if v >= 42 else ("normal", txt, "")
    if feature == "WeightGain":
        grp = bmi_group(float(patient["BMI"]))
        lo, _ = expected_weight_gain(float(patient["BMI"]), ga)  # minimum expected by this week
        hi = IOM_TOTAL[grp][1]  # maximum for the whole pregnancy
        txt = f"at least {lo:.1f} kg by week {ga:g} and no more than {hi:g} kg in total, for a {grp} BMI (IOM)"
        if v < lo:
            return "below normal", txt, "less weight gain than recommended"
        if v > hi:
            return "above normal", txt, "more weight gain than recommended"
        return "normal", txt, ""
    if feature == "AntenatalVisitCount":
        need = sum(ga >= w for w in WHO_CONTACT_WEEKS)
        txt = f"at least {need} visit(s) by week {ga:g} (WHO: 8 contacts at weeks 12, 20, 26, 30, 34, 36, 38, 40)"
        return ("below normal", txt, "fewer antenatal visits than recommended") if v < need else ("normal", txt, "")
    if feature in BINARY_NOTES:
        return ("above normal", "No (0)", BINARY_NOTES[feature]) if v >= 1 else ("normal", "No (0)", "")
    raise KeyError(feature)


# =============================================================================
# 3. INPUT FORM DEFINITION
# =============================================================================
# kind: "int" / "float" use number inputs, "binary" uses a Yes/No selector.
FEATURE_SPECS = {
    "Age": dict(kind="int", min=12, max=60, default=28, help="Mother's age in years."),
    "BMI": dict(kind="float", min=12.0, max=60.0, default=24.0, step=0.1, help="Body mass index in kg/m²."),
    "GestationalAge": dict(kind="int", min=4, max=42, default=24, help="Weeks of pregnancy completed."),
    "PreviousPregnancies": dict(kind="int", min=0, max=15, default=1, help="Number of earlier pregnancies."),
    "AntenatalVisitCount": dict(kind="int", min=0, max=20, default=5, help="Antenatal clinic visits so far."),
    "SystolicBP": dict(kind="int", min=70, max=200, default=115, help="Upper blood-pressure reading (mmHg)."),
    "DiastolicBP": dict(kind="int", min=40, max=130, default=75, help="Lower blood-pressure reading (mmHg)."),
    "HeartRate": dict(kind="int", min=40, max=180, default=78, help="Resting heart rate (beats per minute)."),
    "BodyTemperature": dict(kind="float", min=94.0, max=106.0, default=98.4, step=0.1, help="Temperature in °F."),
    "BloodSugar": dict(kind="float", min=3.0, max=25.0, default=7.0, step=0.1,
                       help="Blood glucose in mmol/L (treated as a non-fasting reading)."),
    "HemoglobinLevel": dict(kind="float", min=5.0, max=18.0, default=12.0, step=0.1, help="Haemoglobin in g/dL."),
    "WeightGain": dict(kind="float", min=-5.0, max=30.0, default=8.0, step=0.1,
                       help="Total weight gained so far in this pregnancy (kg)."),
    "SmokingAlcoholExposure": dict(kind="binary", default="No", help="Is the mother exposed to smoking or alcohol?"),
    "PreviousComplications": dict(kind="binary", default="No", help="Any complications in earlier pregnancies?"),
}
# Unit shown beside each input label so the clinician always knows what to enter.
INPUT_UNITS = {"Age": "years", "BMI": "kg/m²", "GestationalAge": "weeks", "PreviousPregnancies": "count",
               "AntenatalVisitCount": "visits", "SystolicBP": "mmHg", "DiastolicBP": "mmHg", "HeartRate": "bpm",
               "BodyTemperature": "°F", "BloodSugar": "mmol/L", "HemoglobinLevel": "g/dL", "WeightGain": "kg"}
FORM_GROUPS = [
    ("👩 Obstetric profile", ["Age", "BMI", "GestationalAge", "PreviousPregnancies", "AntenatalVisitCount"]),
    ("🩺 Vital signs", ["SystolicBP", "DiastolicBP", "HeartRate", "BodyTemperature", "BloodSugar"]),
    ("🧪 Labs & history", ["HemoglobinLevel", "WeightGain", "SmokingAlcoholExposure", "PreviousComplications"]),
]

# =============================================================================
# 4. EXAMPLE PATIENTS  (the three cases from the notebook, Section 13)
# =============================================================================
EXAMPLES = {
    "🔴 Demo: high-risk looking": {"Age": 36, "SystolicBP": 142, "DiastolicBP": 94, "BloodSugar": 12.5,
                                  "BodyTemperature": 99.4, "HeartRate": 88, "BMI": 32.0, "HemoglobinLevel": 9.6,
                                  "GestationalAge": 28, "SmokingAlcoholExposure": 0, "PreviousComplications": 1,
                                  "WeightGain": 7.0, "PreviousPregnancies": 3, "AntenatalVisitCount": 3},
    "🟢 Demo: low-risk looking": {"Age": 23, "SystolicBP": 105, "DiastolicBP": 68, "BloodSugar": 6.8,
                                 "BodyTemperature": 98.1, "HeartRate": 70, "BMI": 22.5, "HemoglobinLevel": 12.6,
                                 "GestationalAge": 22, "SmokingAlcoholExposure": 0, "PreviousComplications": 0,
                                 "WeightGain": 8.0, "PreviousPregnancies": 1, "AntenatalVisitCount": 8},
    "🟡 Demo: borderline": {"Age": 30, "SystolicBP": 120, "DiastolicBP": 78, "BloodSugar": 8.2,
                           "BodyTemperature": 98.6, "HeartRate": 76, "BMI": 27.0, "HemoglobinLevel": 11.3,
                           "GestationalAge": 26, "SmokingAlcoholExposure": 0, "PreviousComplications": 0,
                           "WeightGain": 9.0, "PreviousPregnancies": 2, "AntenatalVisitCount": 6},
}


# =============================================================================
# 5. CACHED LOADERS  (model file, dataset, explainer)
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_artifact(path: str) -> dict:
    """Load the dictionary the notebook saved with pickle (model + settings + metrics)."""
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner=False)
def load_clean_data(path: str) -> pd.DataFrame:
    """Re-apply the notebook's cleaning (Section 2b): tidy labels, Yes/No -> 0/1, drop duplicates,
    turn impossible values into missing. Missing values are NOT filled here (the model's imputer does that)."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["RiskLevel"] = df["RiskLevel"].str.strip().str.lower()
    for col in BINARY_FEATURES:
        df[col] = df[col].astype(str).str.strip().str.lower().map(YES_NO_MAP)
    df = df.drop_duplicates().reset_index(drop=True)
    for col, (lo, hi) in PLAUSIBLE.items():
        bad = df[col].notna() & ~df[col].between(lo, hi)
        df.loc[bad, col] = np.nan
    df["RiskLevel"] = df["RiskLevel"].map({"low risk": 0, "mid risk": 1, "high risk": 2})
    return df


def training_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """The notebook's 80 % training split (same stratification and random_state)."""
    x_train, _, _, _ = train_test_split(df[features], df["RiskLevel"], test_size=0.2,
                                        random_state=RANDOM_STATE, stratify=df["RiskLevel"])
    return x_train


class LocalExplainer:
    """Explains ONE patient with the XAI technique the notebook selected (SHAP or LIME).

    `contributions(x, class_index)` returns how strongly each feature pushed the prediction
    toward that class (positive = pushed toward it, negative = pushed away).
    """

    def __init__(self, model: SkPipeline, x_train: pd.DataFrame, features: list[str],
                 classes: list[str], method: str):
        self.features, self.method = features, method
        self.prep = SkPipeline(model.steps[:-1])  # imputer + scaler
        self.clf = model.steps[-1][1]
        background = pd.DataFrame(self.prep.transform(x_train), columns=features)

        if method == "LIME":
            from lime.lime_tabular import LimeTabularExplainer
            self._lime = LimeTabularExplainer(background.values, feature_names=features, class_names=classes,
                                              mode="classification", random_state=RANDOM_STATE)
        else:
            self._explain = self._build_shap(background)

    def _build_shap(self, background: pd.DataFrame):
        import shap
        if isinstance(self.clf, (RandomForestClassifier, GradientBoostingClassifier)):
            try:  # fast and exact for tree models (not always supported for multi-class Gradient Boosting)
                tree = shap.TreeExplainer(self.clf)
                fn = lambda data: tree(data, check_additivity=False)  # noqa: E731
                fn(background.iloc[:3])
                return fn
            except Exception:
                pass
        generic = shap.Explainer(self.clf.predict_proba, background.sample(100, random_state=RANDOM_STATE))
        return lambda data: generic(data)

    def contributions(self, x: pd.DataFrame, pred: int) -> np.ndarray:
        z = pd.DataFrame(self.prep.transform(x), columns=self.features)
        if self.method == "LIME":
            exp = self._lime.explain_instance(z.values[0], self.clf.predict_proba, labels=(pred,),
                                              num_features=len(self.features), num_samples=1000)
            w = np.zeros(len(self.features))
            for j, wt in exp.as_map()[pred]:
                w[j] = wt
            return w
        return self._explain(z).values[0, :, pred]


@st.cache_resource(show_spinner="Preparing the explanation engine (first run only)…")
def build_explainer(path_model: str, path_data: str) -> LocalExplainer:
    art = load_artifact(path_model)
    df = load_clean_data(path_data)
    x_train = training_features(df, art["features"])
    return LocalExplainer(art["model"], x_train, art["features"], art["classes"], art.get("xai_method", "SHAP"))


# =============================================================================
# 6. CLINICAL KNOWLEDGE BASE  (why each factor matters + recommended actions)
# =============================================================================
# These are general, guideline-style prompts (WHO / ACOG / NICE-type antenatal care) written as decision
# support. They are NOT a prescription: the treating clinician applies local protocols and judgement.

# One sentence per measurement: why an abnormal value matters in pregnancy.
WHY_IT_MATTERS = {
    "SystolicBP": "Raised blood pressure in pregnancy is linked to pre-eclampsia, placental abruption, poor fetal growth and preterm birth.",
    "DiastolicBP": "Raised blood pressure in pregnancy is linked to pre-eclampsia, placental abruption, poor fetal growth and preterm birth.",
    "BloodSugar": "High glucose raises the risk of gestational diabetes, a large baby, birth complications and neonatal hypoglycaemia.",
    "BodyTemperature": "Fever may signal infection (e.g. malaria, urinary or genital tract infection) which can trigger preterm labour or sepsis.",
    "HeartRate": "An abnormal resting heart rate can point to anaemia, infection, dehydration, thyroid disease or a cardiac problem.",
    "HemoglobinLevel": "Anaemia reduces oxygen delivery to the baby and increases the risk of preterm birth, low birth weight and heavy bleeding at delivery.",
    "BMI": "BMI outside the normal range is associated with hypertension, diabetes, growth problems and difficult delivery.",
    "Age": "Very young and older mothers have higher rates of hypertensive disorders, preterm birth and delivery complications.",
    "GestationalAge": "A pregnancy beyond term needs closer fetal monitoring because stillbirth risk rises.",
    "WeightGain": "Weight gain outside the recommended range is linked to poor fetal growth (too little) or diabetes, hypertension and difficult delivery (too much).",
    "AntenatalVisitCount": "Fewer antenatal contacts mean problems such as hypertension or anaemia may go undetected.",
    "SmokingAlcoholExposure": "Smoking and alcohol are linked to growth restriction, preterm birth, placental problems and stillbirth.",
    "PreviousComplications": "A history of pregnancy complications strongly predicts recurrence in the current pregnancy.",
    "PreviousPregnancies": "Grand multiparity increases the risk of malpresentation, obstructed labour and postpartum haemorrhage.",
}

# Groups that share the same advice (so systolic + diastolic give one set of actions, not two).
ACTION_GROUP = {"SystolicBP": "BP", "DiastolicBP": "BP"}

# Recommended actions: (group or feature, "above"/"below") -> list of (category, text).
# category: "Immediate" = act at this visit | "Follow-up" = monitoring/tests | "Counselling" = advice to the woman.
ACTIONS = {
    ("BP", "above"): [
        ("Immediate", "Repeat the blood pressure after 15 minutes' rest with a correctly sized cuff to confirm the reading."),
        ("Immediate", "Ask about headache, visual disturbance, upper-abdominal pain and swelling; test urine for protein."),
        ("Follow-up", "Evaluate for gestational hypertension / pre-eclampsia (BP twice weekly or as the protocol directs; lab tests if available)."),
        ("Counselling", "Advise rest, adequate fluids, and to return immediately if headache, blurred vision or convulsions occur."),
    ],
    ("BP", "below"): [
        ("Immediate", "Recheck lying and standing; ask about dizziness, bleeding, vomiting or poor intake."),
        ("Counselling", "Advise adequate fluids and meals; rise slowly from lying or sitting."),
    ],
    ("BloodSugar", "above"): [
        ("Immediate", "Confirm with a repeat or fasting glucose / oral glucose tolerance test as per local protocol."),
        ("Follow-up", "If gestational diabetes is confirmed, refer to diabetes-in-pregnancy care and monitor fetal growth."),
        ("Counselling", "Give dietary advice (reduce refined sugars, regular balanced meals) and encourage safe daily activity."),
    ],
    ("BloodSugar", "below"): [
        ("Immediate", "Check for symptoms (sweating, shaking, confusion); give oral glucose if symptomatic and recheck."),
        ("Counselling", "Advise regular meals and snacks; do not skip meals."),
    ],
    ("BodyTemperature", "above"): [
        ("Immediate", "Look for the source of fever: test for malaria where endemic, check urine for infection, examine for other signs."),
        ("Immediate", "Treat the cause per protocol; refer if fever persists, or if there is abdominal pain, abnormal discharge or reduced fetal movement."),
        ("Counselling", "Advise fluids and rest; return urgently if the fever persists or worsens."),
    ],
    ("BodyTemperature", "mild"): [
        ("Follow-up", "Recheck the temperature at this visit and the next; ask about symptoms of infection (chills, pain on passing urine, abnormal discharge)."),
        ("Counselling", "Advise fluids and rest; return if the temperature reaches 100.4 °F or symptoms appear."),
    ],
    ("BodyTemperature", "below"): [
        ("Immediate", "Recheck the temperature; keep the patient warm and look for signs of sepsis or shock."),
    ],
    ("HeartRate", "above"): [
        ("Immediate", "Recheck after resting; look for fever, pain, bleeding or dehydration."),
        ("Follow-up", "Check haemoglobin; if tachycardia persists, evaluate for infection, thyroid or cardiac causes (ECG where available)."),
    ],
    ("HeartRate", "below"): [
        ("Immediate", "Recheck the pulse; ask about dizziness, fainting or medication that slows the heart."),
        ("Follow-up", "Arrange an ECG if symptoms are present or the rate stays low."),
    ],
    ("HemoglobinLevel", "below"): [
        ("Immediate", "Confirm with a repeat haemoglobin and examine for pallor, breathlessness or heart failure signs."),
        ("Follow-up", "Start / continue iron and folic acid; investigate causes (malaria, hookworm, poor diet); recheck in 4 weeks."),
        ("Counselling", "Advise iron-rich foods (beans, leafy greens, meat, fish) with vitamin C; take tablets as prescribed."),
    ],
    ("BMI", "above"): [
        ("Follow-up", "Screen for gestational diabetes and hypertension; monitor weight gain at each visit."),
        ("Counselling", "Give nutrition and safe-activity advice; discuss delivery planning with the obstetric team."),
    ],
    ("BMI", "below"): [
        ("Follow-up", "Do a nutritional assessment, consider supplementation, and monitor fetal growth."),
        ("Counselling", "Advise energy- and protein-rich meals and regular antenatal follow-up."),
    ],
    ("Age", "above"): [
        ("Follow-up", "Offer enhanced surveillance (blood pressure, glucose, fetal growth) and discuss birth planning."),
    ],
    ("Age", "below"): [
        ("Follow-up", "Provide adolescent-friendly counselling, nutrition support and closer follow-up; plan skilled-attendant delivery."),
    ],
    ("GestationalAge", "above"): [
        ("Immediate", "Assess fetal wellbeing (movements, heart rate) and plan induction or delivery per protocol."),
    ],
    ("WeightGain", "above"): [
        ("Counselling", "Review diet and activity; avoid dieting during pregnancy; check for oedema if the gain is sudden."),
    ],
    ("WeightGain", "below"): [
        ("Follow-up", "Review nutrition and appetite, rule out vomiting or illness, and monitor fetal growth."),
        ("Counselling", "Advise small frequent nutritious meals."),
    ],
    ("AntenatalVisitCount", "below"): [
        ("Follow-up", "Schedule the missed WHO contacts and trace / remind the patient of her next appointment."),
        ("Counselling", "Explain why regular antenatal visits protect mother and baby."),
    ],
    ("SmokingAlcoholExposure", "above"): [
        ("Counselling", "Give a brief intervention: advise stopping smoking / alcohol and avoiding second-hand smoke; refer to cessation support."),
    ],
    ("PreviousComplications", "above"): [
        ("Follow-up", "Review the earlier pregnancy records and plan delivery at a facility with obstetric and emergency care."),
        ("Follow-up", "Clinician to consider preventive measures suited to the history (e.g. low-dose aspirin for earlier pre-eclampsia)."),
    ],
    ("PreviousPregnancies", "above"): [
        ("Follow-up", "Plan facility delivery with a skilled attendant and prepare for prevention of postpartum haemorrhage."),
    ],
}

# Extra urgent action when a finding is in a dangerous range (matched on the clinical note).
RED_FLAGS = {
    "severe-range hypertension": "BP is in the SEVERE range: treat as an obstetric emergency and refer / escalate immediately per protocol.",
    "severe anaemia": "Haemoglobin is SEVERELY low: urgent referral (possible transfusion); do not delay.",
    "fever": "Fever recorded: examine for sepsis and malaria and start treatment or referral without delay.",
    "diabetes range": "Glucose is in the diabetes range: confirm urgently and arrange early specialist review.",
    "post-term pregnancy": "Pregnancy is post-term: arrange urgent fetal assessment and a delivery plan.",
}

# What to do for each overall level of risk.
LEVEL_PLAN = {
    "Urgent": ("Urgent", "#D64527",
               "Escalate now. Arrange same-day review by an obstetrician or referral to a higher-level facility, "
               "stabilise the patient, and document what was done."),
    "Priority": ("Priority", "#E8A100",
                 "The model flags this Mid-risk patient as likely to progress. Ask a senior clinician to review her promptly, "
                 "shorten the interval to her next visit and treat the flagged factors."),
    "Enhanced": ("Enhanced follow-up", "#C98A00",
                 "Manage the flagged factors, book a closer follow-up than routine care and reassess the risk at the next visit."),
    "Routine": ("Routine care", "#1E9E6A",
                "No concerning pattern found. Continue routine antenatal care (WHO: 8 contacts) and reassess at each visit."),
}
DANGER_SIGNS = ["Vaginal bleeding or leaking of fluid", "Severe headache or blurred vision", "Fits / convulsions",
                "Severe abdominal pain", "Fever or chills", "Reduced or absent baby movements",
                "Swelling of the face or hands, or breathlessness"]


def build_care_plan(result: dict, alert: bool, patient: dict) -> dict:
    """Turn the abnormal findings + risk level into a structured care plan for the clinician."""
    flagged = [d for d in result["by_push"] if d["status"] != "normal"]      # strongest push first
    seen, buckets, red_flags = set(), {"Immediate": [], "Follow-up": [], "Counselling": []}, []

    for d in flagged:
        direction = "above" if d["status"] == "above normal" else "below"
        if d["feature"] == "BodyTemperature" and direction == "above" and d["note"] != "fever":
            direction = "mild"          # 99.0-100.3 °F: slightly raised, not yet a fever
        group = ACTION_GROUP.get(d["feature"], d["feature"])
        if d["note"] in RED_FLAGS and RED_FLAGS[d["note"]] not in red_flags:
            red_flags.append(RED_FLAGS[d["note"]])
        if (group, direction) in seen:
            continue
        seen.add((group, direction))
        label = "Blood pressure" if group == "BP" else FRIENDLY[d["feature"]]
        for category, text in ACTIONS.get((group, direction), []):
            buckets[category].append((label, text))

    if result["risk"] == "High" or red_flags:
        level = "Urgent"
    elif alert:
        level = "Priority"
    elif result["risk"] == "Mid":
        level = "Enhanced"
    else:
        level = "Routine"
    title, color, summary = LEVEL_PLAN[level]
    return {"level": level, "title": title, "color": color, "summary": summary, "red_flags": red_flags,
            "buckets": buckets, "danger_signs": DANGER_SIGNS}


# =============================================================================
# 7. PATIENT ASSESSMENT
# =============================================================================
def assess_patient(patient: dict, art: dict, explainer: LocalExplainer) -> dict:
    """Predict the risk level and work out which abnormal measurements drove the result."""
    features = art["features"]
    x = pd.DataFrame([patient])[features].astype(float)
    train_range = pd.DataFrame(art["train_range"])

    notes = [f"{FRIENDLY[f]} = {x.at[0, f]:g} is outside the range the model was trained on "
             f"({train_range.at['min', f]:g} to {train_range.at['max', f]:g}), so treat the result with extra care."
             for f in features if not train_range.at["min", f] <= x.at[0, f] <= train_range.at["max", f]]

    prob = art["model"].predict_proba(x)[0]
    pred = int(prob.argmax())
    contrib = explainer.contributions(x, pred)
    push = max(contrib[contrib > 0].sum(), 1e-12)  # total push toward the predicted class

    findings = []
    for f, c in zip(features, contrib):
        status, normal_txt, note = clinical_status(f, x.at[0, f], patient)
        findings.append({"feature": f, "value": float(x.at[0, f]), "unit": art["units"][f], "status": status,
                         "normal": normal_txt, "note": note, "contribution": float(c),
                         "share": float(max(c, 0) / push * 100)})
    by_push = sorted(findings, key=lambda d: -d["contribution"])
    return {
        "pred": pred, "risk": art["classes"][pred], "prob": prob, "notes": notes, "all": findings,
        "drivers": [d for d in by_push if d["status"] != "normal" and d["share"] >= 5],       # abnormal AND pushing
        "others": [d for d in by_push if d["status"] != "normal" and d["share"] < 5],         # abnormal, little push
        "supporting": [d for d in by_push if d["status"] == "normal" and d["share"] >= 5][:3],  # normal, still helped
        "by_push": by_push,
    }


def alert_flag(result: dict, threshold: float) -> bool:
    """Early-warning rule: High risk, or Mid risk whose High-risk probability reaches the threshold."""
    return result["pred"] == 2 or (result["pred"] == 1 and result["prob"][2] >= threshold)


def show_value(d: dict) -> str:
    """Readable value: Yes/No for the two yes/no features, otherwise number + unit."""
    if d["feature"] in BINARY_NOTES:
        return "Yes" if d["value"] >= 1 else "No"
    return f"{d['value']:g} {d['unit']}"


def explain_driver(d: dict, risk: str) -> str:
    """One readable sentence on why this measurement mattered for this patient."""
    note = f" ({d['note']})" if d["note"] else ""
    return (f"{FRIENDLY[d['feature']]} is {show_value(d)}, which is {d['status']}{note}; normal is {d['normal']}. "
            f"It supplied about {d['share']:.0f}% of the push toward {risk}-risk. {WHY_IT_MATTERS[d['feature']]}")


# =============================================================================
# 8. UI BUILDING BLOCKS
# =============================================================================
CSS = f"""
<style>
.block-container {{ padding-top: 1rem; max-width: 1240px; }}
#MainMenu, footer {{ visibility: hidden; }}
.topbar {{ background: linear-gradient(100deg, {NAVY} 0%, {TEAL} 100%); color: white; border-radius: 14px;
           padding: .95rem 1.4rem; display: flex; justify-content: space-between; align-items: center;
           margin-bottom: 1rem; box-shadow: 0 4px 14px rgba(11,60,93,.22); }}
.topbar .brand {{ font-size: 1.35rem; font-weight: 800; letter-spacing: .02em; }}
.topbar .sub {{ font-size: .85rem; opacity: .85; }}
.topbar .meta {{ text-align: right; font-size: .85rem; opacity: .92; line-height: 1.4; }}
.pagetitle {{ font-size: 1.5rem; font-weight: 700; margin: .2rem 0 .1rem 0; color: {NAVY}; }}
.pagesub {{ opacity: .7; margin-bottom: .8rem; }}
.section {{ font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: {TEAL};
            margin: .2rem 0 .5rem 0; }}
.patientbar {{ border: 1px solid rgba(128,128,128,.3); border-left: 6px solid {TEAL}; border-radius: 10px;
               padding: .7rem 1rem; display: flex; flex-wrap: wrap; gap: 1.8rem; margin-bottom: 1rem;
               background: color-mix(in srgb, {TEAL} 6%, transparent); }}
.patientbar div span {{ display: block; font-size: .7rem; text-transform: uppercase; letter-spacing: .08em; opacity: .65; }}
.patientbar div b {{ font-size: 1.02rem; }}
.risk-card {{ border-radius: 14px; padding: 1.2rem 1.4rem; border: 2px solid var(--c);
              background: color-mix(in srgb, var(--c) 11%, transparent); }}
.risk-card .tag {{ font-size: .75rem; letter-spacing: .14em; text-transform: uppercase; opacity: .75; }}
.risk-card .big {{ font-size: 2.5rem; font-weight: 800; line-height: 1.1; color: var(--c); }}
.risk-card .sub {{ font-size: .98rem; opacity: .85; margin-top: .3rem; }}
.plan-banner {{ border-radius: 12px; padding: .9rem 1.1rem; border-left: 8px solid var(--c);
                background: color-mix(in srgb, var(--c) 10%, transparent); margin: .6rem 0 .4rem 0; }}
.plan-banner b {{ color: var(--c); font-size: 1.1rem; }}
.pill {{ display: inline-block; padding: .1rem .6rem; border-radius: 999px; font-size: .74rem; font-weight: 700; color: white; }}
.pill.above {{ background: #D64527; }} .pill.below {{ background: #3B6FB6; }} .pill.normal {{ background: #1E9E6A; }}
.act {{ border: 1px solid rgba(128,128,128,.28); border-radius: 10px; padding: .55rem .85rem; margin-bottom: .45rem; }}
.act .lab {{ font-size: .7rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; opacity: .65; }}
.step {{ border-radius: 12px; padding: 1rem 1.1rem; border: 1px solid rgba(128,128,128,.28);
         background: color-mix(in srgb, {TEAL} 7%, transparent); height: 100%; }}
.step b {{ color: {TEAL}; }}
.foot {{ text-align: center; opacity: .55; font-size: .82rem; margin-top: 2rem; }}
</style>
"""


def topbar(clinician: str) -> None:
    """Header strip shown on every page: system name, clinician and today's date."""
    st.markdown(
        f'<div class="topbar"><div><div class="brand">🩺 {APP_NAME}</div><div class="sub">{APP_TAGLINE}</div></div>'
        f'<div class="meta">Clinician: <b>{html.escape(clinician or "Not set")}</b><br>'
        f'{datetime.now():%A, %d %B %Y}</div></div>', unsafe_allow_html=True)


def page_title(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="pagetitle">{title}</div><div class="pagesub">{subtitle}</div>', unsafe_allow_html=True)


def section(label: str) -> None:
    st.markdown(f'<div class="section">{label}</div>', unsafe_allow_html=True)


def patient_bar(meta: dict, patient: dict) -> None:
    items = [("Patient", meta["name"] or "Unnamed"), ("Patient ID", meta["pid"]), ("Visit date", meta["visit"]),
             ("Age", f"{patient['Age']} yrs"), ("Gestation", f"{patient['GestationalAge']} weeks"),
             ("Assessed by", meta["clinician"] or "-")]
    st.markdown('<div class="patientbar">' + "".join(
        f"<div><span>{k}</span><b>{html.escape(str(v))}</b></div>" for k, v in items) + "</div>", unsafe_allow_html=True)


def risk_card(result: dict) -> None:
    color = RISK_COLORS[result["risk"]]
    st.markdown(
        f'<div class="risk-card" style="--c:{color}"><div class="tag">Predicted risk level</div>'
        f'<div class="big">{RISK_ICONS[result["risk"]]} {result["risk"].upper()} RISK</div>'
        f'<div class="sub">Model confidence: <b>{result["prob"][result["pred"]]:.0%}</b></div></div>',
        unsafe_allow_html=True)


def probability_chart(prob: np.ndarray) -> go.Figure:
    fig = go.Figure(go.Bar(x=prob, y=[f"{l} risk" for l in LABELS], orientation="h",
                           marker_color=[RISK_COLORS[l] for l in LABELS],
                           text=[f"{p:.0%}" for p in prob], textposition="outside", cliponaxis=False))
    fig.update_layout(height=200, margin=dict(l=0, r=30, t=10, b=10), xaxis=dict(range=[0, 1.12], tickformat=".0%"),
                      yaxis=dict(autorange="reversed"), showlegend=False)
    return fig


def contribution_chart(result: dict, top_n: int = 8) -> go.Figure:
    rows = sorted(result["by_push"], key=lambda d: -abs(d["contribution"]))[:top_n][::-1]
    color = RISK_COLORS[result["risk"]]
    fig = go.Figure(go.Bar(
        x=[d["contribution"] for d in rows], orientation="h",
        y=[f"{FRIENDLY[d['feature']]} = {show_value(d)}" for d in rows],
        marker_color=[color if d["contribution"] > 0 else "#3B6FB6" for d in rows],
        hovertemplate="%{y}<br>contribution: %{x:.3f}<extra></extra>"))
    fig.update_layout(height=60 + 38 * len(rows), margin=dict(l=0, r=10, t=10, b=10),
                      xaxis_title=f"← pushes away from {result['risk']} risk  |  pushes toward {result['risk']} risk →")
    return fig


def driver_card(d: dict, risk: str) -> None:
    """One abnormal measurement that helped cause the result, with the reason it matters."""
    pill = "above" if d["status"] == "above normal" else "below"
    with st.container(border=True):
        left, right = st.columns([4, 1.4])
        left.markdown(f"**{FRIENDLY[d['feature']]}**  ·  {show_value(d)}  "
                      f"<span class='pill {pill}'>{d['status']}</span>", unsafe_allow_html=True)
        if d["note"]:
            left.markdown(f"*{d['note'].capitalize()}*  ·  normal: {d['normal']}")
        else:
            left.markdown(f"normal: {d['normal']}")
        left.caption(f"Why it matters: {WHY_IT_MATTERS[d['feature']]}")
        right.metric("Share of push", f"{d['share']:.0f}%")
        right.progress(min(int(d["share"]), 100))


def inject_example(name: str) -> None:
    """Callback: copy one demo patient into the form widgets."""
    for feature, value in EXAMPLES[name].items():
        spec = FEATURE_SPECS[feature]
        st.session_state[f"in_{feature}"] = ("Yes" if value else "No") if spec["kind"] == "binary" else value
    st.session_state.pop("result", None)


def reset_form() -> None:
    """Callback: put every widget back to its default value."""
    for feature, spec in FEATURE_SPECS.items():
        st.session_state[f"in_{feature}"] = spec["default"]
    st.session_state["in_name"] = ""
    st.session_state.pop("result", None)


def render_input(feature: str) -> None:
    """Draw the right widget for one feature."""
    spec, key = FEATURE_SPECS[feature], f"in_{feature}"
    unit = INPUT_UNITS.get(feature)
    label = f"{FRIENDLY[feature]} ({unit})" if unit else FRIENDLY[feature]
    normal = FIXED_RANGES.get(feature, (0, 0, ""))[2]
    help_text = spec["help"] + (f"  Normal: {normal}." if normal else "")
    if spec["kind"] == "binary":
        st.radio(label, ["No", "Yes"], key=key, horizontal=True, help=help_text)
    elif spec["kind"] == "int":
        st.number_input(label, min_value=spec["min"], max_value=spec["max"], step=1, key=key, help=help_text)
    else:
        st.number_input(label, min_value=spec["min"], max_value=spec["max"], step=spec["step"], format="%.1f",
                        key=key, help=help_text)


def collect_patient() -> dict:
    """Read the form widgets into the dict the model expects (Yes/No -> 1/0)."""
    patient = {}
    for feature, spec in FEATURE_SPECS.items():
        value = st.session_state[f"in_{feature}"]
        patient[feature] = (1 if value == "Yes" else 0) if spec["kind"] == "binary" else value
    return patient


def next_patient_id() -> str:
    return f"MR-{len(st.session_state.get('records', [])) + 1:04d}"


# ---- report ------------------------------------------------------------------
def build_report_text(meta: dict, patient: dict, result: dict, alert: bool, plan: dict, threshold: float, xai: str) -> str:
    """Plain-text clinical summary (shown on screen and downloadable)."""
    L = [f"{APP_NAME.upper()}  -  MATERNAL RISK ASSESSMENT REPORT", "=" * 58,
         f"Patient : {meta['name'] or 'Unnamed'}   ID: {meta['pid']}   Visit: {meta['visit']}",
         f"Clinician: {meta['clinician'] or '-'}   Age: {patient['Age']}   Gestation: {patient['GestationalAge']} weeks", "",
         f"PREDICTED RISK     : {result['risk'].upper()} RISK (confidence {result['prob'][result['pred']]:.0%})",
         "Probabilities      : " + ", ".join(f"{l} {p:.1%}" for l, p in zip(LABELS, result["prob"])),
         f"EARLY-WARNING ALERT: {'YES' if alert else 'No'}   (rule: High risk, or Mid risk with High-risk prob >= {threshold:.0%})",
         f"CARE LEVEL         : {plan['title']} - {plan['summary']}", ""]
    if plan["red_flags"]:
        L += ["CRITICAL FINDINGS"] + [f"  ! {t}" for t in plan["red_flags"]] + [""]
    L.append("MEASUREMENTS")
    L += [f"  - {FRIENDLY[d['feature']]}: {show_value(d)} [{d['status']}]  (normal: {d['normal']})" for d in result["all"]]
    if result["drivers"]:
        L += ["", f"FACTORS THAT LED TO THE {result['risk'].upper()}-RISK RESULT (strongest first, explained with {xai})"]
        L += [f"  - {explain_driver(d, result['risk'])}" for d in result["drivers"]]
    for cat in ("Immediate", "Follow-up", "Counselling"):
        if plan["buckets"][cat]:
            L += ["", f"RECOMMENDED - {cat.upper()}"] + [f"  - [{lab}] {txt}" for lab, txt in plan["buckets"][cat]]
    L += ["", "DANGER SIGNS TO TELL THE PATIENT: " + "; ".join(plan["danger_signs"]),
          "", "Decision-support output only. It does not replace clinical judgement; follow local protocols."]
    return "\n".join(L)


def build_report_html(text: str, color: str) -> str:
    """Printable HTML version of the report (open it and use Print -> Save as PDF)."""
    body = html.escape(text)
    return (f"<!doctype html><html><head><meta charset='utf-8'><title>{APP_NAME} report</title>"
            f"<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:820px;margin:30px auto;padding:0 18px}}"
            f"h1{{color:{NAVY};border-bottom:4px solid {color};padding-bottom:6px}}"
            f"pre{{white-space:pre-wrap;font-family:inherit;line-height:1.5}}</style></head><body>"
            f"<h1>{APP_NAME}</h1><pre>{body}</pre></body></html>")


# =============================================================================
# 9. PAGES
# =============================================================================
def page_assess(art: dict, explainer: LocalExplainer, threshold: float, clinician: str) -> None:
    page_title("New patient assessment", "Register the patient, enter her measurements, and run the risk assessment.")

    st.markdown("**Quick demo (for presentations):**")
    cols = st.columns(len(EXAMPLES) + 1)
    for col, name in zip(cols, EXAMPLES):
        col.button(name, on_click=inject_example, args=(name,), use_container_width=True)
    cols[-1].button("↺ Clear form", on_click=reset_form, use_container_width=True)

    with st.form("patient_form"):
        section("Step 1 · Patient details")
        c1, c2, c3, c4 = st.columns([1, 2, 1.2, 1.2])
        pid = c1.text_input("Patient ID", value=next_patient_id())
        name = c2.text_input("Full name", key="in_name", placeholder="e.g. Amina Yusuf")
        visit = c3.date_input("Visit date", value=date.today())
        c4.text_input("Clinician", value=clinician or "Set in sidebar", disabled=True)

        section("Step 2 · Clinical measurements")
        columns = st.columns(len(FORM_GROUPS))
        for col, (title, feats) in zip(columns, FORM_GROUPS):
            with col.container(border=True):
                st.markdown(f"##### {title}")
                for feature in feats:
                    render_input(feature)
        submitted = st.form_submit_button("🔍  Run risk assessment", type="primary", use_container_width=True)

    if submitted:
        patient = collect_patient()
        meta = {"pid": pid.strip() or next_patient_id(), "name": name.strip(), "visit": f"{visit:%d %b %Y}",
                "clinician": clinician}
        with st.spinner("Analysing…"):
            data = assess_patient(patient, art, explainer)
        alert = alert_flag(data, threshold)
        plan = build_care_plan(data, alert, patient)
        st.session_state["result"] = {"meta": meta, "patient": patient, "data": data}
        st.session_state.setdefault("records", []).append({
            "Time": datetime.now().strftime("%d %b %Y %H:%M"), "Patient ID": meta["pid"],
            "Name": meta["name"] or "Unnamed", "Age": patient["Age"], "Gestation (wks)": patient["GestationalAge"],
            "Risk": data["risk"], "High-risk prob.": round(float(data["prob"][2]), 3),
            "Alert": "YES" if alert else "No", "Care level": plan["title"],
            "Main factors": ", ".join(FRIENDLY[d["feature"]] for d in data["drivers"][:3]) or "-"})

    if "result" not in st.session_state:
        st.info("Fill in the patient details and measurements (or load a demo patient) and press **Run risk assessment**.", icon="👆")
        return
    show_result(art, st.session_state["result"], threshold)


def show_result(art: dict, stored: dict, threshold: float) -> None:
    """Everything shown after a patient has been assessed."""
    meta, patient, result = stored["meta"], stored["patient"], stored["data"]
    alert = alert_flag(result, threshold)
    plan = build_care_plan(result, alert, patient)
    xai = art.get("xai_method", "SHAP")

    st.divider()
    section("Assessment result")
    patient_bar(meta, patient)

    left, right = st.columns([1.1, 1])
    with left:
        risk_card(result)
        if alert:
            st.error("**🚨 EARLY-WARNING ALERT** — escalate for prompt clinical review.")
        else:
            st.success("**✅ No early-warning alert** for this patient.")
        st.caption(f"Alert rule: High risk, or Mid risk with a High-risk probability of at least {threshold:.0%}.")
    with right:
        st.markdown("**Probability of each risk level**")
        st.plotly_chart(probability_chart(result["prob"]), use_container_width=True, config={"displayModeBar": False})

    for note in result["notes"]:
        st.warning(note, icon="⚠️")

    tab_factors, tab_why, tab_plan, tab_report = st.tabs(
        ["🔎 Contributing factors", "💡 Why this result", "🩹 Recommended care plan", "📄 Report"])

    with tab_factors:
        icon = {"normal": "✅ Normal", "above normal": "⬆️ Above normal", "below normal": "⬇️ Below normal"}
        shares = {d["feature"]: d["share"] for d in result["all"]}
        table = pd.DataFrame([{"Measurement": FRIENDLY[d["feature"]], "Value": show_value(d),
                               "Status": icon[d["status"]], "Normal range": d["normal"],
                               "Influence on result (%)": round(shares[d["feature"]], 1),
                               "Clinical note": d["note"].capitalize()} for d in result["by_push"]])
        st.dataframe(table, hide_index=True, use_container_width=True,
                     column_config={"Influence on result (%)": st.column_config.ProgressColumn(
                         "Influence on result (%)", min_value=0, max_value=100, format="%.0f%%")})
        st.caption(f"Rows are ordered by how strongly each measurement pushed the prediction toward "
                   f"{result['risk']} risk (explained with {xai}).")
        st.plotly_chart(contribution_chart(result), use_container_width=True, config={"displayModeBar": False})

    with tab_why:
        if result["drivers"]:
            st.markdown(f"##### Abnormal measurements that led to the **{result['risk']}-risk** result")
            st.caption("Strongest first. 'Share of push' is how much of the push toward this result came from each value.")
            for d in result["drivers"]:
                driver_card(d, result["risk"])
        else:
            st.info(f"No abnormal measurement is pushing this patient toward {result['risk']} risk.", icon="ℹ️")
        if result["supporting"]:
            st.markdown("**Normal values that also supported this result**")
            for d in result["supporting"]:
                st.markdown(f"- {FRIENDLY[d['feature']]} = {show_value(d)} (normal) · {d['share']:.0f}% of the push")
        if result["others"]:
            with st.expander(f"Other abnormal values with little influence ({len(result['others'])})"):
                for d in result["others"]:
                    note = f" - {d['note']}" if d["note"] else ""
                    st.markdown(f"- **{FRIENDLY[d['feature']]}** = {show_value(d)}: {d['status']}{note}  \n"
                                f"  <small>Normal: {d['normal']}</small>", unsafe_allow_html=True)

    with tab_plan:
        st.markdown(f'<div class="plan-banner" style="--c:{plan["color"]}"><b>{plan["title"]}</b><br>{plan["summary"]}</div>',
                    unsafe_allow_html=True)
        for text in plan["red_flags"]:
            st.error(text, icon="🚨")
        cats = [("Immediate", "⚡ Immediate actions at this visit"), ("Follow-up", "📅 Follow-up, tests and monitoring"),
                ("Counselling", "💬 Counselling for the patient")]
        shown = False
        for cat, title in cats:
            if plan["buckets"][cat]:
                shown = True
                st.markdown(f"##### {title}")
                for label, text in plan["buckets"][cat]:
                    st.markdown(f'<div class="act"><div class="lab">{html.escape(label)}</div>{html.escape(text)}</div>',
                                unsafe_allow_html=True)
        if not shown:
            st.info("No abnormal measurements need specific action. Continue routine antenatal care.", icon="ℹ️")
        with st.container(border=True):
            st.markdown("##### 🚩 Danger signs to teach the patient (return immediately if any occur)")
            st.markdown("\n".join(f"- {s}" for s in plan["danger_signs"]))
        st.caption("These are general guideline-style prompts to support the clinician. Apply local protocols and clinical judgement.")

    with tab_report:
        text = build_report_text(meta, patient, result, alert, plan, threshold, xai)
        st.code(text, language=None)
        a, b = st.columns(2)
        a.download_button("⬇️ Download printable report (.html)", build_report_html(text, plan["color"]),
                          file_name=f"{meta['pid']}_risk_report.html", mime="text/html", use_container_width=True)
        b.download_button("⬇️ Download as text (.txt)", text, file_name=f"{meta['pid']}_risk_report.txt",
                          use_container_width=True)
        st.caption("Tip: open the .html report in your browser and choose Print → Save as PDF.")

    st.caption("⚕️ Decision-support tool only. It does not replace a clinician's judgement.")


def page_records() -> None:
    page_title("Patient records", "Assessments done in this session. Download them before closing the app.")
    records = st.session_state.get("records", [])
    if not records:
        st.info("No patients assessed yet. Run an assessment and it will appear here.", icon="🗂️")
        return
    df = pd.DataFrame(records)
    a, b, c, d = st.columns(4)
    a.metric("Patients assessed", len(df))
    b.metric("High risk", int((df["Risk"] == "High").sum()))
    c.metric("Mid risk", int((df["Risk"] == "Mid").sum()))
    d.metric("Alerts raised", int((df["Alert"] == "YES").sum()))
    st.dataframe(df.iloc[::-1], hide_index=True, use_container_width=True)
    x, y = st.columns(2)
    x.download_button("⬇️ Download records (.csv)", df.to_csv(index=False), file_name="patient_records.csv",
                      mime="text/csv", use_container_width=True)
    if y.button("🗑️ Clear all records", use_container_width=True):
        st.session_state["records"] = []
        st.session_state.pop("result", None)
        st.rerun()


def page_insights(art: dict) -> None:
    page_title("Model insights", f"How the selected model, <b>{art['model_name']}</b>, performed on the held-out test set.")
    metrics = art["test_metrics"]

    top = ["Accuracy", "Sensitivity", "Specificity", "ROC-AUC"]
    for col, name in zip(st.columns(4), top):
        col.metric(name, f"{metrics[name]:.1%}" if name != "ROC-AUC" else f"{metrics[name]:.3f}")
    for col, name in zip(st.columns(4), ["Precision", "Recall", "F1", "High-risk Sensitivity"]):
        col.metric(name, f"{metrics[name]:.1%}")

    left, right = st.columns([1.3, 1])
    with left:
        shown = ["Accuracy", "Precision", "Recall", "F1", "Sensitivity", "Specificity", "ROC-AUC", "High-risk Sensitivity"]
        fig = px.bar(x=shown, y=[metrics[m] for m in shown], text=[f"{metrics[m]:.2f}" for m in shown],
                     color_discrete_sequence=[TEAL])
        fig.update_layout(height=360, yaxis_range=[0, 1.05], xaxis_title=None, yaxis_title="Score",
                          margin=dict(t=30, b=10), title="Test-set performance")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        with st.container(border=True):
            st.markdown("##### Model card")
            st.markdown(f"**Algorithm:** {art['model_name']}  \n"
                        f"**Explanation method:** {art.get('xai_method', 'SHAP')}  \n"
                        f"**SMOTE used in training:** {'Yes' if art.get('smote') else 'No'}  \n"
                        f"**Early-warning threshold:** {art['alert_threshold']:.0%}  \n"
                        f"**Input features:** {len(art['features'])}")
            st.markdown("**Tuned hyperparameters**")
            st.json(art.get("hyperparameters", {}), expanded=False)

    clf = art["model"].steps[-1][1]
    importance = getattr(clf, "feature_importances_", None)
    if importance is None and hasattr(clf, "coef_"):
        importance = np.abs(clf.coef_).mean(axis=0)
    if importance is not None:
        imp = pd.Series(importance, index=[FRIENDLY[f] for f in art["features"]]).sort_values()
        fig = px.bar(imp, orientation="h", color_discrete_sequence=[NAVY], title="Which measurements matter most to the model")
        fig.update_layout(height=460, showlegend=False, xaxis_title="Relative importance", yaxis_title=None, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if art.get("xai_comparison"):
        with st.expander("How the explanation technique was chosen (notebook Section 12)"):
            st.dataframe(pd.DataFrame(art["xai_comparison"]).round(3), use_container_width=True)


def page_data(df: pd.DataFrame, art: dict) -> None:
    page_title("Data explorer", "A look at the cleaned dataset the model was trained on.")
    names = {0: "Low", 1: "Mid", 2: "High"}
    view = df.assign(Risk=df["RiskLevel"].map(names))

    a, b, c = st.columns(3)
    a.metric("Patients (after cleaning)", f"{len(df):,}")
    b.metric("Features", len(art["features"]))
    c.metric("Missing cells (imputed by the model)", f"{int(df[art['features']].isna().sum().sum()):,}")

    left, right = st.columns([1, 1.6])
    counts = view["Risk"].value_counts().reindex(LABELS)
    with left:
        fig = px.pie(values=counts.values, names=[f"{l} risk" for l in LABELS], hole=0.55,
                     color_discrete_sequence=[RISK_COLORS[l] for l in LABELS], title="Risk-level balance")
        fig.update_layout(height=360, margin=dict(t=50, b=10))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        feature = st.selectbox("Explore a feature", art["features"], format_func=lambda f: FRIENDLY[f])
        fig = px.histogram(view.dropna(subset=[feature]), x=feature, color="Risk", barmode="overlay", opacity=0.65,
                           nbins=30, category_orders={"Risk": LABELS}, color_discrete_map=RISK_COLORS,
                           labels={feature: f"{FRIENDLY[feature]} ({art['units'][feature]})"})
        fig.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with st.expander("Feature correlation heatmap"):
        corr = df[art["features"]].corr().round(2)
        corr.index = corr.columns = [FRIENDLY[f] for f in corr.columns]
        fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig.update_layout(height=620, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with st.expander("Preview the data"):
        st.dataframe(view.drop(columns="RiskLevel").head(100), use_container_width=True)


def page_about(art: dict) -> None:
    page_title("About this system", "An intelligent maternal health risk prediction and early-warning tool built with machine learning.")
    steps = [("1 · Data", "3,500 patient records with 14 clinical features, cleaned and labelled Low / Mid / High risk."),
             ("2 · Models", "Logistic Regression, KNN, Random Forest and Gradient Boosting were tuned and compared."),
             ("3 · Selection", "The best model was chosen using accuracy, F1, sensitivity, specificity, ROC-AUC and High-risk sensitivity."),
             ("4 · Explanation", f"{art.get('xai_method', 'SHAP')} shows which measurements drove each prediction."),
             ("5 · Early warning", "An alert is raised for High risk, or Mid risk with a high probability of High risk."),
             ("6 · Care plan", "Each flagged factor is linked to guideline-style actions the clinician can follow.")]
    for row in (steps[:3], steps[3:]):
        for col, (title, text) in zip(st.columns(3), row):
            col.markdown(f'<div class="step"><b>{title}</b><br>{text}</div>', unsafe_allow_html=True)
        st.write("")

    with st.container(border=True):
        st.markdown("##### How to read a result")
        st.markdown("- **Risk level** is the class with the highest probability.\n"
                    "- **Early-warning alert** is a safety net: it can fire for Mid-risk patients whose chance of being High risk is high.\n"
                    "- **Contributing factors / Why this result** list the abnormal measurements that pushed the prediction, using clinical reference ranges "
                    "(ACOG, ADA/IADPSG, CDC, WHO, IOM).\n"
                    "- **Recommended care plan** gives general guideline-style actions for those factors.")
    st.warning("This tool supports early detection. It does **not** replace a clinician's judgement, a clinical examination or local protocols.", icon="⚕️")


# =============================================================================
# 10. MAIN
# =============================================================================
def missing_file_screen(path: Path, how_to_fix: str) -> None:
    topbar("")
    page_title("Almost there", "A required file could not be found.")
    st.error(f"Could not find **{path.name}** in `{path.parent}`.")
    st.markdown(how_to_fix)
    st.stop()


def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon="🩺", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    if not MODEL_FILE.exists():
        missing_file_screen(MODEL_FILE, "Run the notebook from top to bottom (the last code cell, **Section 14**, saves "
                                        "`finalized_maternal_model.sav`), then put that file in the same folder as this script.")
    if not DATA_FILE.exists():
        missing_file_screen(DATA_FILE, "Copy `Maternal_Health_Risk_RAW_UNCLEANED.csv` into the same folder as this script.")

    art = load_artifact(str(MODEL_FILE))
    explainer = build_explainer(str(MODEL_FILE), str(DATA_FILE))
    df = load_clean_data(str(DATA_FILE))

    for feature, spec in FEATURE_SPECS.items():  # first run: fill the form with defaults
        st.session_state.setdefault(f"in_{feature}", spec["default"])
    st.session_state.setdefault("in_name", "")
    st.session_state.setdefault("records", [])

    with st.sidebar:
        st.markdown(f"## 🩺 {APP_NAME}")
        st.caption(APP_TAGLINE)
        clinician = st.text_input("Clinician name", key="clinician", placeholder="e.g. Dr. A. Bello")
        page = st.radio("Navigate", ["🩺 New Assessment", "🗂️ Patient Records", "📊 Model Insights",
                                     "📈 Data Explorer", "ℹ️ About"], label_visibility="collapsed")
        st.divider()
        with st.container(border=True):
            st.markdown(f"**Model:** {art['model_name']}  \n**Test accuracy:** {art['test_metrics']['Accuracy']:.1%}  \n"
                        f"**Explained with:** {art.get('xai_method', 'SHAP')}")
        threshold = st.slider("Alert sensitivity", 0.10, 0.90, float(art["alert_threshold"]), 0.05, format="%.2f",
                              help="A Mid-risk patient triggers an alert when her High-risk probability reaches this value. "
                                   "Lower = more cautious (more alerts).")
        st.caption("Decision support only.")

    topbar(clinician)
    if page.endswith("New Assessment"):
        page_assess(art, explainer, threshold, clinician)
    elif page.endswith("Patient Records"):
        page_records()
    elif page.endswith("Model Insights"):
        page_insights(art)
    elif page.endswith("Data Explorer"):
        page_data(df, art)
    else:
        page_about(art)

    st.markdown(f'<div class="foot">{APP_NAME} · Intelligent Maternal Health Risk Prediction · Final Year Project</div>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
