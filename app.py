import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

data_path = BASE_DIR / "hospital_appointment.csv"

df_app = pd.read_csv(
    data_path,
    dtype={
        "PatientId": "string",
        "AppointmentID": "string"
    }
)

df_app = df_app.rename(columns={
    "PatientId": "patient_id",
    "AppointmentID": "appointment_id"
})

df_app["ScheduledDay"] = pd.to_datetime(df_app["ScheduledDay"])
df_app["AppointmentDay"] = pd.to_datetime(df_app["AppointmentDay"])

df_app["wait_days"] = (df_app["AppointmentDay"] - df_app["ScheduledDay"].dt.normalize()).dt.days

df_app = df_app[df_app["wait_days"] >= 0].copy()

model = joblib.load(BASE_DIR / "no_show_model.pkl")
preprocessor = joblib.load(BASE_DIR / "no_show_preprocessor.pkl")

st.set_page_config(
    page_title="Hospital Appointment No-Show Prediction",
    page_icon="🏥",
    layout="centered"
)
st.title("🏥 Hospital Appointment No-Show Prediction")
st.write("Enter patient appointment details to estimate no-show risk.")
st.subheader("Patient Lookup")

patient_id_input = st.text_input(
    "Enter Patient ID"
)
patient_matches = pd.DataFrame()

previous_appointments = 0
previous_no_shows = 0
previous_no_show_rate = 0.0

if patient_id_input:
    patient_matches = df_app[
        df_app["patient_id"].astype(str) == patient_id_input.strip()
    ]

    if patient_matches.empty:
        st.warning("Patient ID not found in the dataset.")
    else:
        st.success(
            f"{len(patient_matches)} appointment(s) found for this Patient ID."
        )

        # Sort appointments chronologically
        patient_matches = patient_matches.sort_values(
            ["AppointmentDay", "appointment_id"]
        ).reset_index(drop=True)

        appointment_options = patient_matches["appointment_id"].astype(str).tolist()

        selected_appointment = st.selectbox(
            "Select Appointment ID",
            appointment_options
        )

        selected_row = patient_matches[
            patient_matches["appointment_id"].astype(str) == selected_appointment
        ].iloc[0]

        # Calculate previous appointment history

        patient_id = patient_id_input.strip()
        selected_date = selected_row["AppointmentDay"]
        selected_id = str(selected_row["appointment_id"])
        previous_records = df_app[
            (df_app["patient_id"].astype(str) == patient_id) &
            (
                (df_app["AppointmentDay"] < selected_date) |
                (
                    (df_app["AppointmentDay"] == selected_date) &
                    (df_app["appointment_id"] < selected_id)
                )
            )
        ]

        previous_appointments = len(previous_records)

        previous_no_shows = (
            previous_records["No-show"]
            .eq("Yes")
            .sum()
        )

        if previous_appointments > 0:
            previous_no_show_rate = (
                previous_no_shows / previous_appointments
            )
        else:
            previous_no_show_rate = 0.0
st.subheader("Patient Details")

if patient_id_input and not patient_matches.empty:

    gender = selected_row["Gender"]
    age = int(selected_row["Age"]) if selected_row["Age"] >= 0 else 37
    neighbourhood = selected_row["Neighbourhood"]
    scholarship = int(selected_row["Scholarship"])
    hypertension = int(selected_row["Hipertension"])
    diabetes = int(selected_row["Diabetes"])
    alcoholism = int(selected_row["Alcoholism"])
    handicap = int(selected_row["Handcap"])
    sms_received = int(selected_row["SMS_received"])

    st.write("**Gender:**", gender)
    st.write("**Age:**", age)
    st.write("**Neighbourhood:**", neighbourhood)
    st.write("**Scholarship:**", scholarship)
    st.write("**Hypertension:**", hypertension)
    st.write("**Diabetes:**", diabetes)
    st.write("**Alcoholism:**", alcoholism)
    st.write("**Handicap:**", handicap)
    st.write("**SMS Received:**", sms_received)

else:

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["F", "M"])
        age = st.number_input("Age", min_value=0, max_value=115, value=44)

        neighbourhood_options = sorted(
            df_app["Neighbourhood"].dropna().unique()
        )

        neighbourhood = st.selectbox(
            "Neighbourhood",
            neighbourhood_options
        )

        scholarship = st.selectbox("Scholarship", [0, 1])

    with col2:
        hypertension = st.selectbox("Hypertension", [0, 1])
        diabetes = st.selectbox("Diabetes", [0, 1])
        alcoholism = st.selectbox("Alcoholism", [0, 1])
        handicap = st.selectbox("Handicap", [0, 1, 2, 3, 4])
        sms_received = st.selectbox("SMS Received", [0, 1])

st.subheader("Appointment Details")

if patient_id_input and not patient_matches.empty:

    scheduled_day = selected_row["ScheduledDay"]
    appointment_day = selected_row["AppointmentDay"]

    wait_days = (
    appointment_day.normalize() - scheduled_day.normalize()
).days

    appointment_weekday = appointment_day.strftime("%A")
    appointment_month = appointment_day.month
    appointment_day_num = appointment_day.day
    scheduled_hour = scheduled_day.hour

    st.write("**Waiting Days:**", wait_days)
    st.write("**Appointment Weekday:**", appointment_weekday)
    st.write("**Appointment Month:**", appointment_month)
    st.write("**Appointment Day:**", appointment_day_num)
    st.write("**Scheduled Hour:**", scheduled_hour)

else:

    col1, col2 = st.columns(2)

    with col1:
        wait_days = st.number_input(
            "Waiting Days",
            min_value=0,
            max_value=365,
            value=3
        )

        appointment_weekday = st.selectbox(
            "Appointment Weekday",
            ["Monday", "Tuesday", "Wednesday", "Thursday",
             "Friday", "Saturday", "Sunday"]
        )

        appointment_month = st.number_input(
            "Appointment Month",
            min_value=1,
            max_value=12,
            value=6
        )

    with col2:
        appointment_day_num = st.number_input(
            "Appointment Day",
            min_value=1,
            max_value=31,
            value=3
        )

        scheduled_hour = st.number_input(
            "Scheduled Hour",
            min_value=0,
            max_value=23,
            value=10
        )

st.subheader("Patient History")
st.caption("Automatically calculated from the patient's previous appointments.")

if patient_id_input and not patient_matches.empty:

    st.write("**Previous Appointments:**", previous_appointments)
    st.write("**Previous No-shows:**", previous_no_shows)
    st.write(
        "**Previous No-show Rate:**",
        f"{previous_no_show_rate * 100:.1f}%"
    )

else:

    st.write("Enter a Patient ID to automatically calculate patient history.")
# Automatically calculate derived features

if age < 18:
    age_group = "Child"
elif age < 35:
    age_group = "Young Adult"
elif age < 60:
    age_group = "Adult"
else:
    age_group = "Senior"


if wait_days == 0:
    wait_time_group = "Same Day"
elif wait_days <= 3:
    wait_time_group = "1-3 Days"
elif wait_days <= 7:
    wait_time_group = "4-7 Days"
elif wait_days <= 14:
    wait_time_group = "8-14 Days"
elif wait_days <= 30:
    wait_time_group = "15-30 Days"
else:
    wait_time_group = "More than 30 Days"


appointment_weekend = int(
    appointment_weekday in ["Saturday", "Sunday"]
)
if st.button("Predict No-Show Risk"):

    patient_data = {
        "gender": gender,
        "age": age,
        "neighbourhood": neighbourhood,
        "scholarship": scholarship,
        "hypertension": hypertension,
        "diabetes": diabetes,
        "alcoholism": alcoholism,
        "handicap": handicap,
        "sms_received": sms_received,
        "wait_days": wait_days,
        "appointment_weekday": appointment_weekday,
        "appointment_month": appointment_month,
        "appointment_day_num": appointment_day_num,
        "scheduled_hour": scheduled_hour,
        "age_group": age_group,
        "appointment_weekend": appointment_weekend,
        "wait_time_group": wait_time_group,
        "previous_appointments": previous_appointments,
        "previous_no_shows": previous_no_shows,
        "previous_no_show_rate": previous_no_show_rate
    }

    patient_df = pd.DataFrame([patient_data])

    processed_data = preprocessor.transform(patient_df)

    probability = model.predict_proba(processed_data)[0, 1]

    if probability < 0.30:
        risk = "Low Risk"
    elif probability < 0.50:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    st.subheader("Prediction Result")

    st.metric(
        "No-Show Risk Score",
        f"{probability * 100:.1f}%"
    )

    if risk == "High Risk":
        st.error("🔴 High Risk")
    elif risk == "Medium Risk":
        st.warning("🟡 Medium Risk")
    else:
        st.success("🟢 Low Risk")

    if risk == "High Risk":
        st.write(
            "**Recommended Action:** Contact the patient and confirm the appointment."
        )
    elif risk == "Medium Risk":
        st.write(
            "**Recommended Action:** Consider sending a reminder to the patient."
        )
    else:
        st.write(
            "**Recommended Action:** No additional action required."
        )
    st.info(
        f"The model estimates a {probability * 100:.1f}% no-show risk score for this appointment."
    )
    st.subheader("Key Factors Influencing Prediction")
        
    st.caption(
        "These factors show how the trained model influenced this prediction. "
        "They indicate model associations, not direct causes of a no-show."
    )

    contributions = processed_data[0] * model.coef_[0]

    explanation = pd.DataFrame({
        "feature": preprocessor.get_feature_names_out(),
        "contribution": contributions
    })

    positive_factors = (
        explanation[explanation["contribution"] > 0]
        .sort_values("contribution", ascending=False)
        .head(5)
    )

    negative_factors = (
        explanation[explanation["contribution"] < 0]
        .sort_values("contribution")
        .head(3)
    )

    feature_labels = {
        "appointment_day_num": "Appointment day",
        "previous_appointments": "Previous appointments",
        "previous_no_shows": "Previous no-shows",
        "previous_no_show_rate": "Previous no-show rate",
        "sms_received": "SMS received",
        "wait_time_group_1-3 Days": "Waiting period: 1–3 days",
        "wait_time_group_4-7 Days": "Waiting period: 4–7 days",
        "wait_time_group_8-14 Days": "Waiting period: 8–14 days",
        "wait_time_group_15-30 Days": "Waiting period: 15–30 days",
        "wait_time_group_More than 30 Days": "Waiting period: more than 30 days",
        "wait_time_group_Same Day": "Same-day appointment",
        "age": "Age",
        "age_group_Adult": "Adult age group",
        "age_group_Child": "Child age group",
        "age_group_Young Adult": "Young adult age group",
        "age_group_Senior": "Senior age group",
        "appointment_weekday_Monday": "Monday appointment",
        "appointment_weekday_Tuesday": "Tuesday appointment",
        "appointment_weekday_Wednesday": "Wednesday appointment",
        "appointment_weekday_Thursday": "Thursday appointment",
        "appointment_weekday_Friday": "Friday appointment",
        "appointment_weekday_Saturday": "Saturday appointment",
        "appointment_weekday_Sunday": "Sunday appointment"
    }

    if not positive_factors.empty:
        st.write("**Factors increasing no-show risk:**")

        for _, row in positive_factors.iterrows():
            clean_feature = row["feature"].replace("num__", "").replace("cat__", "")
            clean_feature = feature_labels.get(clean_feature, clean_feature)

            st.write(
                f"• {clean_feature} ↑  "
                f"(contribution: +{row['contribution']:.3f})"
            )

    if not negative_factors.empty:
        st.write("**Factors reducing no-show risk:**")

        for _, row in negative_factors.iterrows():
            clean_feature = row["feature"].replace("num__", "").replace("cat__", "")
            clean_feature = feature_labels.get(clean_feature, clean_feature)

            st.write(
                f"• {clean_feature} ↓  "
                f"(contribution: {row['contribution']:.3f})"
            )