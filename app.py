
import streamlit as st
import joblib
import pandas as pd
from datetime import time


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="wide"
)

# --------------------------------------------------
# LOAD DATASET AND MODEL
# --------------------------------------------------

df = pd.read_csv("flight_data_sample.csv")
classifier = joblib.load("flight_delay_model.pkl")

# --------------------------------------------------
# AIRPORT DATABASE
# --------------------------------------------------

try:
    import airportsdata
    airport_database = airportsdata.load("IATA")
except ImportError:
    airport_database = {}

# Fallback airport names
airport_names = {
    "ABE": "Lehigh Valley International Airport",
    "ATL": "Hartsfield-Jackson Atlanta International Airport",
    "AUS": "Austin-Bergstrom International Airport",
    "BGR": "Bangor International Airport",
    "BHM": "Birmingham-Shuttlesworth International Airport",
    "BIL": "Billings Logan International Airport",
    "BIS": "Bismarck Municipal Airport",
    "BJI": "Bemidji Regional Airport",
    "BLI": "Bellingham International Airport",
    "BLV": "MidAmerica St. Louis Airport",
    "BNA": "Nashville International Airport",
    "BOS": "Boston Logan International Airport",
    "BWI": "Baltimore/Washington International Airport",
    "CLT": "Charlotte Douglas International Airport",
    "DAL": "Dallas Love Field",
    "DCA": "Ronald Reagan Washington National Airport",
    "DEN": "Denver International Airport",
    "DFW": "Dallas/Fort Worth International Airport",
    "DTW": "Detroit Metropolitan Airport",
    "EWR": "Newark Liberty International Airport",
    "FLL": "Fort Lauderdale-Hollywood International Airport",
    "HNL": "Daniel K. Inouye International Airport",
    "HOU": "William P. Hobby Airport",
    "IAD": "Washington Dulles International Airport",
    "IAH": "George Bush Intercontinental Airport",
    "JFK": "John F. Kennedy International Airport",
    "LAS": "Harry Reid International Airport",
    "LAX": "Los Angeles International Airport",
    "LGA": "LaGuardia Airport",
    "MCO": "Orlando International Airport",
    "MDW": "Chicago Midway International Airport",
    "MIA": "Miami International Airport",
    "MSP": "Minneapolis-Saint Paul International Airport",
    "MSY": "Louis Armstrong New Orleans International Airport",
    "OAK": "Oakland International Airport",
    "ORD": "Chicago O'Hare International Airport",
    "PDX": "Portland International Airport",
    "PHL": "Philadelphia International Airport",
    "PHX": "Phoenix Sky Harbor International Airport",
    "RDU": "Raleigh-Durham International Airport",
    "SAN": "San Diego International Airport",
    "SEA": "Seattle-Tacoma International Airport",
    "SFO": "San Francisco International Airport",
    "SJC": "San Jose International Airport",
    "SLC": "Salt Lake City International Airport",
    "SMF": "Sacramento International Airport",
    "TPA": "Tampa International Airport"
}

# --------------------------------------------------
# AIRLINE NAMES
# --------------------------------------------------

carrier_names = {
    "9E": "Endeavor Air",
    "AA": "American Airlines",
    "AS": "Alaska Airlines",
    "B6": "JetBlue Airways",
    "DL": "Delta Air Lines",
    "F9": "Frontier Airlines",
    "G4": "Allegiant Air",
    "HA": "Hawaiian Airlines",
    "MQ": "Envoy Air",
    "NK": "Spirit Airlines",
    "OH": "PSA Airlines",
    "OO": "SkyWest Airlines",
    "UA": "United Airlines",
    "WN": "Southwest Airlines",
    "YX": "Republic Airways"
}

# --------------------------------------------------
# DISPLAY FORMATTING
# --------------------------------------------------

def format_airport(code):

    airport = airport_database.get(code)

    if airport:
        return airport["name"]

    return airport_names.get(code, code)


def format_carrier(code):

    name = carrier_names.get(code, "Other Airline")

    return f"{name} ({code})"


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_delay(month, day_of_month, day_of_week,
                  carrier, origin, dest,
                  crs_dep_time, crs_arr_time,
                  crs_elapsed_time, distance):

    input_data = pd.DataFrame({
        "month": [month],
        "day_of_month": [day_of_month],
        "day_of_week": [day_of_week],
        "op_unique_carrier": [carrier],
        "origin": [origin],
        "dest": [dest],
        "crs_dep_time": [crs_dep_time],
        "crs_arr_time": [crs_arr_time],
        "crs_elapsed_time": [crs_elapsed_time],
        "distance": [distance]
    })

    prediction = classifier.predict(input_data)

    label = "Delayed" if prediction[0] == 1 else "On Time"

    return label


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

def main():

    # CUSTOM DESIGN

    st.markdown("""
    <style>

    .stApp {
        background: linear-gradient(135deg, #071426, #102B49);
        color: #FFFFFF;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    h1, h2, h3, p, label {
        color: #FFFFFF !important;
    }

    .main-title {
        background: linear-gradient(90deg, #123B65, #087EAA);
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 25px;
        border: 1px solid #28577E;
    }

    .main-title h1 {
        color: white !important;
        font-size: 38px;
        margin-bottom: 8px;
    }

    .main-title p {
        color: #D5E9F7 !important;
        font-size: 16px;
    }

    .section-title {
        color: #63C5F1 !important;
        font-size: 22px;
        font-weight: bold;
    }

    .stSelectbox div[data-baseweb="select"] > div,
    .stDateInput input,
    .stTimeInput input {
        background-color: #193653;
        color: white;
        border-radius: 8px;
    }

    [data-baseweb="select"] span {
        color: white !important;
    }

    .stButton button {
        background: linear-gradient(90deg, #087EAA, #1769AA);
        color: white;
        border: none;
        border-radius: 10px;
        height: 50px;
        font-size: 17px;
        font-weight: bold;
        width: 100%;
    }

    .stButton button:hover {
        background: #2196C7;
        color: white;
    }

    [data-testid="stMetric"] {
        background-color: #142D48;
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #294967;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: white !important;
    }

    hr {
        border-color: #294967;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    </style>
    """, unsafe_allow_html=True)

    # TITLE

    st.markdown("""
    <div class="main-title">
        <h1>✈️ Flight Delay Prediction System</h1>
        <p>Your Flight. Your Schedule. Better Planning.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write(
        "Enter your flight details to check the likelihood "
        "of an arrival delay."
    )

    st.divider()

    # --------------------------------------------------
    # FLIGHT INFORMATION
    # --------------------------------------------------

    st.markdown(
        '<p class="section-title">🛫 Flight Information</p>',
        unsafe_allow_html=True
    )

    # AIRLINES - ALPHABETICAL ORDER BY FULL NAME

    carriers = sorted(
        df["op_unique_carrier"].dropna().unique(),
        key=format_carrier
    )

    # AIRPORTS - ALPHABETICAL ORDER BY AIRPORT NAME

    airports = sorted(
        set(df["origin"].dropna().unique())
        | set(df["dest"].dropna().unique()),
        key=format_airport
    )

    # OPERATING AIRLINE

    carrier = st.selectbox(
        "Operating Airline",
        carriers,
        format_func=format_carrier
    )

    # DEPARTURE AND DESTINATION AIRPORTS

    col1, col2 = st.columns(2)

    with col1:

        origin = st.selectbox(
            "Departure Airport",
            airports,
            format_func=format_airport
        )

    with col2:

        dest = st.selectbox(
            "Destination Airport",
            airports,
            format_func=format_airport
        )

    # DEPARTURE DATE

    departure_date = st.date_input(
        "Scheduled Departure Date"
    )

    # DEPARTURE AND ARRIVAL TIMES SIDE BY SIDE

    time_col1, time_col2 = st.columns(2)

    with time_col1:

        departure_time = st.time_input(
            "Scheduled Departure Time",
            value=time(9, 0),
            step=300
        )

    with time_col2:

        arrival_time = st.time_input(
            "Scheduled Arrival Time",
            value=time(12, 0),
            step=300
        )

    # --------------------------------------------------
    # AUTOMATIC DISTANCE AND DURATION
    # --------------------------------------------------

    distance = (
        df[
            (df["origin"] == origin) &
            (df["dest"] == dest)
        ]["distance"].median()
    )

    duration = (
        df[
            (df["origin"] == origin) &
            (df["dest"] == dest) &
            (df["op_unique_carrier"] == carrier)
        ]["crs_elapsed_time"].median()
    )

    if pd.isna(duration):

        duration = (
            df[
                (df["origin"] == origin) &
                (df["dest"] == dest)
            ]["crs_elapsed_time"].median()
        )

    # --------------------------------------------------
    # FLIGHT SUMMARY CARDS
    # --------------------------------------------------

    st.divider()

    st.markdown(
        '<p class="section-title">📋 Flight Summary</p>',
        unsafe_allow_html=True
    )

    summary1, summary2, summary3 = st.columns(3)

    summary1.metric(
        "Operating Airline",
        carrier_names.get(carrier, carrier)
    )

    summary2.metric(
        "Flight Distance",
        f"{distance:,.0f} miles"
        if pd.notna(distance) else "Unavailable"
    )

    summary3.metric(
        "Flight Duration",
        f"{duration:.0f} minutes"
        if pd.notna(duration) else "Unavailable"
    )

    st.caption(
        "Distance and duration are estimated using historical "
        "flight records."
    )

    # --------------------------------------------------
    # FLIGHT DELAY PREDICTION
    # --------------------------------------------------

    st.divider()

    st.markdown(
        '<p class="section-title">🔍 Flight Delay Analysis</p>',
        unsafe_allow_html=True
    )

    st.write(
        "Check your flight's predicted arrival status below."
    )

    if st.button("✈️ Predict Flight Status"):

        if origin == dest:

            st.error(
                "Please select different departure and destination airports."
            )

        elif pd.isna(distance) or pd.isna(duration):

            st.error(
                "Historical information is unavailable for this route."
            )

        else:

            month = departure_date.month
            day_of_month = departure_date.day

            # Monday = 1, Sunday = 7

            day_of_week = departure_date.isoweekday()

            # CONVERT TIME TO HHMM FORMAT

            crs_dep_time = (
                departure_time.hour * 100
                + departure_time.minute
            )

            crs_arr_time = (
                arrival_time.hour * 100
                + arrival_time.minute
            )

            # CALL PREDICTION FUNCTION

            result = predict_delay(
                month,
                day_of_month,
                day_of_week,
                carrier,
                origin,
                dest,
                crs_dep_time,
                crs_arr_time,
                duration,
                distance
            )

            # DISPLAY RESULT

            if result == "Delayed":

                st.error(
                    "⚠️ Prediction: Flight Likely to Be Delayed"
                )

                st.write(
                    "The flight is predicted to arrive "
                    "at least 15 minutes late."
                )

            else:

                st.success(
                    "✅ Prediction: Flight Likely to Be On Time"
                )

                st.write(
                    "The flight is predicted not to arrive "
                    "15 minutes or more late."
                )

            st.caption(
                "Predictions are based on historical flight data. "
                "Actual arrival conditions may differ."
            )

    # --------------------------------------------------
    # FOOTER
    # --------------------------------------------------

    st.divider()

    st.markdown(
        """
        <div style="text-align:center; color:#AFC5DA;">
            Flight Delay Prediction System
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# RUN APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    main()
