import streamlit as st
import pandas as pd
import pickle
import json
import plotly.graph_objects as go
from datetime import date

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aetheris — Hotel Analytics",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# LOAD CSS
# ─────────────────────────────────────────────────────────────────
with open("style.css", "r", encoding="utf-8") as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

# Extra styling for expander
st.markdown("""
<style>
div[data-testid="stExpander"] {
    background: rgba(8,13,28,0.85) !important;
    border: 1px solid rgba(212,175,55,0.18) !important;
    border-radius: 16px !important;
    overflow: hidden !important;
}
div[data-testid="stExpander"] summary {
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #d4af37 !important;
    padding: 1rem 1.4rem !important;
}
div[data-testid="stExpander"] summary:hover {
    background: rgba(212,175,55,0.05) !important;
}
div[data-testid="stExpander"] > div > div {
    padding: 0.2rem 1.4rem 1.2rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# LOAD MODEL & MAPPINGS
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    with open("hotel_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def load_mappings():
    with open("category_mappings.json", "r") as f:
        return json.load(f)

model = load_model()
MAPS  = load_mappings()

def enc(col: str, val, default: int = 0) -> int:
    lst = MAPS.get(col, [])
    return lst.index(val) if val in lst else default

# Meal plan friendly labels
MEAL_LABELS = {
    "BB":        "BB — Bed & Breakfast",
    "HB":        "HB — Half Board",
    "FB":        "FB — Full Board",
    "SC":        "SC — Self Catering",
    "Undefined": "Undefined",
}

# Full country name map (ISO-3 code -> display name)
COUNTRY_NAMES = {
    "ABW": "Aruba",          "AGO": "Angola",         "AIA": "Anguilla",
    "ALB": "Albania",        "AND": "Andorra",         "ARE": "UAE",
    "ARG": "Argentina",      "ARM": "Armenia",         "ASM": "American Samoa",
    "ATA": "Antarctica",     "ATF": "Fr. S. Territories","AUS": "Australia",
    "AUT": "Austria",        "AZE": "Azerbaijan",      "BDI": "Burundi",
    "BEL": "Belgium",        "BEN": "Benin",           "BFA": "Burkina Faso",
    "BGD": "Bangladesh",     "BGR": "Bulgaria",        "BHR": "Bahrain",
    "BHS": "Bahamas",        "BIH": "Bosnia & Herz.",  "BLR": "Belarus",
    "BOL": "Bolivia",        "BRA": "Brazil",          "BRB": "Barbados",
    "BWA": "Botswana",       "CAF": "C. African Rep.", "CHE": "Switzerland",
    "CHL": "Chile",          "CHN": "China",           "CIV": "Ivory Coast",
    "CMR": "Cameroon",       "CN":  "China (CN)",      "COL": "Colombia",
    "COM": "Comoros",        "CPV": "Cape Verde",      "CRI": "Costa Rica",
    "CUB": "Cuba",           "CYM": "Cayman Islands",  "CYP": "Cyprus",
    "CZE": "Czech Republic", "DEU": "Germany",         "DJI": "Djibouti",
    "DMA": "Dominica",       "DNK": "Denmark",         "DOM": "Dominican Rep.",
    "DZA": "Algeria",        "ECU": "Ecuador",         "EGY": "Egypt",
    "ESP": "Spain",          "EST": "Estonia",         "ETH": "Ethiopia",
    "FIN": "Finland",        "FJI": "Fiji",            "FRA": "France",
    "FRO": "Faroe Islands",  "GAB": "Gabon",           "GBR": "United Kingdom",
    "GEO": "Georgia",        "GGY": "Guernsey",        "GHA": "Ghana",
    "GIB": "Gibraltar",      "GLP": "Guadeloupe",      "GNB": "Guinea-Bissau",
    "GRC": "Greece",         "GTM": "Guatemala",       "GUY": "Guyana",
    "HKG": "Hong Kong",      "HND": "Honduras",        "HRV": "Croatia",
    "HUN": "Hungary",        "IDN": "Indonesia",       "IMN": "Isle of Man",
    "IND": "India",          "IRL": "Ireland",         "IRN": "Iran",
    "IRQ": "Iraq",           "ISL": "Iceland",         "ISR": "Israel",
    "ITA": "Italy",          "JAM": "Jamaica",         "JEY": "Jersey",
    "JOR": "Jordan",         "JPN": "Japan",           "KAZ": "Kazakhstan",
    "KEN": "Kenya",          "KHM": "Cambodia",        "KIR": "Kiribati",
    "KNA": "St. Kitts & Nevis","KOR": "South Korea",  "KWT": "Kuwait",
    "LAO": "Laos",           "LBN": "Lebanon",         "LBY": "Libya",
    "LCA": "Saint Lucia",    "LIE": "Liechtenstein",   "LKA": "Sri Lanka",
    "LTU": "Lithuania",      "LUX": "Luxembourg",      "LVA": "Latvia",
    "MAC": "Macau",          "MAR": "Morocco",         "MCO": "Monaco",
    "MDG": "Madagascar",     "MDV": "Maldives",        "MEX": "Mexico",
    "MKD": "N. Macedonia",   "MLI": "Mali",            "MLT": "Malta",
    "MMR": "Myanmar",        "MNE": "Montenegro",      "MOZ": "Mozambique",
    "MRT": "Mauritania",     "MUS": "Mauritius",       "MWI": "Malawi",
    "MYS": "Malaysia",       "MYT": "Mayotte",         "NAM": "Namibia",
    "NCL": "New Caledonia",  "NGA": "Nigeria",         "NIC": "Nicaragua",
    "NLD": "Netherlands",    "NOR": "Norway",          "NPL": "Nepal",
    "NZL": "New Zealand",    "OMN": "Oman",            "PAK": "Pakistan",
    "PAN": "Panama",         "PER": "Peru",            "PHL": "Philippines",
    "PLW": "Palau",          "POL": "Poland",          "PRI": "Puerto Rico",
    "PRT": "Portugal",       "PRY": "Paraguay",        "PYF": "Fr. Polynesia",
    "QAT": "Qatar",          "ROU": "Romania",         "RUS": "Russia",
    "RWA": "Rwanda",         "SAU": "Saudi Arabia",    "SDN": "Sudan",
    "SEN": "Senegal",        "SGP": "Singapore",       "SLE": "Sierra Leone",
    "SLV": "El Salvador",    "SMR": "San Marino",      "SRB": "Serbia",
    "STP": "Sao Tome & Pr.", "SUR": "Suriname",        "SVK": "Slovakia",
    "SVN": "Slovenia",       "SWE": "Sweden",          "SYC": "Seychelles",
    "SYR": "Syria",          "TGO": "Togo",            "THA": "Thailand",
    "TJK": "Tajikistan",     "TMP": "East Timor",      "TUN": "Tunisia",
    "TUR": "Turkey",         "TWN": "Taiwan",          "TZA": "Tanzania",
    "UGA": "Uganda",         "UKR": "Ukraine",         "UMI": "US Outlying Is.",
    "URY": "Uruguay",        "USA": "United States",   "UZB": "Uzbekistan",
    "VEN": "Venezuela",      "VGB": "British Virgin Is.","VNM": "Vietnam",
    "ZAF": "South Africa",   "ZMB": "Zambia",          "ZWE": "Zimbabwe",
}

def country_label(code: str) -> str:
    name = COUNTRY_NAMES.get(code, code)
    return f"{name} ({code})"

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-diamond">💎</div>
        <div class="sb-logo">AETHERIS</div>
        <p class="sb-sub">Luxury Hotel Analytics</p>
        <p class="sb-tagline">Enterprise Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-status">
        <div class="sb-status-dot">
            <div class="dot-live"></div>
            <span class="dot-label">Systems Online</span>
        </div>
        <div class="sb-stat">
            <span class="sb-stat-key">Status</span>
            <span class="sb-stat-val gold">Operational</span>
        </div>
        <div class="sb-stat">
            <span class="sb-stat-key">Accuracy</span>
            <span class="sb-stat-val">85.0%</span>
        </div>
        <div class="sb-stat">
            <span class="sb-stat-key">Features</span>
            <span class="sb-stat-val">24 Active</span>
        </div>
        <div class="sb-stat">
            <span class="sb-stat-key">Build</span>
            <span class="sb-stat-val">2026</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-decor">
        <div class="sb-decor-title">Aetheris Analytics Suite</div>
        <div class="sb-gem-row">💎 🏨 ✦ 🏅 ✦</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-glow"></div>
    <div class="hero-eyebrow">✦ &nbsp;Predictive Revenue Intelligence &nbsp;✦</div>
    <div class="hero-brand">AETHERIS</div>
    <div class="hero-title">Hotel Booking Cancellation Prediction System</div>
    <p class="hero-desc">
        Enterprise-grade machine learning platform — predict booking cancellation risk
        in real time, protect revenue, and optimise occupancy across every property.
    </p>
</div>
<div class="gold-divider"></div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SECTION A — COMMON BOOKING DETAILS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="form-card">
    <div class="form-header">
        <div class="form-header-icon">📋</div>
        <div>
            <div class="form-header-title">Booking Details</div>
            <div class="form-header-sub">Core booking information</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Row 1 — Hotel / Dates
c1, c2, c3 = st.columns(3)
with c1:
    hotel_type = st.selectbox("Hotel Type",
        options=MAPS.get("hotel", ["City Hotel", "Resort Hotel"]))
with c2:
    arrival_date = st.date_input("Estimated Arrival Date", value=date(2016, 6, 15))
with c3:
    checkout_date = st.date_input("Check-out Date", value=date(2016, 6, 19))

# Row 2 — Room / Guests
c4, c5, c6 = st.columns(3)
with c4:
    room_type = st.selectbox("Room Type",
        options=MAPS.get("reserved_room_type", ["A","B","C","D","E","F","G","H","L","P"]))
with c5:
    adults   = st.number_input("Adult Guests",  min_value=1, max_value=10, value=2, step=1)
with c6:
    children = st.number_input("Children",      min_value=0, max_value=10, value=0, step=1)

# Row 3 — Stay duration / parking / deposit
c7, c8, c9 = st.columns(3)
with c7:
    weekend_nights = st.number_input("Weekend Nights", min_value=0, max_value=14, value=1, step=1)
with c8:
    week_nights    = st.number_input("Week Nights",    min_value=0, max_value=30, value=3, step=1)
with c9:
    deposit_type = st.selectbox("Deposit Type",
        options=MAPS.get("deposit_type", ["No Deposit","Non Refund","Refundable"]))

# Row 4 — Parking / Days until arrival
c10, c11, _ = st.columns(3)
with c10:
    parking = st.selectbox("Required Parking Spaces", options=[0,1,2,3], index=0)
with c11:
    days_until = st.slider("Days Until Arrival", min_value=0, max_value=700, value=45,
        help="Higher = higher cancellation risk")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SECTION B — ADVANCED FEATURES (all remaining model features)
# ─────────────────────────────────────────────────────────────────
with st.expander("⚙️  Advanced Features — All 24 Model Inputs (optional tuning)"):
    st.markdown("""
    <div style="padding:0.2rem 0 0.8rem;font-size:0.78rem;color:#475569;letter-spacing:0.04em;">
        These fields are pre-set to smart defaults. Adjust them for a more precise prediction.
    </div>
    """, unsafe_allow_html=True)

    # --- Guest Origin & Meal ---
    st.markdown('<p style="font-size:0.72rem;font-weight:700;letter-spacing:0.15em;color:#d4af37;text-transform:uppercase;margin-bottom:0.4rem;">Guest Profile</p>', unsafe_allow_html=True)
    adv1, adv2, adv3 = st.columns(3)
    with adv1:
        country_list   = MAPS.get("country", ["PRT"])
        country_labels = [country_label(c) for c in country_list]
        default_idx    = country_list.index("PRT") if "PRT" in country_list else 0
        selected_label = st.selectbox("Guest Country", options=country_labels,
            index=default_idx)
        country_code   = country_list[country_labels.index(selected_label)]
    with adv2:
        meal_keys   = list(MEAL_LABELS.keys())
        meal_labels = [MEAL_LABELS[k] for k in meal_keys]
        meal_label  = st.selectbox("Meal Plan",
            options=meal_labels, index=0)
        meal_code   = meal_keys[meal_labels.index(meal_label)]
    with adv3:
        customer_type = st.selectbox("Customer Type",
            options=MAPS.get("customer_type",
                ["Contract","Group","Transient","Transient-Party"]),
            index=2)   # default: Transient

    # --- Booking Channel ---
    st.markdown('<p style="font-size:0.72rem;font-weight:700;letter-spacing:0.15em;color:#d4af37;text-transform:uppercase;margin:1rem 0 0.4rem;">Booking Channel</p>', unsafe_allow_html=True)
    adv4, adv5, adv6 = st.columns(3)
    with adv4:
        market_segment = st.selectbox("Market Segment",
            options=MAPS.get("market_segment",
                ["Aviation","Complementary","Corporate","Direct","Groups","Offline TA/TO","Online TA","Undefined"]),
            index=6)   # default: Online TA
    with adv5:
        dist_channel = st.selectbox("Distribution Channel",
            options=MAPS.get("distribution_channel",
                ["Corporate","Direct","GDS","TA/TO","Undefined"]),
            index=3)   # default: TA/TO
    with adv6:
        agent_id = st.number_input("Agent ID", min_value=0, max_value=999, value=9, step=1)

    # --- Assigned Room ---
    st.markdown('<p style="font-size:0.72rem;font-weight:700;letter-spacing:0.15em;color:#d4af37;text-transform:uppercase;margin:1rem 0 0.4rem;">Room Assignment</p>', unsafe_allow_html=True)
    adv7, adv8, adv9 = st.columns(3)
    with adv7:
        assigned_room = st.selectbox("Assigned Room Type",
            options=MAPS.get("assigned_room_type",
                ["A","B","C","D","E","F","G","H","I","K","L","P"]),
            index=0)
    with adv8:
        adr = st.number_input("Avg. Daily Rate (ADR)", min_value=0.0, max_value=5000.0,
            value=120.0, step=10.0,
            help="Average price per night in euros")
    with adv9:
        special_requests = st.number_input("Special Requests", min_value=0, max_value=5, value=1, step=1)

    # --- History & Changes ---
    st.markdown('<p style="font-size:0.72rem;font-weight:700;letter-spacing:0.15em;color:#d4af37;text-transform:uppercase;margin:1rem 0 0.4rem;">Booking History</p>', unsafe_allow_html=True)
    adv10, adv11, _ = st.columns(3)
    with adv10:
        prev_cancellations = st.number_input("Previous Cancellations",
            min_value=0, max_value=30, value=0, step=1)
    with adv11:
        booking_changes = st.number_input("Booking Changes",
            min_value=0, max_value=20, value=0, step=1)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────────────────────────
btn_col1, btn_col2, btn_col3 = st.columns([1.5, 1, 1.5])
with btn_col2:
    predict_clicked = st.button("🔮  PREDICT RISK", use_container_width=True)

# ─────────────────────────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────────────────────────
if predict_clicked:
    month_str = arrival_date.strftime("%B")

    feature_row = {
        "hotel":                     enc("hotel", hotel_type),
        "lead_time":                 days_until,
        "arrival_date_year":         arrival_date.year,
        "arrival_date_month":        enc("arrival_date_month", month_str, default=5),
        "arrival_date_week_number":  arrival_date.isocalendar()[1],
        "arrival_date_day_of_month": arrival_date.day,
        "stays_in_weekend_nights":   weekend_nights,
        "stays_in_week_nights":      week_nights,
        "adults":                    adults,
        "children":                  children,
        "meal":                      enc("meal", meal_code),
        "country":                   enc("country", country_code, default=135),
        "market_segment":            enc("market_segment", market_segment),
        "distribution_channel":      enc("distribution_channel", dist_channel),
        "previous_cancellations":    prev_cancellations,
        "reserved_room_type":        enc("reserved_room_type", room_type),
        "assigned_room_type":        enc("assigned_room_type", assigned_room),
        "booking_changes":           booking_changes,
        "deposit_type":              enc("deposit_type", deposit_type),
        "agent":                     agent_id,
        "customer_type":             enc("customer_type", customer_type),
        "adr":                       adr,
        "required_car_parking_spaces": parking,
        "total_of_special_requests": special_requests,
    }

    features_df = pd.DataFrame([feature_row])
    pred   = model.predict(features_df)[0]
    proba  = model.predict_proba(features_df)[0]
    risk   = round(proba[1] * 100, 1)
    conf   = round(proba[pred]  * 100, 1)

    # Classify risk
    if risk < 35:
        level       = "STABLE"
        lvl_cls     = "green"
        gauge_color = "#10b981"
        status_cls  = "confirmed"
        icon        = "✅"
        verdict     = "Booking Likely Confirmed"
        vrd_cls     = "confirmed"
    elif risk < 65:
        level       = "ELEVATED"
        lvl_cls     = "yellow"
        gauge_color = "#f59e0b"
        status_cls  = "medium-risk"
        icon        = "⚠️"
        verdict     = "Moderate Cancellation Risk"
        vrd_cls     = "medium"
    else:
        level       = "CRITICAL"
        lvl_cls     = "red"
        gauge_color = "#f43f5e"
        status_cls  = "at-risk"
        icon        = "🚨"
        verdict     = "High Cancellation Risk"
        vrd_cls     = "at-risk"

    # ── RESULTS ──────────────────────────────────────────────
    st.markdown('<div class="result-wrap">', unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([1, 2, 1])

    with col_l:
        st.markdown(f"""
        <div class="status-panel {status_cls}">
            <div class="sp-icon">{icon}</div>
            <div class="sp-verdict {vrd_cls}">{verdict}</div>
            <p class="sp-sub">
                All 24 model features analysed<br>
                via Random Forest engine
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_c:
        fig = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = risk,
            number= {"suffix": "%",
                     "font": {"size": 44, "color": "#ffffff", "family": "Outfit"}},
            title = {"text": "CANCELLATION RISK PROBABILITY",
                     "font": {"size": 10, "color": "#475569", "family": "Outfit"}},
            gauge = {
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": "#1e293b",
                    "ticksuffix": "%",
                    "nticks": 6,
                    "tickfont": {"color": "#475569", "size": 10},
                },
                "bar":       {"color": gauge_color, "thickness": 0.26},
                "bgcolor":   "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,  35], "color": "rgba(16,185,129,0.07)"},
                    {"range": [35, 65], "color": "rgba(245,158,11,0.07)"},
                    {"range": [65,100], "color": "rgba(244,63,94,0.07)"},
                ],
                "threshold": {
                    "line":      {"color": "#d4af37", "width": 2},
                    "thickness": 0.8,
                    "value":     risk,
                },
            },
        ))
        fig.update_layout(
            paper_bgcolor = "rgba(0,0,0,0)",
            plot_bgcolor  = "rgba(0,0,0,0)",
            font          = {"color": "#ffffff", "family": "Outfit"},
            margin        = dict(l=30, r=30, t=55, b=20),
            height        = 290,
        )
        st.markdown('<div class="gauge-panel">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown(f"""
        <div class="metric-pair">
            <div class="mcard">
                <div class="mcard-lbl">Model Confidence</div>
                <div class="mcard-val">{conf}%</div>
            </div>
            <div class="mcard">
                <div class="mcard-lbl">Risk Level</div>
                <div class="mcard-val {lvl_cls}">{level}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:2.5rem 1rem;">
        <div style="font-size:2.4rem;margin-bottom:0.8rem;opacity:0.2;">💎</div>
        <div style="color:#334155;font-size:0.9rem;font-weight:500;">
            Fill in the booking details above and click<br>
            <span style="color:rgba(212,175,55,0.55);font-weight:700;letter-spacing:0.06em;">
                🔮 PREDICT RISK
            </span>
            to run the full analysis
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="aetheris-footer">
    <div class="footer-gems">✦ 💎 ✦</div>
    <div class="footer-logo">AETHERIS</div>
    <div class="footer-copy">&copy; 2026 Aetheris Analytics &mdash; All Rights Reserved</div>
</div>
""", unsafe_allow_html=True)
