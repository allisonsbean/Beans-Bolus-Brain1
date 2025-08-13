import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone
from pydexcom import Dexcom
import time
import pytz
import requests
import json

# Page configuration
st.set_page_config(
    page_title="Bean's Bolus Brain",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for mobile-friendly design
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .current-bg {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .bg-normal { background-color: #d4edda; color: #155724; }
    .bg-high { background-color: #f8d7da; color: #721c24; }
    .bg-low { background-color: #fff3cd; color: #856404; }
    .bg-stale { background-color: #e2e3e5; color: #6c757d; }
    .manual-entry { background-color: #fff3cd; color: #856404; }
    .signal-loss { background-color: #f8d7da; color: #721c24; border: 2px dashed #dc3545; }
    
    .info-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .iob-card {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .insulin-card {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4169e1;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .meal-card {
        background-color: #f0fff0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #32cd32;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .correction-card {
        background-color: #fff8dc;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffa500;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .alert-card {
        background-color: #ffe6e6;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin: 0.5rem 0;
    }
    
    .section-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #ecf0f1;
    }
    
    .compact-table {
        font-size: 0.9rem;
    }
    
    /* Hide Streamlit footer and menu */
    .css-1d391kg, .css-1rs6os, .css-17ziqus {display: none !important;}
    footer {visibility: hidden;}
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Get user's timezone (EST for you)
USER_TIMEZONE = pytz.timezone('America/New_York')

def get_user_time():
    """Get current time in user's timezone"""
    return datetime.now(USER_TIMEZONE)

def normalize_datetime(dt):
    """Convert any datetime to user's timezone"""
    if dt.tzinfo is None:
        # Naive datetime - assume it's in user's timezone
        return USER_TIMEZONE.localize(dt)
    else:
        # Convert to user's timezone
        return dt.astimezone(USER_TIMEZONE)

def get_bg_class(glucose_value, is_manual=False, is_stale=False, signal_loss=False):
    if signal_loss:
        return "signal-loss"
    elif is_manual:
        return "manual-entry"
    elif is_stale:
        return "bg-stale"
    elif glucose_value < 80:
        return "bg-low"
    elif glucose_value > 180:
        return "bg-high"
    else:
        return "bg-normal"

def identify_glucose_status(glucose_value, trend=None):
    """Identify if glucose is high, low, or normal with trend info"""
    if glucose_value < 70:
        status = "🔴 LOW"
        severity = "URGENT"
    elif glucose_value < 80:
        status = "🟡 Below Target"
        severity = "CAUTION"
    elif glucose_value > 250:
        status = "🔴 HIGH"
        severity = "URGENT"
    elif glucose_value > 180:
        status = "🟠 Above Target"
        severity = "ATTENTION"
    else:
        status = "🟢 In Range"
        severity = "GOOD"
    
    # Add trend information
    if trend:
        if "rising" in trend.lower():
            trend_icon = "📈"
        elif "falling" in trend.lower():
            trend_icon = "📉"
        else:
            trend_icon = "➡️"
        status += f" {trend_icon}"
    
    return status, severity

def calculate_iob(insulin_doses, iob_duration_hours=4):
    """Calculate Insulin on Board using linear decay model"""
    current_time = get_user_time()
    total_iob = 0
    
    for dose in insulin_doses:
        dose_time = normalize_datetime(dose['datetime'])
        hours_since = (current_time - dose_time).total_seconds() / 3600
        
        if hours_since < iob_duration_hours and hours_since >= 0:
            # Linear decay: 100% at time 0, 0% at iob_duration_hours
            remaining_percent = max(0, 1 - (hours_since / iob_duration_hours))
            dose_iob = dose['amount'] * remaining_percent
            total_iob += dose_iob
    
    return round(total_iob, 1)

def calculate_correction_suggestion(current_glucose, target_glucose=115, correction_factor=50, iob=0):
    """Calculate correction bolus suggestion"""
    if current_glucose <= target_glucose:
        return 0, "No correction needed"
    
    raw_correction = (current_glucose - target_glucose) / correction_factor
    iob_adjusted_correction = max(0, raw_correction - iob)
    
    if iob > 0:
        message = f"Suggested: {iob_adjusted_correction:.1f}u (Raw: {raw_correction:.1f}u - {iob:.1f}u IOB)"
    else:
        message = f"Suggested: {iob_adjusted_correction:.1f}u"
    
    return round(iob_adjusted_correction, 1), message

def calculate_bolus_suggestion(current_glucose, carbs, protein, target_glucose=115, carb_ratio=12, correction_factor=50, protein_factor=10, iob=0):
    """Calculate suggested bolus dose including protein and IOB"""
    # Carb bolus
    carb_bolus = carbs / carb_ratio
    
    # Protein bolus (10% of protein weight converted to carb equivalent)
    protein_carb_equivalent = protein * (protein_factor / 100)
    protein_bolus = protein_carb_equivalent / carb_ratio
    
    # Correction bolus (adjusted for IOB)
    raw_correction = max(0, (current_glucose - target_glucose) / correction_factor)
    correction_bolus = max(0, raw_correction - iob)
    
    total_bolus = carb_bolus + protein_bolus + correction_bolus
    
    return {
        'carb_bolus': carb_bolus,
        'protein_bolus': protein_bolus,
        'protein_carb_equivalent': protein_carb_equivalent,
        'correction_bolus': correction_bolus,
        'raw_correction': raw_correction,
        'iob_adjustment': iob,
        'total_bolus': total_bolus
    }

def calculate_estimated_a1c(readings_df):
    """Calculate estimated A1C from 90 days of glucose data"""
    if readings_df.empty:
        return None
    
    # Calculate average glucose
    avg_glucose = readings_df['glucose'].mean()
    
    # Convert to estimated A1C using standard formula
    estimated_a1c = (avg_glucose + 46.7) / 28.7
    
    return round(estimated_a1c, 1)

def write_to_apple_health(data_type, value, date=None):
    """Write data to Apple Health (placeholder - requires HealthKit integration)"""
    # This would require a proper HealthKit bridge
    # For now, we'll just log what would be written
    if date is None:
        date = get_user_time()
    
    st.write(f"📱 Would write to Apple Health: {data_type} = {value} at {date.strftime('%I:%M %p')}")
    return True

def create_glucose_chart(readings_df, manual_readings=None, meals=None, insulin_doses=None):
    fig = go.Figure()
    
    # Add Dexcom readings
    if not readings_df.empty:
        fig.add_trace(go.Scatter(
            x=readings_df['datetime'],
            y=readings_df['glucose'],
            mode='lines+markers',
            name='Glucose',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=6)
        ))
    
    # Add manual glucose readings if any
    if manual_readings and len(manual_readings) > 0:
        manual_df = pd.DataFrame(manual_readings)
        fig.add_trace(go.Scatter(
            x=manual_df['datetime'],
            y=manual_df['glucose'],
            mode='markers',
            name='Manual Glucose',
            marker=dict(color='orange', size=10, symbol='diamond')
        ))
    
    # Add meal markers (simplified)
    if meals and len(meals) > 0:
        for meal in meals:
            try:
                # Convert datetime to timestamp for Plotly
                if hasattr(meal['datetime'], 'timestamp'):
                    meal_x = meal['datetime']
                else:
                    meal_x = pd.to_datetime(meal['datetime'])
                
                fig.add_vline(
                    x=meal_x,
                    line_dash="dot",
                    line_color="green",
                    annotation_text=f"🍽️ {meal['carbs']}g",
                    annotation_position="top"
                )
            except:
                # Skip this meal marker if there's an error
                continue
    
    # Temporarily disable insulin markers to prevent crashes
    # TODO: Fix insulin chart markers later
    # if insulin_doses and len(insulin_doses) > 0:
    #     for dose in insulin_doses:
    #         if dose['type'] == 'bolus':
    #             # Chart markers temporarily disabled
    #             pass
    
    # Add target range
    fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Target Low")
    fig.add_hline(y=130, line_dash="dash", line_color="green", annotation_text="Target High")
    fig.add_hrect(y0=80, y1=130, fillcolor="green", opacity=0.1)
    
    # Add critical levels
    fig.add_hline(y=70, line_dash="solid", line_color="red", annotation_text="Low Alert")
    fig.add_hline(y=180, line_dash="solid", line_color="red", annotation_text="High Alert")
    
    fig.update_layout(
        title="Glucose Trends (Last 12 Hours)",
        xaxis_title="Time",
        yaxis_title="Glucose (mg/dL)",
        height=400,
        showlegend=True
    )
    return fig

@st.cache_data(ttl=60)
def get_dexcom_data(username, password):
    try:
        dexcom = Dexcom(username=username, password=password)
        current = dexcom.get_current_glucose_reading()
        readings = dexcom.get_glucose_readings(minutes=720, max_count=144)
        return current, readings, None
    except Exception as e:
        return None, None, str(e)

def is_reading_stale(reading_time, max_minutes=30):
    if not reading_time:
        return True
    current_time = get_user_time()
    reading_time_normalized = normalize_datetime(reading_time)
    time_diff = current_time - reading_time_normalized
    return time_diff.total_seconds() > (max_minutes * 60)

def main():
    st.markdown('<h1 class="main-header">🧠 Bean\'s Bolus Brain</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'manual_readings' not in st.session_state:
        st.session_state.manual_readings = []
    if 'meals' not in st.session_state:
        st.session_state.meals = []
    if 'insulin_doses' not in st.session_state:
        st.session_state.insulin_doses = []
    if 'typical_basal' not in st.session_state:
        st.session_state.typical_basal = 20  # Default basal dose
    
    # Sidebar for all inputs
    with st.sidebar:
        st.header("🩸 Dexcom Login")
        username = st.text_input("Username", value="allisonsbean@gmail.com")
        password = st.text_input("Password", type="password", value="Allison9")
        
        if st.button("Test Connection"):
            if username and password:
                current, readings, error = get_dexcom_data(username, password)
                if current:
                    st.success(f"Connected! Current: {current.value} mg/dL")
                elif error:
                    st.error(f"Connection failed: {error}")
                else:
                    st.warning("Connected but no current reading available")
        
        st.header("💉 Insulin Logging")
        
        # Bolus insulin
        st.subheader("Bolus Insulin")
        bolus_amount = st.number_input("Bolus Amount (units)", min_value=0.0, max_value=20.0, value=0.0, step=0.5, key="bolus_amount")
        bolus_time = st.time_input("Bolus Time", value=get_user_time().time(), key="bolus_time")
        
        if st.button("Log Bolus"):
            if bolus_amount > 0:
                # Round to nearest whole number as requested
                rounded_bolus = round(bolus_amount)
                bolus_entry = {
                    'datetime': normalize_datetime(datetime.combine(get_user_time().date(), bolus_time)),
                    'amount': rounded_bolus,
                    'type': 'bolus'
                }
                st.session_state.insulin_doses.append(bolus_entry)
                st.success(f"Logged: {rounded_bolus}u bolus at {bolus_time}")
                
                # Write to Apple Health
                write_to_apple_health("Insulin", rounded_bolus, bolus_entry['datetime'])
            else:
                st.error("Please enter bolus amount")
        
        # Basal insulin
        st.subheader("Basal Insulin")
        typical_basal = st.number_input("Typical Daily Basal (units)", min_value=0, max_value=100, value=st.session_state.typical_basal, key="basal_input")
        
        # Update session state when value changes
        if typical_basal != st.session_state.typical_basal:
            st.session_state.typical_basal = typical_basal
        
        if st.button("Log Today's Basal"):
            basal_entry = {
                'datetime': normalize_datetime(datetime.combine(get_user_time().date(), datetime.min.time())),
                'amount': typical_basal,
                'type': 'basal'
            }
            # Check if basal already logged today
            today_basal = [d for d in st.session_state.insulin_doses 
                          if d['type'] == 'basal' and d['datetime'].date() == get_user_time().date()]
            if not today_basal:
                st.session_state.insulin_doses.append(basal_entry)
                st.success(f"Logged: {typical_basal}u basal for today")
                write_to_apple_health("Insulin", typical_basal, basal_entry['datetime'])
            else:
                st.warning("Basal already logged for today")
        
        st.header("✋ Manual Glucose")
        st.write("Use when Dexcom has signal loss")
        manual_glucose = st.number_input("Glucose (mg/dL)", min_value=40, max_value=400, value=100, key="manual_glucose")
        manual_time = st.time_input("Time", value=get_user_time().time(), key="manual_time")
        
        if st.button("Add Manual Reading"):
            manual_entry = {
                'datetime': normalize_datetime(datetime.combine(get_user_time().date(), manual_time)),
                'glucose': manual_glucose,
                'source': 'manual'
            }
            st.session_state.manual_readings.append(manual_entry)
            st.success(f"Added: {manual_glucose} mg/dL at {manual_time}")
        
        st.header("🍽️ Meal Tracking")
        meal_name = st.text_input("Meal/Food", placeholder="e.g. Breakfast, Apple, Pizza")
        meal_carbs = st.number_input("Carbs (g)", min_value=0, max_value=200, value=0, key="meal_carbs")
        meal_protein = st.number_input("Protein (g)", min_value=0, max_value=100, value=0, key="meal_protein")
        meal_calories = st.number_input("Calories", min_value=0, max_value=2000, value=0, key="meal_calories")
        meal_time = st.time_input("Meal Time", value=get_user_time().time(), key="meal_time")
        
        if st.button("Log Meal"):
            if meal_name and (meal_carbs > 0 or meal_protein > 0):
                meal_entry = {
                    'datetime': normalize_datetime(datetime.combine(get_user_time().date(), meal_time)),
                    'name': meal_name,
                    'carbs': meal_carbs,
                    'protein': meal_protein,
                    'calories': meal_calories
                }
                st.session_state.meals.append(meal_entry)
                st.success(f"Logged: {meal_name} - {meal_carbs}g carbs, {meal_protein}g protein")
            else:
                st.error("Please enter meal name and carbs or protein")
        
        # Quick meal buttons (simplified - remove for now since you want personalized)
        st.write("**Quick Add:** (These will learn your most-used foods)")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 Coming Soon"):
                st.info("Feature coming: Smart suggestions based on your meal history!")
        
        with col2:
            if st.button("📱 Add Custom"):
                st.info("Use the meal form above to log custom foods")
        
        # Clear data options
        if st.session_state.manual_readings or st.session_state.meals or st.session_state.insulin_doses:
            st.header("🗑️ Clear Data")
            if st.button("Clear Manual Readings"):
                st.session_state.manual_readings = []
                st.success("Manual readings cleared")
            if st.button("Clear Meals"):
                st.session_state.meals = []
                st.success("Meals cleared")
            if st.button("Clear Insulin Doses"):
                st.session_state.insulin_doses = []
                st.success("Insulin doses cleared")
    
    # Main content
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    # Get Dexcom data
    current, readings, error = get_dexcom_data(username, password) if username and password else (None, None, None)
    
    # Determine what to display for glucose
    display_glucose = None
    display_time = None
    display_source = None
    signal_loss = False
    is_stale = False
    trend_description = None
    
    if current:
        display_glucose = current.value
        display_time = normalize_datetime(current.datetime)
        display_source = f"Dexcom • {current.trend_description} {current.trend_arrow}"
        trend_description = current.trend_description
        is_stale = is_reading_stale(current.datetime)
    elif readings and len(readings) > 0:
        latest = max(readings, key=lambda r: r.datetime)
        display_glucose = latest.value
        display_time = normalize_datetime(latest.datetime)
        display_source = f"Last Dexcom • {latest.trend_description}"
        trend_description = latest.trend_description
        is_stale = is_reading_stale(latest.datetime)
    elif st.session_state.manual_readings:
        latest_manual = max(st.session_state.manual_readings, key=lambda r: r['datetime'])
        display_glucose = latest_manual['glucose']
        display_time = latest_manual['datetime']
        display_source = "Manual Entry"
        signal_loss = True
    else:
        signal_loss = True
    
    # Calculate IOB
    current_iob = calculate_iob(st.session_state.insulin_doses)
    
    # Display current status
    if display_glucose:
        bg_class = get_bg_class(display_glucose, 
                               is_manual=(display_source == "Manual Entry"),
                               is_stale=is_stale,
                               signal_loss=signal_loss)
        
        # Get glucose status
        glucose_status, severity = identify_glucose_status(display_glucose, trend_description)
        
        time_str = display_time.strftime("%I:%M %p") if display_time else "Unknown time"
        
        status_icon = ""
        if signal_loss:
            status_icon = "📡❌ Signal Loss"
        elif is_stale:
            status_icon = "⏰ Stale Data"
        elif "Manual" in display_source:
            status_icon = "✋ Manual"
        else:
            status_icon = "📡✅ Live"
        
        st.markdown(f'''
            <div class="current-bg {bg_class}">
                {display_glucose} mg/dL
                <br><small>{glucose_status}</small>
                <br><small>{time_str}</small>
                <br><small>{status_icon}</small>
                <br><small>{display_source}</small>
            </div>
        ''', unsafe_allow_html=True)
        
        # Show correction suggestion if needed
        if display_glucose > 130 and not signal_loss:
            correction_amount, correction_message = calculate_correction_suggestion(
                display_glucose, iob=current_iob
            )
            if correction_amount > 0:
                st.markdown(f'''
                    <div class="correction-suggestion">
                        <strong>💉 Correction Suggestion</strong><br>
                        {correction_message}<br>
                        <small>⚠️ Always confirm with your healthcare provider's guidelines</small>
                    </div>
                ''', unsafe_allow_html=True)
        
        # Show IOB prominently (always show, even if zero)
        st.markdown(f'''
            <div class="insulin-summary">
                <strong>🔄 Current IOB (Insulin on Board): {current_iob} units</strong>
            </div>
        ''', unsafe_allow_html=True)
        
        # Show insulin summary if any doses logged
        if st.session_state.insulin_doses:
            today_bolus = sum(d['amount'] for d in st.session_state.insulin_doses 
                            if d['type'] == 'bolus' and d['datetime'].date() == get_user_time().date())
            today_basal = sum(d['amount'] for d in st.session_state.insulin_doses 
                            if d['type'] == 'basal' and d['datetime'].date() == get_user_time().date())
            
            st.markdown(f'''
                <div class="meal-summary">
                    <strong>💉 Today's Insulin: {today_bolus}u bolus + {today_basal}u basal = {today_bolus + today_basal}u total</strong>
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.info("💡 Log some insulin doses in the sidebar to see IOB calculations and get better bolus suggestions!")
        
        # Debug IOB calculation (you can remove this later)
        if st.checkbox("🔍 Show IOB Debug Info"):
            st.write(f"**Current time:** {get_user_time().strftime('%I:%M %p')}")
            st.write(f"**Total insulin doses logged:** {len(st.session_state.insulin_doses)}")
            
            if st.session_state.insulin_doses:
                st.write("**Recent doses (last 6 hours):**")
                current_time = get_user_time()
                for dose in st.session_state.insulin_doses:
                    dose_time = normalize_datetime(dose['datetime'])
                    hours_ago = (current_time - dose_time).total_seconds() / 3600
                    if hours_ago <= 6:  # Show doses from last 6 hours
                        remaining_percent = max(0, 1 - (hours_ago / 4)) if hours_ago < 4 else 0
                        contributing_iob = dose['amount'] * remaining_percent
                        st.write(f"- {dose['amount']}u {dose['type']} at {dose_time.strftime('%I:%M %p')} ({hours_ago:.1f}h ago) → {contributing_iob:.1f}u IOB")
            else:
                st.write("No insulin doses logged yet. Use the sidebar to log some insulin!")
        
        # Show recent meals summary
        if st.session_state.meals:
            today_meals = [m for m in st.session_state.meals 
                          if m['datetime'].date() == get_user_time().date()]
            if today_meals:
                total_carbs_today = sum(m['carbs'] for m in today_meals)
                total_protein_today = sum(m.get('protein', 0) for m in today_meals)
                total_calories_today = sum(m['calories'] for m in today_meals)
                
                st.markdown(f'''
                    <div class="meal-summary">
                        <strong>🍽️ Today's Meals: {len(today_meals)} meals, {total_carbs_today}g carbs, {total_protein_today}g protein, {total_calories_today} calories</strong>
                    </div>
                ''', unsafe_allow_html=True)
        
        # Bolus calculator (organized in expandable section)
        if st.session_state.meals and display_glucose:
            recent_meals = [m for m in st.session_state.meals 
                           if m['datetime'] > get_user_time() - timedelta(hours=2)]
            if recent_meals:
                recent_carbs = sum(m['carbs'] for m in recent_meals)
                recent_protein = sum(m.get('protein', 0) for m in recent_meals)
                
                with st.expander("💉 Bolus Calculator (Last 2 Hours)", expanded=True):
                    bolus_calc = calculate_bolus_suggestion(display_glucose, recent_carbs, recent_protein, iob=current_iob)
                    
                    # Round all suggestions to whole numbers
                    carb_bolus_rounded = round(bolus_calc['carb_bolus'])
                    protein_bolus_rounded = round(bolus_calc['protein_bolus'])
                    correction_bolus_rounded = round(bolus_calc['correction_bolus'])
                    total_bolus_rounded = round(bolus_calc['total_bolus'])
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Carb Bolus", f"{carb_bolus_rounded}u", f"{recent_carbs}g carbs")
                    with col2:
                        st.metric("Protein Bolus", f"{protein_bolus_rounded}u", f"{recent_protein}g protein")
                    with col3:
                        st.metric("Correction", f"{correction_bolus_rounded}u", "IOB adjusted")
                    with col4:
                        st.metric("**Total Suggested**", f"{total_bolus_rounded}u", "Review below")
                    
                    # Bolus confirmation section
                    if total_bolus_rounded > 0:
                        st.markdown("---")
                        st.subheader("💉 Confirm Bolus Dose")
                        
                        col_dose, col_time, col_button = st.columns([2, 2, 1])
                        with col_dose:
                            confirmed_dose = st.number_input(
                                "Adjust dose if needed:", 
                                min_value=0, 
                                max_value=20, 
                                value=total_bolus_rounded, 
                                step=1,
                                key="confirm_bolus_dose"
                            )
                        with col_time:
                            bolus_time_confirm = st.time_input("Bolus time:", value=get_user_time().time(), key="confirm_bolus_time")
                        with col_button:
                            st.write("")  # Add spacing
                            if st.button("✅ Log Bolus", type="primary"):
                                bolus_entry = {
                                    'datetime': normalize_datetime(datetime.combine(get_user_time().date(), bolus_time_confirm)),
                                    'amount': confirmed_dose,
                                    'type': 'bolus'
                                }
                                st.session_state.insulin_doses.append(bolus_entry)
                                st.success(f"✅ Logged {confirmed_dose}u bolus!")
                                write_to_apple_health("Insulin", confirmed_dose, bolus_entry['datetime'])
                                st.rerun()  # Refresh to update IOB
                    
                    # Calculation details
                    if bolus_calc['iob_adjustment'] > 0:
                        st.info(f"💡 Correction adjusted for IOB: {bolus_calc['raw_correction']:.1f}u - {bolus_calc['iob_adjustment']:.1f}u IOB = {bolus_calc['correction_bolus']:.1f}u")
                    if recent_protein > 0:
                        st.info(f"💡 Protein calculation: {recent_protein}g protein = {bolus_calc['protein_carb_equivalent']:.1f}g carb equivalent")
                    st.caption("⚠️ This is a suggestion only. Always verify with your healthcare provider's guidelines.")
        
        # Key metrics in organized cards
        st.markdown('<div class="section-header">📊 Key Metrics</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            if current and hasattr(current, 'trend_description'):
                st.metric("Current Trend", current.trend_description.title())
            else:
                st.metric("Current Trend", "Unknown")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            if readings:
                recent_readings = [r for r in readings if normalize_datetime(r.datetime) > get_user_time() - timedelta(hours=24)]
                in_range = len([r for r in recent_readings if 80 <= r.value <= 130])
                time_in_range = (in_range / len(recent_readings) * 100) if recent_readings else 0
                st.metric("Time in Range", f"{time_in_range:.1f}%", "Last 24 hours")
                
                # Calculate estimated A1C from available data
                readings_df_temp = pd.DataFrame([{'glucose': r.value} for r in readings])
                estimated_a1c = calculate_estimated_a1c(readings_df_temp)
                if estimated_a1c:
                    st.metric("Est. A1C", f"{estimated_a1c}%", f"{len(readings)} readings")
            else:
                st.metric("Time in Range", "No data", "Connect Dexcom")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            if st.session_state.meals:
                today_carbs = sum(m['carbs'] for m in st.session_state.meals if m['datetime'].date() == get_user_time().date())
                today_protein = sum(m.get('protein', 0) for m in st.session_state.meals if m['datetime'].date() == get_user_time().date())
                st.metric("Today's Carbs", f"{today_carbs}g", f"{today_protein}g protein")
            else:
                st.metric("Today's Carbs", "0g", "Log meals")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Create combined chart
        readings_df = pd.DataFrame()
        if readings:
            readings_data = []
            for reading in readings:
                readings_data.append({
                    'datetime': normalize_datetime(reading.datetime),
                    'glucose': reading.value,
                    'trend': reading.trend_description
                })
            readings_df = pd.DataFrame(readings_data)
            readings_df = readings_df.sort_values('datetime')
        
        # Filter meals and insulin to last 12 hours for chart
        recent_meals_for_chart = [m for m in st.session_state.meals 
                                 if m['datetime'] > get_user_time() - timedelta(hours=12)]
        recent_insulin_for_chart = [d for d in st.session_state.insulin_doses 
                                   if d['datetime'] > get_user_time() - timedelta(hours=12)]
        
        if not readings_df.empty or st.session_state.manual_readings or recent_meals_for_chart:
            fig = create_glucose_chart(readings_df, st.session_state.manual_readings, 
                                     recent_meals_for_chart, recent_insulin_for_chart)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent readings table
        if not readings_df.empty:
            st.subheader("Recent Glucose Readings")
            recent_df = readings_df.tail(5).copy()
            recent_df['Time'] = recent_df['datetime'].dt.strftime('%I:%M %p')
            recent_df['Glucose'] = recent_df['glucose'].astype(str) + ' mg/dL'
            
            st.dataframe(
                recent_df[['Time', 'Glucose', 'trend']].rename(columns={'trend': 'Trend'}),
                use_container_width=True,
                hide_index=True
            )
        
        # Recent insulin table
        if st.session_state.insulin_doses:
            st.subheader("Recent Insulin Doses")
            insulin_df = pd.DataFrame(st.session_state.insulin_doses)
            insulin_df = insulin_df.sort_values('datetime', ascending=False)
            insulin_df['Time'] = insulin_df['datetime'].dt.strftime('%I:%M %p')
            insulin_df['Type'] = insulin_df['type'].str.title()
            insulin_df['Amount'] = insulin_df['amount'].astype(str) + ' units'
            
            st.dataframe(
                insulin_df[['Time', 'Type', 'Amount']].head(10),
                use_container_width=True,
                hide_index=True
            )
        
        # Recent meals table
        if st.session_state.meals:
            st.subheader("Recent Meals")
            meals_df = pd.DataFrame(st.session_state.meals)
            meals_df = meals_df.sort_values('datetime', ascending=False)
            meals_df['Time'] = meals_df['datetime'].dt.strftime('%I:%M %p')
            
            st.dataframe(
                meals_df[['Time', 'name', 'carbs', 'protein', 'calories']].rename(columns={
                    'name': 'Food', 'carbs': 'Carbs (g)', 'protein': 'Protein (g)', 'calories': 'Calories'
                }).head(10),
                use_container_width=True,
                hide_index=True
            )
    
    else:
        # No glucose data available
        st.markdown(f'''
            <div class="current-bg signal-loss">
                📡❌ No Glucose Data
                <br><small>Use manual entry in sidebar</small>
            </div>
        ''', unsafe_allow_html=True)
    
    # Connection status
    if error:
        st.error(f"Dexcom API Error: {error}")
    elif not username or not password:
        st.info("👈 Enter your Dexcom credentials in the sidebar to get started!")
    

if __name__ == "__main__":
    main()