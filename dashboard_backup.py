import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pytz
from pydexcom import Dexcom
import requests
import json
from glucose_predictor import GlucosePredictionEngine

# Page configuration
st.set_page_config(
    page_title="Bean's Bolus Brain 🧠",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for data storage
if 'glucose_readings' not in st.session_state:
    st.session_state.glucose_readings = []
if 'insulin_log' not in st.session_state:
    st.session_state.insulin_log = []
if 'meal_log' not in st.session_state:
    st.session_state.meal_log = []
if 'basal_dose' not in st.session_state:
    st.session_state.basal_dose = 20

# Diabetes settings
CARB_RATIO = 12  # 1 unit per 12g carbs
CORRECTION_FACTOR = 50  # 1 unit per 50 mg/dL above target
TARGET_GLUCOSE = 115
GLUCOSE_RANGE = (80, 130)
MAX_BOLUS = 20
IOB_DURATION_HOURS = 4

# Eastern Time timezone
eastern = pytz.timezone('US/Eastern')

@st.cache_data(ttl=60)
def get_dexcom_data():
    """Get real-time glucose data from Dexcom Share"""
    try:
        dexcom = Dexcom(username=st.secrets["dexcom"]["username"], password=st.secrets["dexcom"]["password"])
        glucose_value = dexcom.get_current_glucose_reading()
        
        if glucose_value:
            glucose_data = {
                'value': glucose_value.value,
                'trend': glucose_value.trend_description,
                'trend_arrow': glucose_value.trend_arrow,
                'timestamp': glucose_value.time.astimezone(eastern)
            }
            
            # Add to session state if not duplicate
            if not st.session_state.glucose_readings or \
               st.session_state.glucose_readings[-1]['timestamp'] != glucose_data['timestamp']:
                st.session_state.glucose_readings.append(glucose_data)
                
            return glucose_data
        return None
    except Exception as e:
        st.error(f"Dexcom connection error: {e}")
        return None

def calculate_iob():
    """Calculate current insulin on board using 4-hour linear decay"""
    now = datetime.now(eastern)
    total_iob = 0
    
    for entry in st.session_state.insulin_log:
        entry_time = entry['timestamp']
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time).replace(tzinfo=eastern)
        
        hours_elapsed = (now - entry_time).total_seconds() / 3600
        
        if hours_elapsed < IOB_DURATION_HOURS and entry['type'] == 'bolus':
            remaining_fraction = max(0, (IOB_DURATION_HOURS - hours_elapsed) / IOB_DURATION_HOURS)
            total_iob += entry['dose'] * remaining_fraction
    
    return total_iob

def add_insulin_entry(dose, insulin_type, notes=""):
    """Add insulin entry to log"""
    entry = {
        'timestamp': datetime.now(eastern),
        'type': insulin_type,
        'dose': dose,
        'notes': notes
    }
    st.session_state.insulin_log.append(entry)

def add_meal_entry(carbs, protein=0, calories=0, description=""):
    """Add meal entry to log"""
    entry = {
        'timestamp': datetime.now(eastern),
        'carbs': carbs,
        'protein': protein,
        'calories': calories,
        'description': description
    }
    st.session_state.meal_log.append(entry)

def calculate_bolus_suggestion(carbs, protein, current_glucose, current_iob):
    """Calculate bolus suggestion with IOB adjustment"""
    # Carb bolus
    carb_bolus = carbs / CARB_RATIO
    
    # Protein bolus (10% of protein grams converted to carb equivalent)
    protein_carb_equivalent = protein * 0.1
    protein_bolus = protein_carb_equivalent / CARB_RATIO
    
    # Correction bolus (adjusted for IOB)
    correction_needed = max(0, current_glucose - TARGET_GLUCOSE)
    correction_bolus = correction_needed / CORRECTION_FACTOR
    
    # Adjust for IOB
    adjusted_correction = max(0, correction_bolus - current_iob)
    
    total_bolus = carb_bolus + protein_bolus + adjusted_correction
    total_bolus = min(total_bolus, MAX_BOLUS)
    
    return {
        'carb_bolus': round(carb_bolus, 1),
        'protein_bolus': round(protein_bolus, 1),
        'correction_bolus': round(adjusted_correction, 1),
        'total_bolus': round(total_bolus),
        'iob_adjustment': round(current_iob, 1)
    }

def display_glucose_status(glucose_data):
    """Display current glucose status with color coding"""
    if not glucose_data:
        st.error("❌ No glucose data available")
        return
    
    value = glucose_data['value']
    trend = glucose_data.get('trend', 'Unknown')
    arrow = glucose_data.get('trend_arrow', '')
    
    # Determine status and color
    if value < 70:
        status = "🔴 URGENT LOW"
        color = "red"
    elif value < 80:
        status = "🟠 LOW"
        color = "orange"
    elif value <= 130:
        status = "🟢 IN RANGE"
        color = "green"
    elif value <= 180:
        status = "🟠 ATTENTION"
        color = "orange"
    else:
        status = "🔴 HIGH"
        color = "red"
    
    st.markdown(f"<h1 style='text-align: center; color: {color};'>{value} mg/dL {arrow}</h1>", 
                unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>{status}</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;'>Trend: {trend}</p>", unsafe_allow_html=True)

def create_glucose_chart():
    """Create glucose trend chart with meal and insulin markers"""
    if not st.session_state.glucose_readings:
        return None
    
    # Convert to DataFrame
    df_glucose = pd.DataFrame(st.session_state.glucose_readings)
    df_glucose['timestamp'] = pd.to_datetime(df_glucose['timestamp'])
    
    # Filter to last 12 hours
    cutoff_time = datetime.now(eastern) - timedelta(hours=12)
    df_glucose = df_glucose[df_glucose['timestamp'] >= cutoff_time]
    
    if df_glucose.empty:
        return None
    
    # Create the chart
    fig = go.Figure()
    
    # Add glucose line
    fig.add_trace(go.Scatter(
        x=df_glucose['timestamp'],
        y=df_glucose['value'],
        mode='lines+markers',
        name='Glucose',
        line=dict(color='blue', width=3),
        marker=dict(size=6)
    ))
    
    # Add target range
    fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Target Range")
    fig.add_hline(y=130, line_dash="dash", line_color="green")
    
    # Add meal markers
    for meal in st.session_state.meal_log:
        meal_time = meal['timestamp']
        if isinstance(meal_time, str):
            meal_time = datetime.fromisoformat(meal_time).replace(tzinfo=eastern)
        
        if meal_time >= cutoff_time:
            fig.add_vline(x=meal_time, line_dash="dot", line_color="orange", 
                         annotation_text=f"🍽️ {meal['carbs']}g")
    
    # Add insulin markers
    for insulin in st.session_state.insulin_log:
        insulin_time = insulin['timestamp']
        if isinstance(insulin_time, str):
            insulin_time = datetime.fromisoformat(insulin_time).replace(tzinfo=eastern)
        
        if insulin_time >= cutoff_time and insulin['type'] == 'bolus':
            fig.add_vline(x=insulin_time, line_dash="dot", line_color="purple",
                         annotation_text=f"💉 {insulin['dose']}u")
    
    fig.update_layout(
        title="12-Hour Glucose Trend",
        xaxis_title="Time",
        yaxis_title="Glucose (mg/dL)",
        height=400,
        showlegend=False
    )
    
    return fig

# Main app layout
def main():
    st.title("🧠 Bean's Bolus Brain")
    st.subheader("AI-Powered Diabetes Management Dashboard")
    
    # Get current data
    glucose_data = get_dexcom_data()
    current_iob = calculate_iob()
    
    # NEW: Add predictive alerts
    prediction_engine = GlucosePredictionEngine()
    
    if glucose_data and len(st.session_state.glucose_readings) >= 2:
        # Prepare data for prediction
        recent_readings = st.session_state.glucose_readings[-6:]  # Last 6 readings
        prediction_results = prediction_engine.predict_glucose_trends(
            recent_readings, current_iob
        )
        
        # Display alerts prominently at the top
        if prediction_results['alerts']:
            st.markdown("## 🚨 GLUCOSE ALERTS")
            for alert in prediction_results['alerts']:
                if alert['severity'] == 'URGENT':
                    st.error(f"**{alert['message']}**")
                    st.error(f"**Action needed:** {alert['recommendation']}")
                else:
                    st.warning(f"**{alert['message']}**")
                    st.warning(f"**Recommendation:** {alert['recommendation']}")
            st.markdown("---")
        
        # Display predictions
        if prediction_results['predictions']:
            st.markdown("## 📈 Glucose Predictions")
            
            pred_col1, pred_col2, pred_col3 = st.columns(3)
            
            with pred_col1:
                if '1h' in prediction_results['predictions']:
                    pred_1h = prediction_results['predictions']['1h']
                    st.metric(
                        label="1 Hour Prediction",
                        value=f"{pred_1h['predicted_value']:.0f} mg/dL",
                        help=f"Confidence: {pred_1h['confidence']:.0f}%"
                    )
                    
            with pred_col2:
                if '2h' in prediction_results['predictions']:
                    pred_2h = prediction_results['predictions']['2h']
                    st.metric(
                        label="2 Hour Prediction", 
                        value=f"{pred_2h['predicted_value']:.0f} mg/dL",
                        help=f"Confidence: {pred_2h['confidence']:.0f}%"
                    )
            
            with pred_col3:
                st.metric(
                    label="Prediction Status",
                    value="✅ Active" if prediction_results['data_quality']['usable'] else "❌ Limited",
                    help="Based on recent glucose trend analysis"
                )
            
            st.markdown("---")
    
    # Main dashboard layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Current glucose status
        st.markdown("### 🩸 Current Glucose")
        display_glucose_status(glucose_data)
        
        # IOB display (prominent)
        st.markdown("### 💉 Insulin on Board")
        iob_color = "orange" if current_iob > 3 else "green"
        st.markdown(f"<h2 style='text-align: center; color: {iob_color};'>{current_iob:.1f} units</h2>", 
                   unsafe_allow_html=True)
        
        # Glucose chart
        chart = create_glucose_chart()
        if chart:
            st.plotly_chart(chart, use_container_width=True)
    
    with col2:
        # Quick stats
        st.markdown("### 📊 Today's Summary")
        
        today = datetime.now(eastern).date()
        today_meals = [m for m in st.session_state.meal_log 
                      if m['timestamp'].date() == today]
        today_insulin = [i for i in st.session_state.insulin_log 
                        if i['timestamp'].date() == today and i['type'] == 'bolus']
        
        total_carbs = sum(meal['carbs'] for meal in today_meals)
        total_protein = sum(meal['protein'] for meal in today_meals)
        total_insulin = sum(insulin['dose'] for insulin in today_insulin)
        
        st.metric("Total Carbs", f"{total_carbs}g")
        st.metric("Total Protein", f"{total_protein}g")
        st.metric("Total Bolus", f"{total_insulin:.1f}u")
        
        # Calculate time in range if we have glucose data
        if st.session_state.glucose_readings:
            today_glucose = [g for g in st.session_state.glucose_readings 
                           if g['timestamp'].date() == today]
            if today_glucose:
                in_range = sum(1 for g in today_glucose 
                             if GLUCOSE_RANGE[0] <= g['value'] <= GLUCOSE_RANGE[1])
                time_in_range = (in_range / len(today_glucose)) * 100
                st.metric("Time in Range", f"{time_in_range:.0f}%")
    
    # Sidebar for logging
    with st.sidebar:
        st.header("📝 Quick Logging")
        
        # Manual glucose entry
        with st.expander("🩸 Manual Glucose Entry"):
            manual_glucose = st.number_input("Glucose (mg/dL)", min_value=40, max_value=400, value=120)
            if st.button("Log Glucose"):
                glucose_entry = {
                    'value': manual_glucose,
                    'trend': 'Manual Entry',
                    'trend_arrow': '',
                    'timestamp': datetime.now(eastern)
                }
                st.session_state.glucose_readings.append(glucose_entry)
                st.success(f"Logged {manual_glucose} mg/dL")
                st.rerun()
        
        # Bolus logging
        with st.expander("💉 Log Bolus"):
            bolus_dose = st.number_input("Bolus dose (units)", min_value=0.0, max_value=20.0, step=0.5)
            bolus_notes = st.text_input("Notes (optional)")
            if st.button("Log Bolus"):
                add_insulin_entry(bolus_dose, 'bolus', bolus_notes)
                st.success(f"Logged {bolus_dose}u bolus")
                st.rerun()
        
        # Basal logging
        with st.expander("🔄 Log Basal"):
            basal_dose = st.number_input("Daily basal (units)", 
                                       min_value=0.0, max_value=50.0, 
                                       value=st.session_state.basal_dose, step=1.0)
            if st.button("Update Basal"):
                st.session_state.basal_dose = basal_dose
                add_insulin_entry(basal_dose, 'basal', f"Daily basal: {basal_dose}u")
                st.success(f"Updated daily basal to {basal_dose}u")
                st.rerun()
        
        # Meal logging with bolus suggestion
        with st.expander("🍽️ Log Meal & Get Bolus Suggestion"):
            meal_carbs = st.number_input("Carbs (g)", min_value=0, max_value=200, value=30)
            meal_protein = st.number_input("Protein (g)", min_value=0, max_value=100, value=0)
            meal_calories = st.number_input("Calories", min_value=0, max_value=2000, value=0)
            meal_description = st.text_input("Meal description")
            
            if glucose_data:
                bolus_suggestion = calculate_bolus_suggestion(
                    meal_carbs, meal_protein, glucose_data['value'], current_iob
                )
                
                st.markdown("**Bolus Suggestion:**")
                st.write(f"• Carb bolus: {bolus_suggestion['carb_bolus']}u")
                if bolus_suggestion['protein_bolus'] > 0:
                    st.write(f"• Protein bolus: {bolus_suggestion['protein_bolus']}u")
                if bolus_suggestion['correction_bolus'] > 0:
                    st.write(f"• Correction: {bolus_suggestion['correction_bolus']}u")
                st.write(f"**Total suggested: {bolus_suggestion['total_bolus']}u**")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Log Meal Only"):
                        add_meal_entry(meal_carbs, meal_protein, meal_calories, meal_description)
                        st.success("Meal logged!")
                        st.rerun()
                
                with col2:
                    if st.button("Log Meal + Bolus"):
                        add_meal_entry(meal_carbs, meal_protein, meal_calories, meal_description)
                        add_insulin_entry(bolus_suggestion['total_bolus'], 'bolus', 
                                        f"Meal bolus: {meal_description}")
                        st.success(f"Logged meal + {bolus_suggestion['total_bolus']}u bolus!")
                        st.rerun()
            else:
                if st.button("Log Meal"):
                    add_meal_entry(meal_carbs, meal_protein, meal_calories, meal_description)
                    st.success("Meal logged!")
                    st.rerun()
        
        # Correction suggestion
        if glucose_data and glucose_data['value'] > 130:
            st.markdown("### 🎯 Correction Suggestion")
            correction_bolus = max(0, (glucose_data['value'] - TARGET_GLUCOSE) / CORRECTION_FACTOR - current_iob)
            
            if correction_bolus > 0.5:
                st.warning(f"Consider {round(correction_bolus)}u correction")
                if st.button("Log Correction"):
                    add_insulin_entry(round(correction_bolus), 'bolus', "Correction bolus")
                    st.success(f"Logged {round(correction_bolus)}u correction!")
                    st.rerun()
            else:
                st.info("No correction needed (IOB sufficient)")
    
    # Data tables in expandable sections
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["📈 Glucose History", "💉 Insulin History", "🍽️ Meal History"])
    
    with tab1:
        if st.session_state.glucose_readings:
            glucose_df = pd.DataFrame(st.session_state.glucose_readings)
            glucose_df['timestamp'] = pd.to_datetime(glucose_df['timestamp']).dt.strftime('%m/%d %H:%M')
            st.dataframe(glucose_df[['timestamp', 'value', 'trend']].head(20), use_container_width=True)
        else:
            st.info("No glucose readings yet")
    
    with tab2:
        if st.session_state.insulin_log:
            insulin_df = pd.DataFrame(st.session_state.insulin_log)
            insulin_df['timestamp'] = pd.to_datetime(insulin_df['timestamp']).dt.strftime('%m/%d %H:%M')
            st.dataframe(insulin_df[['timestamp', 'type', 'dose', 'notes']].head(20), use_container_width=True)
        else:
            st.info("No insulin entries yet")
    
    with tab3:
        if st.session_state.meal_log:
            meal_df = pd.DataFrame(st.session_state.meal_log)
            meal_df['timestamp'] = pd.to_datetime(meal_df['timestamp']).dt.strftime('%m/%d %H:%M')
            st.dataframe(meal_df[['timestamp', 'carbs', 'protein', 'calories', 'description']].head(20), use_container_width=True)
        else:
            st.info("No meal entries yet")

if __name__ == "__main__":
    main()