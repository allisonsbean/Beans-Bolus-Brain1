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
import base64
import io
from PIL import Image
import anthropic

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
    st.session_state.basal_dose = 19
if 'daily_calorie_goal' not in st.session_state:
    st.session_state.daily_calorie_goal = 1200

# Diabetes settings
CARB_RATIO = 12
CORRECTION_FACTOR = 50
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
        dexcom = Dexcom(username="allisonsbean@gmail.com", password="Allison9")
        glucose_value = dexcom.get_current_glucose_reading()
        
        if glucose_value:
            timestamp = glucose_value.datetime
            
            if timestamp.tzinfo is None:
                timestamp = eastern.localize(timestamp)
            else:
                timestamp = timestamp.astimezone(eastern)
            
            glucose_data = {
                'value': glucose_value.value,
                'trend': glucose_value.trend_description,
                'trend_arrow': glucose_value.trend_arrow,
                'timestamp': timestamp
            }
            
            # Check if this reading already exists
            existing_reading = None
            for reading in st.session_state.glucose_readings:
                if (reading['timestamp'] == glucose_data['timestamp'] and 
                    reading['value'] == glucose_data['value']):
                    existing_reading = reading
                    break
            
            # Only add if it's a new reading
            if not existing_reading:
                st.session_state.glucose_readings.append(glucose_data)
                if len(st.session_state.glucose_readings) > 100:
                    st.session_state.glucose_readings = st.session_state.glucose_readings[-100:]
                
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
    carb_bolus = carbs / CARB_RATIO
    protein_carb_equivalent = protein * 0.1
    protein_bolus = protein_carb_equivalent / CARB_RATIO
    correction_needed = max(0, current_glucose - TARGET_GLUCOSE)
    correction_bolus = correction_needed / CORRECTION_FACTOR
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
    """Create glucose trend chart"""
    if not st.session_state.glucose_readings:
        return None
    
    df_glucose = pd.DataFrame(st.session_state.glucose_readings)
    df_glucose['timestamp'] = pd.to_datetime(df_glucose['timestamp'])
    
    cutoff_time = datetime.now(eastern) - timedelta(hours=12)
    df_glucose = df_glucose[df_glucose['timestamp'] >= cutoff_time]
    
    if df_glucose.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_glucose['timestamp'],
        y=df_glucose['value'],
        mode='lines+markers',
        name='Glucose',
        line=dict(color='blue', width=3),
        marker=dict(size=6)
    ))
    
    fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Target Range")
    fig.add_hline(y=130, line_dash="dash", line_color="green")
    
    fig.update_layout(
        title="12-Hour Glucose Trend",
        xaxis_title="Time",
        yaxis_title="Glucose (mg/dL)",
        height=400,
        showlegend=False
    )
    
    return fig

def analyze_food_photo(image_file):
    """Analyze food photo using Claude Vision API"""
    try:
        api_key = st.secrets.get("claude", {}).get("api_key")
        if not api_key:
            return {"error": "Claude API key not configured"}
        
        current_time = datetime.now()
        if 'last_api_request' in st.session_state:
            time_since_last = (current_time - st.session_state.last_api_request).total_seconds()
            if time_since_last < 5:
                return {"error": f"Please wait {5 - int(time_since_last)} more seconds"}
        
        if hasattr(image_file, 'read'):
            image_bytes = image_file.read()
        else:
            image_bytes = image_file
            
        img = Image.open(io.BytesIO(image_bytes))
        
        max_size = 1024
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        client = anthropic.Anthropic(api_key=api_key)
        st.session_state.last_api_request = current_time
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=800,
            temperature=0.1,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": """Analyze this food photo for diabetes management and calorie tracking. I need accurate estimates for insulin dosing and staying within my 1200 calorie daily goal.

Please respond with ONLY a JSON object in this exact format:

{
    "foods": [
        {
            "name": "food_name",
            "portion": "1 cup",
            "carbs": 30,
            "protein": 5,
            "calories": 150
        }
    ],
    "total_carbs": 30,
    "total_protein": 5,
    "total_calories": 150,
    "notes": "cooking method, portion confidence, calorie density notes"
}

Be conservative with portions since this is for medical insulin dosing and calorie tracking."""
                        }
                    ]
                }
            ]
        )
        
        content = message.content[0].text
        
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            analysis = json.loads(json_match.group())
            analysis['success'] = True
            return analysis
        else:
            return {"error": "Could not parse Claude response"}
            
    except Exception as e:
        return {"error": f"Analysis failed: {e}"}

def run_basic_pattern_analysis():
    """Simplified pattern analysis"""
    if len(st.session_state.glucose_readings) < 10:
        return "Need at least 10 glucose readings for pattern analysis"
    
    df = pd.DataFrame(st.session_state.glucose_readings)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['in_range'] = ((df['value'] >= 80) & (df['value'] <= 130))
    
    insights = []
    
    # Time in range analysis
    tir = df['in_range'].mean()
    if tir < 0.7:
        insights.append(f"⚠️ Time in range is {tir*100:.0f}% - target is 70%+")
    else:
        insights.append(f"🎉 Great time in range: {tir*100:.0f}%!")
    
    # Meal and calorie analysis
    if st.session_state.meal_log:
        meal_df = pd.DataFrame(st.session_state.meal_log)
        total_days = max(1, (meal_df['timestamp'].max() - meal_df['timestamp'].min()).days + 1)
        avg_daily_calories = meal_df['calories'].sum() / total_days
        
        if avg_daily_calories > 0:
            insights.append(f"🔥 Average daily calories: {avg_daily_calories:.0f} (Goal: {st.session_state.daily_calorie_goal})")
    
    return insights

def main():
    st.title("🧠 Bean's Bolus Brain")
    st.subheader("AI-Powered Diabetes Management Dashboard")
    
    glucose_data = get_dexcom_data()
    current_iob = calculate_iob()
    
    # Predictive alerts
    prediction_engine = GlucosePredictionEngine()
    
    if glucose_data and len(st.session_state.glucose_readings) >= 3:
        recent_readings = st.session_state.glucose_readings[-6:]
        prediction_results = prediction_engine.predict_glucose_trends(recent_readings, current_iob)
        
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
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🩸 Current Glucose")
        display_glucose_status(glucose_data)
        
        st.markdown("### 💉 Insulin on Board")
        iob_color = "orange" if current_iob > 3 else "green"
        st.markdown(f"<h2 style='text-align: center; color: {iob_color};'>{current_iob:.1f} units</h2>", 
                   unsafe_allow_html=True)
        
        chart = create_glucose_chart()
        if chart:
            st.plotly_chart(chart, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Today's Summary")
        
        today = datetime.now(eastern).date()
        today_meals = [m for m in st.session_state.meal_log if m['timestamp'].date() == today]
        today_insulin = [i for i in st.session_state.insulin_log if i['timestamp'].date() == today and i['type'] == 'bolus']
        
        total_carbs = sum(meal['carbs'] for meal in today_meals)
        total_protein = sum(meal['protein'] for meal in today_meals)
        total_calories = sum(meal['calories'] for meal in today_meals)
        total_insulin = sum(insulin['dose'] for insulin in today_insulin)
        
        st.metric("Total Carbs", f"{total_carbs}g")
        st.metric("Total Protein", f"{total_protein}g")
        st.metric("Total Calories", f"{total_calories}")
        st.metric("Total Bolus", f"{total_insulin:.1f}u")
        
        # Calorie progress
        progress_percent = min(1.0, total_calories / st.session_state.daily_calorie_goal)
        st.progress(progress_percent)
        
        remaining_cals = st.session_state.daily_calorie_goal - total_calories
        if remaining_cals > 0:
            st.success(f"🎯 {remaining_cals} calories remaining today")
        else:
            st.warning(f"⚠️ {abs(remaining_cals)} calories over goal today")
        
        if st.session_state.glucose_readings:
            today_glucose = [g for g in st.session_state.glucose_readings if g['timestamp'].date() == today]
            if today_glucose:
                in_range = sum(1 for g in today_glucose if GLUCOSE_RANGE[0] <= g['value'] <= GLUCOSE_RANGE[1])
                time_in_range = (in_range / len(today_glucose)) * 100
                st.metric("Time in Range", f"{time_in_range:.0f}%")
    
    with st.sidebar:
        st.header("📝 Quick Logging")
        
        with st.expander("🎯 Daily Goals"):
            new_calorie_goal = st.number_input("Daily Calorie Goal", 
                                             min_value=800, max_value=3000, 
                                             value=st.session_state.daily_calorie_goal, step=50)
            if st.button("Update Calorie Goal"):
                st.session_state.daily_calorie_goal = new_calorie_goal
                st.success(f"Updated daily calorie goal to {new_calorie_goal}")
                st.rerun()
        
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
        
        with st.expander("💉 Log Bolus"):
            bolus_dose = st.number_input("Bolus dose (units)", min_value=0.0, max_value=20.0, step=0.5)
            bolus_notes = st.text_input("Notes (optional)")
            if st.button("Log Bolus"):
                add_insulin_entry(bolus_dose, 'bolus', bolus_notes)
                st.success(f"Logged {bolus_dose}u bolus")
                st.rerun()
        
        with st.expander("🔄 Log Basal"):
            basal_dose = st.number_input("Daily basal (units)", min_value=0.0, max_value=50.0, value=float(st.session_state.basal_dose), step=1.0)
            if st.button("Update Basal"):
                st.session_state.basal_dose = basal_dose
                add_insulin_entry(basal_dose, 'basal', f"Daily basal: {basal_dose}u")
                st.success(f"Updated daily basal to {basal_dose}u")
                st.rerun()
        
        with st.expander("🍽️ Log Meal & Get Bolus Suggestion"):
            st.markdown("**📸 Photo Analysis with Claude AI**")
            uploaded_file = st.file_uploader("Take/upload photo of your meal", type=['png', 'jpg', 'jpeg'])
            
            if uploaded_file:
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("🤖 Analyze Photo"):
                        with st.spinner("Claude is analyzing your meal..."):
                            analysis = analyze_food_photo(uploaded_file)
                            if analysis.get('success'):
                                st.success("✅ Analysis complete!")
                                st.session_state.photo_analysis = analysis
                            else:
                                st.error(f"❌ {analysis.get('error', 'Analysis failed')}")
                with col2:
                    st.image(uploaded_file, width=150)
            
            if 'photo_analysis' in st.session_state and st.session_state.photo_analysis.get('success'):
                analysis = st.session_state.photo_analysis
                st.markdown("**🤖 Claude Analysis:**")
                
                for food in analysis.get('foods', []):
                    st.write(f"• {food['name']} ({food['portion']}): {food['carbs']}g carbs, {food['calories']} cal")
                
                suggested_carbs = analysis.get('total_carbs', 30)
                suggested_protein = analysis.get('total_protein', 0)
                suggested_calories = analysis.get('total_calories', 0)
                
                if analysis.get('notes'):
                    st.info(f"**Notes:** {analysis['notes']}")
                
                if st.button("✅ Use Claude Analysis"):
                    st.session_state.ai_carbs = suggested_carbs
                    st.session_state.ai_protein = suggested_protein
                    st.session_state.ai_calories = suggested_calories
                    st.success("Claude analysis applied!")
            
            st.markdown("---")
            st.markdown("**📝 Manual Entry**")
            
            default_carbs = st.session_state.get('ai_carbs', 30)
            default_protein = st.session_state.get('ai_protein', 0)
            default_calories = st.session_state.get('ai_calories', 0)
            
            meal_carbs = st.number_input("Carbs (g)", min_value=0, max_value=200, value=int(default_carbs))
            meal_protein = st.number_input("Protein (g)", min_value=0, max_value=100, value=int(default_protein))
            meal_calories = st.number_input("Calories", min_value=0, max_value=2000, value=int(default_calories))
            meal_description = st.text_input("Meal description", value="Claude Photo Analysis" if 'photo_analysis' in st.session_state else "")
            
            if glucose_data:
                bolus_suggestion = calculate_bolus_suggestion(meal_carbs, meal_protein, glucose_data['value'], current_iob)
                
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
                        add_insulin_entry(bolus_suggestion['total_bolus'], 'bolus', f"Meal bolus: {meal_description}")
                        st.success(f"Logged meal + {bolus_suggestion['total_bolus']}u bolus!")
                        st.rerun()
            else:
                if st.button("Log Meal"):
                    add_meal_entry(meal_carbs, meal_protein, meal_calories, meal_description)
                    st.success("Meal logged!")
                    st.rerun()
        
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
    
    # Pattern analysis
    st.markdown("---")
    
    if st.button("🧠 Analyze My Patterns"):
        with st.spinner("Analyzing your diabetes patterns..."):
            insights = run_basic_pattern_analysis()
            
            if isinstance(insights, str):
                st.warning(insights)
            else:
                st.markdown("### 🔍 AI Pattern Insights")
                for insight in insights:
                    st.info(insight)
    
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
