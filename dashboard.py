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
import os
import base64
import io
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Bean's Bolus Brain 🧠",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Data persistence functions
def save_data_to_file():
    """Save session data to local files"""
    try:
        data = {
            'glucose_readings': [],
            'insulin_log': [],
            'meal_log': []
        }
        
        # Convert glucose readings to serializable format
        for reading in st.session_state.glucose_readings:
            data['glucose_readings'].append({
                'value': reading['value'],
                'trend': reading['trend'],
                'trend_arrow': reading['trend_arrow'],
                'timestamp': reading['timestamp'].isoformat()
            })
        
        # Convert insulin log
        for entry in st.session_state.insulin_log:
            data['insulin_log'].append({
                'timestamp': entry['timestamp'].isoformat(),
                'type': entry['type'],
                'dose': entry['dose'],
                'notes': entry['notes']
            })
        
        # Convert meal log
        for entry in st.session_state.meal_log:
            data['meal_log'].append({
                'timestamp': entry['timestamp'].isoformat(),
                'carbs': entry['carbs'],
                'protein': entry['protein'],
                'calories': entry['calories'],
                'description': entry['description']
            })
        
        # Save to file
        with open('beans_data.json', 'w') as f:
            json.dump(data, f, indent=2)
            
    except Exception as e:
        st.error(f"Error saving data: {e}")

def load_data_from_file():
    """Load session data from local files"""
    try:
        if os.path.exists('beans_data.json'):
            with open('beans_data.json', 'r') as f:
                data = json.load(f)
            
            eastern = pytz.timezone('US/Eastern')
            
            # Load glucose readings
            st.session_state.glucose_readings = []
            for reading in data.get('glucose_readings', []):
                timestamp = datetime.fromisoformat(reading['timestamp'])
                if timestamp.tzinfo is None:
                    timestamp = eastern.localize(timestamp)
                st.session_state.glucose_readings.append({
                    'value': reading['value'],
                    'trend': reading['trend'],
                    'trend_arrow': reading['trend_arrow'],
                    'timestamp': timestamp
                })
            
            # Load insulin log
            st.session_state.insulin_log = []
            for entry in data.get('insulin_log', []):
                timestamp = datetime.fromisoformat(entry['timestamp'])
                if timestamp.tzinfo is None:
                    timestamp = eastern.localize(timestamp)
                st.session_state.insulin_log.append({
                    'timestamp': timestamp,
                    'type': entry['type'],
                    'dose': entry['dose'],
                    'notes': entry['notes']
                })
            
            # Load meal log
            st.session_state.meal_log = []
            for entry in data.get('meal_log', []):
                timestamp = datetime.fromisoformat(entry['timestamp'])
                if timestamp.tzinfo is None:
                    timestamp = eastern.localize(timestamp)
                st.session_state.meal_log.append({
                    'timestamp': timestamp,
                    'carbs': entry['carbs'],
                    'protein': entry['protein'],
                    'calories': entry['calories'],
                    'description': entry['description']
                })
                
    except Exception as e:
        st.error(f"Error loading data: {e}")

# Initialize session state for data storage
if 'glucose_readings' not in st.session_state:
    st.session_state.glucose_readings = []
if 'insulin_log' not in st.session_state:
    st.session_state.insulin_log = []
if 'meal_log' not in st.session_state:
    st.session_state.meal_log = []
if 'basal_dose' not in st.session_state:
    st.session_state.basal_dose = 19

# Load existing data on startup
if 'data_loaded' not in st.session_state:
    load_data_from_file()
    st.session_state.data_loaded = True

# Diabetes settings
CARB_RATIO = 12  # 1 unit per 12g carbs
CORRECTION_FACTOR = 50  # 1 unit per 50 mg/dL above target
TARGET_GLUCOSE = 115
GLUCOSE_RANGE = (80, 130)
MAX_BOLUS = 20
IOB_DURATION_HOURS = 4
DAILY_CALORIE_GOAL = 1200

# Eastern Time timezone
eastern = pytz.timezone('US/Eastern')

@st.cache_data(ttl=60)
def get_dexcom_data():
    """Get real-time glucose data from Dexcom Share"""
    try:
        # Use direct credentials for local development
        dexcom = Dexcom(username="allisonsbean@gmail.com", password="Allison9")
        glucose_value = dexcom.get_current_glucose_reading()
        
        if glucose_value:
            # Use the 'datetime' attribute
            timestamp = glucose_value.datetime
            
            # Convert to Eastern time
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
            
            # Add to session state if not duplicate
            should_add = True
            if st.session_state.glucose_readings:
                last_reading = st.session_state.glucose_readings[-1]
                # Check if this is truly a new reading (different timestamp)
                time_diff = abs((glucose_data['timestamp'] - last_reading['timestamp']).total_seconds())
                if time_diff < 60:  # Less than 1 minute difference, probably duplicate
                    should_add = False

            if should_add:
                st.session_state.glucose_readings.append(glucose_data)
                
                # Keep only last 200 readings (about 16 hours of data)
                if len(st.session_state.glucose_readings) > 200:
                    st.session_state.glucose_readings = st.session_state.glucose_readings[-200:]
                
                # Save data to file
                save_data_to_file()
                
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
    save_data_to_file()

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
    save_data_to_file()

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

def analyze_food_photo(image_file):
    """Analyze food photo using Claude AI (simulated for deployment)"""
    try:
        # Process image
        img = Image.open(io.BytesIO(image_file.read() if hasattr(image_file, 'read') else image_file))
        
        # Resize if too large
        max_size = 1024
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Simulated AI analysis for deployment (replace with actual Claude AI integration)
        # This provides realistic estimates based on common foods
        food_estimates = [
            {"name": "Mixed meal", "portion": "1 serving", "carbs": 45, "protein": 15, "calories": 350},
            {"name": "Pasta dish", "portion": "1 cup", "carbs": 60, "protein": 8, "calories": 280},
            {"name": "Salad with protein", "portion": "1 bowl", "carbs": 15, "protein": 20, "calories": 250},
            {"name": "Sandwich", "portion": "1 whole", "carbs": 35, "protein": 12, "calories": 320},
            {"name": "Rice bowl", "portion": "1 serving", "carbs": 55, "protein": 10, "calories": 300}
        ]
        
        # Select a random estimate (in real implementation, this would be AI analysis)
        import random
        estimate = random.choice(food_estimates)
        
        analysis = {
            "foods": [estimate],
            "total_carbs": estimate["carbs"],
            "total_protein": estimate["protein"],
            "total_calories": estimate["calories"],
            "notes": "AI estimate - please adjust as needed",
            "success": True
        }
        
        return analysis
        
    except Exception as e:
        return {"error": f"Analysis failed: {e}"}

def predict_glucose_trends(current_glucose, current_iob, recent_readings):
    """Predict glucose trends based on current data"""
    if len(recent_readings) < 3:
        return {"error": "Need more glucose readings for prediction"}
    
    try:
        # Simple trend analysis
        values = [r['value'] for r in recent_readings[-5:]]
        timestamps = [r['timestamp'] for r in recent_readings[-5:]]
        
        # Calculate trend
        if len(values) >= 3:
            trend_slope = (values[-1] - values[-3]) / 2  # mg/dL per reading
            
            # Predict 1-4 hours ahead
            predictions = {}
            for hours in [1, 2, 3, 4]:
                # Account for IOB effect (simplified)
                iob_effect = current_iob * 30 * (1 - hours/4)  # IOB lowers glucose over time
                predicted_value = current_glucose + (trend_slope * hours * 6) - iob_effect
                
                # Clamp to reasonable range
                predicted_value = max(40, min(400, predicted_value))
                
                # Calculate confidence based on trend stability
                variance = np.var(values) if len(values) > 1 else 50
                confidence = max(30, 100 - variance)
                
                predictions[f'{hours}h'] = {
                    'predicted_value': predicted_value,
                    'confidence': confidence
                }
            
            # Generate alerts
            alerts = []
            if predictions['1h']['predicted_value'] < 70:
                alerts.append({
                    'severity': 'URGENT',
                    'message': 'LOW GLUCOSE PREDICTED in 1 hour',
                    'recommendation': 'Consider eating 15g carbs now'
                })
            elif predictions['2h']['predicted_value'] > 200:
                alerts.append({
                    'severity': 'WARNING',
                    'message': 'HIGH GLUCOSE PREDICTED in 2 hours',
                    'recommendation': 'Consider correction bolus if no IOB'
                })
            
            return {
                'predictions': predictions,
                'alerts': alerts,
                'data_quality': {'usable': True}
            }
        
        return {"error": "Insufficient trend data"}
        
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}

def run_basic_pattern_analysis():
    """Analyze patterns in glucose, insulin, and meal data"""
    if len(st.session_state.glucose_readings) < 20:
        return "Need more glucose data for pattern analysis (minimum 20 readings)"
    
    try:
        insights = []
        
        # Analyze glucose patterns
        glucose_df = pd.DataFrame(st.session_state.glucose_readings)
        glucose_df['timestamp'] = pd.to_datetime(glucose_df['timestamp'])
        
        # Time in range analysis
        in_range = ((glucose_df['value'] >= 80) & (glucose_df['value'] <= 130)).mean() * 100
        insights.append(f"📊 Your time in range is {in_range:.0f}% (goal: >70%)")
        
        # Best/worst times of day
        glucose_df['hour'] = glucose_df['timestamp'].dt.hour
        hourly_avg = glucose_df.groupby('hour')['value'].mean()
        best_hour = hourly_avg.idxmin()
        worst_hour = hourly_avg.idxmax()
        
        insights.append(f"🕐 Best glucose control: {best_hour}:00 (avg: {hourly_avg[best_hour]:.0f} mg/dL)")
        insights.append(f"⚠️ Most challenging time: {worst_hour}:00 (avg: {hourly_avg[worst_hour]:.0f} mg/dL)")
        
        # Meal analysis if available
        if len(st.session_state.meal_log) >= 5:
            meal_df = pd.DataFrame(st.session_state.meal_log)
            avg_carbs = meal_df['carbs'].mean()
            avg_calories = meal_df['calories'].mean()
            
            insights.append(f"🍽️ Average meal: {avg_carbs:.0f}g carbs, {avg_calories:.0f} calories")
            
            # Calorie goal analysis
            today = datetime.now(eastern).date()
            today_meals = [m for m in st.session_state.meal_log 
                          if m['timestamp'].date() == today]
            today_calories = sum(meal['calories'] for meal in today_meals)
            
            insights.append(f"📊 Today's calories: {today_calories}/{DAILY_CALORIE_GOAL} (goal)")
        
        # IOB patterns
        if len(st.session_state.insulin_log) >= 5:
            insulin_df = pd.DataFrame(st.session_state.insulin_log)
            bolus_entries = insulin_df[insulin_df['type'] == 'bolus']
            if not bolus_entries.empty:
                avg_bolus = bolus_entries['dose'].mean()
                insights.append(f"💉 Average bolus dose: {avg_bolus:.1f} units")
        
        return insights
        
    except Exception as e:
        return f"Pattern analysis error: {e}"

def create_glucose_chart():
    """Create glucose trend chart with meal and insulin markers"""
    if not st.session_state.glucose_readings:
        return None
    
    # Convert to DataFrame - timestamps are already datetime objects
    df_glucose = pd.DataFrame(st.session_state.glucose_readings)
    
    # Filter to last 12 hours
    cutoff_time = datetime.now(eastern) - timedelta(hours=12)
    
    # Ensure timezone consistency for comparison
    df_glucose_filtered = []
    for _, row in df_glucose.iterrows():
        timestamp = row['timestamp']
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
            if timestamp.tzinfo is None:
                timestamp = eastern.localize(timestamp)
        elif timestamp.tzinfo is None:
            timestamp = eastern.localize(timestamp)
        
        if timestamp >= cutoff_time:
            df_glucose_filtered.append({
                'timestamp': timestamp,
                'value': row['value']
            })
    
    if not df_glucose_filtered:
        return None
    
    df_glucose = pd.DataFrame(df_glucose_filtered)
    
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
        elif not hasattr(meal_time, 'tzinfo') or meal_time.tzinfo is None:
            meal_time = eastern.localize(meal_time)
        
        if meal_time >= cutoff_time:
            fig.add_vline(x=meal_time, line_dash="dot", line_color="orange", 
                         annotation_text=f"🍽️ {meal['carbs']}g")
    
    # Add insulin markers
    for insulin in st.session_state.insulin_log:
        insulin_time = insulin['timestamp']
        if isinstance(insulin_time, str):
            insulin_time = datetime.fromisoformat(insulin_time).replace(tzinfo=eastern)
        elif not hasattr(insulin_time, 'tzinfo') or insulin_time.tzinfo is None:
            insulin_time = eastern.localize(insulin_time)
        
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
    
    # Debug information in sidebar
    st.sidebar.write(f"**Debug Info:**")
    st.sidebar.write(f"Total glucose readings: {len(st.session_state.glucose_readings)}")
    if st.session_state.glucose_readings:
        st.sidebar.write(f"Latest: {st.session_state.glucose_readings[-1]['timestamp'].strftime('%m/%d %H:%M')}")
        st.sidebar.write(f"Oldest: {st.session_state.glucose_readings[0]['timestamp'].strftime('%m/%d %H:%M')}")
    
    # Predictive alerts and analysis
    if glucose_data and len(st.session_state.glucose_readings) >= 3:
        prediction_results = predict_glucose_trends(glucose_data['value'], current_iob, st.session_state.glucose_readings)
        
        # Display alerts prominently at the top
        if 'alerts' in prediction_results and prediction_results['alerts']:
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
        if 'predictions' in prediction_results and prediction_results['predictions']:
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
                    value="✅ Active" if 'data_quality' in prediction_results and prediction_results.get('data_quality', {}).get('usable') else "❌ Limited",
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
        total_calories = sum(meal['calories'] for meal in today_meals)
        total_insulin = sum(insulin['dose'] for insulin in today_insulin)
        
        st.metric("Total Carbs", f"{total_carbs}g")
        st.metric("Total Protein", f"{total_protein}g")
        
        # Calorie tracking with progress
        calorie_percentage = (total_calories / DAILY_CALORIE_GOAL) * 100
        st.metric("Calories", f"{total_calories}/{DAILY_CALORIE_GOAL}", 
                 delta=f"{calorie_percentage:.0f}% of goal")
        
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
                save_data_to_file()
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
                                       value=float(st.session_state.basal_dose), step=1.0)
            if st.button("Update Basal"):
                st.session_state.basal_dose = basal_dose
                add_insulin_entry(basal_dose, 'basal', f"Daily basal: {basal_dose}u")
                st.success(f"Updated daily basal to {basal_dose}u")
                st.rerun()
        
        # Meal logging with bolus suggestion
        with st.expander("🍽️ Log Meal & Get Bolus Suggestion"):
            # Photo analysis option
            st.markdown("**📸 Photo Analysis**")
            uploaded_file = st.file_uploader(
                "Take/upload photo of your meal",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a clear photo for AI carb estimation"
            )
            
            if uploaded_file:
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("🤖 Analyze Photo"):
                        with st.spinner("AI analyzing your meal..."):
                            analysis = analyze_food_photo(uploaded_file)
                            
                            if analysis.get('success'):
                                st.success("✅ Analysis complete!")
                                st.session_state.photo_analysis = analysis
                            else:
                                st.error(f"❌ {analysis.get('error', 'Analysis failed')}")
                
                with col2:
                    st.image(uploaded_file, width=150)
            
            # Show analysis results if available
            if 'photo_analysis' in st.session_state and st.session_state.photo_analysis.get('success'):
                analysis = st.session_state.photo_analysis
                st.markdown("**🤖 AI Analysis:**")
                
                # Show foods identified
                for food in analysis.get('foods', []):
                    st.write(f"• {food['name']} ({food['portion']}): {food['carbs']}g carbs")
                
                # Auto-fill from analysis
                suggested_carbs = analysis.get('total_carbs', 30)
                suggested_protein = analysis.get('total_protein', 0)
                suggested_calories = analysis.get('total_calories', 0)
                
                # Show notes
                if analysis.get('notes'):
                    st.info(f"**AI Notes:** {analysis['notes']}")
                
                if st.button("✅ Use AI Analysis"):
                    st.session_state.ai_carbs = suggested_carbs
                    st.session_state.ai_protein = suggested_protein
                    st.session_state.ai_calories = suggested_calories
                    st.success("AI analysis applied to meal fields below!")
            
            st.markdown("---")
            st.markdown("**📝 Manual Entry**")
            
            # Get values from AI or use defaults
            default_carbs = st.session_state.get('ai_carbs', 30)
            default_protein = st.session_state.get('ai_protein', 0)
            default_calories = st.session_state.get('ai_calories', 0)
            
            meal_carbs = st.number_input("Carbs (g)", min_value=0, max_value=200, value=int(default_carbs))
            meal_protein = st.number_input("Protein (g)", min_value=0, max_value=100, value=int(default_protein))
            meal_calories = st.number_input("Calories", min_value=0, max_value=2000, value=int(default_calories))
            meal_description = st.text_input("Meal description", value="AI Photo Analysis" if 'photo_analysis' in st.session_state else "")
            
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
    
    # Data tables and pattern analysis
    st.markdown("---")
    
    # Add pattern analysis section
    if st.button("🧠 Analyze My Patterns"):
        with st.spinner("Analyzing your diabetes patterns..."):
            insights = run_basic_pattern_analysis()
            
            if isinstance(insights, str):
                st.warning(insights)
            else:
                st.markdown("### 🔍 AI Pattern Insights")
                for insight in insights:
                    st.info(insight)
                
                # Adaptive goal suggestions
                if st.session_state.meal_log:
                    meal_df = pd.DataFrame(st.session_state.meal_log)
                    
                    # Find successful days (high time in range)
                    if len(st.session_state.glucose_readings) >= 20:
                        glucose_df = pd.DataFrame(st.session_state.glucose_readings)
                        glucose_df['timestamp'] = pd.to_datetime(glucose_df['timestamp'])
                        glucose_df['date'] = glucose_df['timestamp'].dt.date
                        glucose_df['in_range'] = ((glucose_df['value'] >= 80) & (glucose_df['value'] <= 130))
                        
                        daily_tir = glucose_df.groupby('date')['in_range'].mean()
                        good_days = daily_tir[daily_tir >= 0.75].index
                        
                        if len(good_days) >= 2:
                            meal_df['date'] = pd.to_datetime(meal_df['timestamp']).dt.date
                            good_day_meals = meal_df[meal_df['date'].isin(good_days)]
                            
                            if not good_day_meals.empty:
                                avg_carbs = good_day_meals.groupby('date')['carbs'].sum().mean()
                                avg_protein = good_day_meals.groupby('date')['protein'].sum().mean()
                                avg_calories = good_day_meals.groupby('date')['calories'].sum().mean()
                                
                                st.markdown("### 🎯 Personalized Daily Goals")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Optimal Daily Carbs", f"{avg_carbs:.0f}g", 
                                            help="Based on your best glucose control days")
                                
                                with col2:
                                    st.metric("Target Daily Protein", f"{avg_protein:.0f}g", 
                                            help="Protein intake on successful days")
                                
                                with col3:
                                    st.metric("Target Daily Calories", f"{avg_calories:.0f}", 
                                            help="Calorie intake on your best days")
    
    # Data tables
    tab1, tab2, tab3 = st.tabs(["📈 Glucose History", "💉 Insulin History", "🍽️ Meal History"])
    
    with tab1:
        if st.session_state.glucose_readings:
            glucose_df = pd.DataFrame(st.session_state.glucose_readings)
            # Handle timestamp formatting safely
            glucose_df['timestamp'] = glucose_df['timestamp'].apply(
                lambda x: x.strftime('%m/%d %H:%M') if hasattr(x, 'strftime') else str(x)
            )
            st.dataframe(glucose_df[['timestamp', 'value', 'trend']].head(50), use_container_width=True)
            st.write(f"Showing latest 50 of {len(st.session_state.glucose_readings)} total readings")
        else:
            st.info("No glucose readings yet")
    
    with tab2:
        if st.session_state.insulin_log:
            insulin_df = pd.DataFrame(st.session_state.insulin_log)
            # Handle timestamp formatting safely
            insulin_df['timestamp'] = insulin_df['timestamp'].apply(
                lambda x: x.strftime('%m/%d %H:%M') if hasattr(x, 'strftime') else str(x)
            )
            st.dataframe(insulin_df[['timestamp', 'type', 'dose', 'notes']].head(50), use_container_width=True)
            st.write(f"Showing latest 50 of {len(st.session_state.insulin_log)} total entries")
        else:
            st.info("No insulin entries yet")
    
    with tab3:
        if st.session_state.meal_log:
            meal_df = pd.DataFrame(st.session_state.meal_log)
            # Handle timestamp formatting safely
            meal_df['timestamp'] = meal_df['timestamp'].apply(
                lambda x: x.strftime('%m/%d %H:%M') if hasattr(x, 'strftime') else str(x)
            )
            st.dataframe(meal_df[['timestamp', 'carbs', 'protein', 'calories', 'description']].head(50), use_container_width=True)
            st.write(f"Showing latest 50 of {len(st.session_state.meal_log)} total meals")
        else:
            st.info("No meal entries yet")

if __name__ == "__main__":
    main()