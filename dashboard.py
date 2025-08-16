import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
from pydexcom import Dexcom
import json
import os
import io
import anthropic
import base64
from PIL import Image

def delete_entry(entry_type, index):
    """Delete an entry from the specified log"""
    if entry_type == 'glucose':
        if 0 <= index < len(st.session_state.glucose_readings):
            deleted_entry = st.session_state.glucose_readings.pop(index)
            st.success(f"Deleted glucose reading: {deleted_entry['value']} mg/dL at {deleted_entry['timestamp'].strftime('%I:%M %p')}")
    elif entry_type == 'insulin':
        if 0 <= index < len(st.session_state.insulin_log):
            deleted_entry = st.session_state.insulin_log.pop(index)
            st.success(f"Deleted insulin: {deleted_entry['dose']} units ({deleted_entry['type']}) at {deleted_entry['timestamp'].strftime('%I:%M %p')}")
    elif entry_type == 'meal':
        if 0 <= index < len(st.session_state.meal_log):
            deleted_entry = st.session_state.meal_log.pop(index)
            st.success(f"Deleted meal: {deleted_entry.get('description', 'Unknown')} at {deleted_entry['timestamp'].strftime('%I:%M %p')}")
    elif entry_type == 'exercise':
        if 0 <= index < len(st.session_state.exercise_log):
            deleted_entry = st.session_state.exercise_log.pop(index)
            st.success(f"Deleted exercise: {deleted_entry['type']} at {deleted_entry['timestamp'].strftime('%I:%M %p')}")
    
    # Save data after deletion
    save_data_to_file()
    # Force page refresh
    st.rerun()
    
# Page configuration
st.set_page_config(
    page_title="Bean's Bolus Brain 🧠",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'glucose_readings' not in st.session_state:
    st.session_state.glucose_readings = []
if 'insulin_log' not in st.session_state:
    st.session_state.insulin_log = []
if 'meal_log' not in st.session_state:
    st.session_state.meal_log = []
if 'exercise_log' not in st.session_state:
    st.session_state.exercise_log = []
if 'basal_dose' not in st.session_state:
    st.session_state.basal_dose = 19

# Constants
CARB_RATIO = 12
CORRECTION_FACTOR = 50
TARGET_GLUCOSE = 115
GLUCOSE_RANGE = (80, 130)
MAX_BOLUS = 20
IOB_DURATION_HOURS = 4
DAILY_CALORIE_GOAL = 1200
eastern = pytz.timezone('US/Eastern')

def save_data_to_file():
    """Save session data to local files"""
    try:
        data = {
            'glucose_readings': [],
            'insulin_log': [],
            'meal_log': [],
            'exercise_log': []
        }
        
        for reading in st.session_state.glucose_readings:
            data['glucose_readings'].append({
                'value': reading['value'],
                'trend': reading['trend'],
                'trend_arrow': reading['trend_arrow'],
                'timestamp': reading['timestamp'].isoformat()
            })
        
        for entry in st.session_state.insulin_log:
            data['insulin_log'].append({
                'timestamp': entry['timestamp'].isoformat(),
                'type': entry['type'],
                'dose': entry['dose'],
                'notes': entry['notes']
            })
        
        for entry in st.session_state.meal_log:
            data['meal_log'].append({
                'timestamp': entry['timestamp'].isoformat(),
                'carbs': entry['carbs'],
                'protein': entry['protein'],
                'calories': entry['calories'],
                'description': entry['description']
            })
        
        for entry in st.session_state.exercise_log:
            data['exercise_log'].append({
                'timestamp': entry['timestamp'].isoformat(),
                'type': entry['type'],
                'duration': entry['duration'],
                'intensity': entry['intensity'],
                'notes': entry['notes']
            })
        
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
            
            st.session_state.exercise_log = []
            for entry in data.get('exercise_log', []):
                timestamp = datetime.fromisoformat(entry['timestamp'])
                if timestamp.tzinfo is None:
                    timestamp = eastern.localize(timestamp)
                st.session_state.exercise_log.append({
                    'timestamp': timestamp,
                    'type': entry['type'],
                    'duration': entry['duration'],
                    'intensity': entry['intensity'],
                    'notes': entry['notes']
                })
                
    except Exception as e:
        st.error(f"Error loading data: {e}")

# Load data on startup
if 'data_loaded' not in st.session_state:
    load_data_from_file()
    st.session_state.data_loaded = True

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
            
            should_add = True
            if st.session_state.glucose_readings:
                last_reading = st.session_state.glucose_readings[-1]
                time_diff = abs((glucose_data['timestamp'] - last_reading['timestamp']).total_seconds())
                if time_diff < 60:
                    should_add = False

            if should_add:
                st.session_state.glucose_readings.append(glucose_data)
                if len(st.session_state.glucose_readings) > 200:
                    st.session_state.glucose_readings = st.session_state.glucose_readings[-200:]
                save_data_to_file()
                
            return glucose_data
        return None
    except Exception as e:
        st.error(f"Dexcom connection error: {e}")
        return None

def calculate_iob():
    """Calculate current insulin on board"""
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

def analyze_food_photo(image_file):
    """Analyze food photo using Claude AI"""
    try:
        # Get API key from Streamlit config
        try:
            api_key = st.secrets["claude"]["api_key"]
        except:
            # Fallback to old format if new format doesn't work
            api_key = st.secrets.get("CLAUDE_API_KEY")
        
        if not api_key:
            return {"error": "Claude AI API key not found", "success": False}
        
        # Initialize Claude client
        client = anthropic.Anthropic(api_key=api_key)
        
        # Process image
        if hasattr(image_file, 'read'):
            image_data = image_file.read()
            image_file.seek(0)  # Reset file pointer for other uses
        else:
            image_data = image_file
        
        # Convert to PIL Image and resize if needed
        img = Image.open(io.BytesIO(image_data))
        
        # Resize image if too large (Claude has size limits)
        max_size = 1024
        if img.width > max_size or img.height > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert back to bytes
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Encode as base64
        image_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
        
        # Create Claude AI prompt for food analysis
        prompt = """Please analyze this food image for diabetes management. I need accurate carbohydrate estimates for insulin dosing.

For each food item visible, estimate:
1. Food name and description
2. Portion size
3. Carbohydrates in grams (be conservative for safety)
4. Protein in grams
5. Calories

Respond in this exact JSON format:
{
    "foods": [
        {
            "name": "Food name",
            "portion": "portion description",
            "carbs": number,
            "protein": number,
            "calories": number
        }
    ],
    "total_carbs": sum_of_all_carbs,
    "total_protein": sum_of_all_protein,
    "total_calories": sum_of_all_calories,
    "confidence": "high/medium/low",
    "notes": "Additional observations or uncertainties"
}

Important: Be conservative with carb estimates for diabetes safety. If uncertain, err on the higher side for carbs."""
        
        # Call Claude AI API
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_base64
                            }
                        }
                    ]
                }
            ]
        )
        
        # Parse Claude's response
        analysis_text = response.content[0].text.strip()
        
        # Try to extract JSON from response
        try:
            import json
            
            # Look for JSON in the response
            start_idx = analysis_text.find('{')
            end_idx = analysis_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = analysis_text[start_idx:end_idx]
                analysis = json.loads(json_str)
                
                # Ensure required fields exist
                if 'foods' not in analysis:
                    analysis['foods'] = []
                if 'total_carbs' not in analysis:
                    analysis['total_carbs'] = sum(food.get('carbs', 0) for food in analysis['foods'])
                if 'total_protein' not in analysis:
                    analysis['total_protein'] = sum(food.get('protein', 0) for food in analysis['foods'])
                if 'total_calories' not in analysis:
                    analysis['total_calories'] = sum(food.get('calories', 0) for food in analysis['foods'])
                
                analysis['success'] = True
                analysis['raw_response'] = analysis_text
                analysis['notes'] = f"🤖 Claude AI Analysis: {analysis.get('notes', 'Analysis completed')}"
                
                return analysis
            else:
                raise json.JSONDecodeError("No valid JSON found", analysis_text, 0)
                
        except json.JSONDecodeError:
            # If JSON parsing fails, create a structured response from the text
            return {
                "foods": [{
                    "name": "Claude AI Analysis (see notes)", 
                    "portion": "Please review", 
                    "carbs": 35, 
                    "protein": 10, 
                    "calories": 250
                }],
                "total_carbs": 35,
                "total_protein": 10,
                "total_calories": 250,
                "confidence": "medium",
                "notes": f"🤖 Claude Response: {analysis_text}",
                "success": True,
                "raw_response": analysis_text
            }
            
    except anthropic.APIError as e:
        return {
            "error": f"Claude AI API error: {str(e)}",
            "success": False
        }
    except Exception as e:
        return {
            "error": f"Analysis failed: {str(e)}",
            "success": False
        }

def run_insulin_sensitivity_analysis():
    """Comprehensive insulin sensitivity analysis to optimize ratios"""
    if len(st.session_state.glucose_readings) < 20:
        return "Need at least 20 glucose readings for insulin sensitivity analysis"
    
    if len(st.session_state.meal_log) < 5:
        return "Need at least 5 meal entries for insulin sensitivity analysis"
    
    try:
        insights = []
        recommendations = []
        
        # Convert data to DataFrames with proper datetime handling
        glucose_data = []
        for reading in st.session_state.glucose_readings:
            timestamp = reading['timestamp']
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            glucose_data.append({
                'timestamp': timestamp,
                'value': reading['value']
            })
        glucose_df = pd.DataFrame(glucose_data)
        glucose_df['timestamp'] = pd.to_datetime(glucose_df['timestamp'])
        
        meal_data = []
        for meal in st.session_state.meal_log:
            timestamp = meal['timestamp']
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp)
            meal_data.append({
                'timestamp': timestamp,
                'carbs': meal['carbs']
            })
        meal_df = pd.DataFrame(meal_data)
        meal_df['timestamp'] = pd.to_datetime(meal_df['timestamp'])
        
        insulin_data = []
        for insulin in st.session_state.insulin_log:
            if insulin['type'] == 'bolus':
                timestamp = insulin['timestamp']
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                insulin_data.append({
                    'timestamp': timestamp,
                    'dose': insulin['dose']
                })
        
        if not insulin_data:
            return "Need bolus insulin entries for analysis"
            
        insulin_df = pd.DataFrame(insulin_data)
        insulin_df['timestamp'] = pd.to_datetime(insulin_df['timestamp'])
        
        # Analysis: Post-meal glucose patterns
        post_meal_lows = 0
        post_meal_highs = 0
        total_meals_analyzed = 0
        carb_ratio_effectiveness = []
        
        for _, meal in meal_df.iterrows():
            meal_time = meal['timestamp']
            meal_carbs = meal['carbs']
            
            # Find glucose readings 1-3 hours after meal
            post_meal_start = meal_time + timedelta(hours=1)
            post_meal_end = meal_time + timedelta(hours=3)
            
            post_meal_glucose = glucose_df[
                (glucose_df['timestamp'] >= post_meal_start) & 
                (glucose_df['timestamp'] <= post_meal_end)
            ]
            
            if len(post_meal_glucose) >= 2:
                total_meals_analyzed += 1
                max_post_meal = post_meal_glucose['value'].max()
                min_post_meal = post_meal_glucose['value'].min()
                
                # Check for post-meal lows
                if min_post_meal < 70:
                    post_meal_lows += 1
                elif min_post_meal < 80:
                    post_meal_lows += 0.5
                
                # Check for post-meal highs
                if max_post_meal > 180:
                    post_meal_highs += 1
                
                # Find corresponding bolus dose
                bolus_window_start = meal_time - timedelta(minutes=30)
                bolus_window_end = meal_time + timedelta(minutes=30)
                
                meal_bolus = insulin_df[
                    (insulin_df['timestamp'] >= bolus_window_start) & 
                    (insulin_df['timestamp'] <= bolus_window_end)
                ]
                
                if not meal_bolus.empty and meal_carbs > 0:
                    total_bolus = meal_bolus['dose'].sum()
                    if total_bolus > 0:
                        actual_ratio = meal_carbs / total_bolus
                        carb_ratio_effectiveness.append({
                            'carbs': meal_carbs,
                            'bolus': total_bolus,
                            'ratio': actual_ratio,
                            'max_glucose': max_post_meal,
                            'min_glucose': min_post_meal,
                            'went_low': min_post_meal < 80,
                            'went_high': max_post_meal > 180
                        })
        
        # Generate insights
        if total_meals_analyzed > 0:
            low_percentage = (post_meal_lows / total_meals_analyzed) * 100
            high_percentage = (post_meal_highs / total_meals_analyzed) * 100
            
            insights.append(f"📊 Analyzed {total_meals_analyzed} meals with post-meal glucose data")
            insights.append(f"⚠️ Post-meal lows: {low_percentage:.0f}% of meals (Target: <10%)")
            insights.append(f"📈 Post-meal highs >180: {high_percentage:.0f}% of meals (Target: <20%)")
            
            # Carb ratio effectiveness analysis
            if len(carb_ratio_effectiveness) >= 3:
                ratio_df = pd.DataFrame(carb_ratio_effectiveness)
                
                low_meals = ratio_df[ratio_df['went_low'] == True]
                good_meals = ratio_df[(ratio_df['went_low'] == False) & (ratio_df['went_high'] == False)]
                
                if len(low_meals) > 0:
                    avg_ratio_low_meals = low_meals['ratio'].mean()
                    insights.append(f"🔍 Average carb ratio when going low: 1:{avg_ratio_low_meals:.0f}")
                
                if len(good_meals) > 0:
                    avg_ratio_good_meals = good_meals['ratio'].mean()
                    insights.append(f"✅ Average carb ratio for good control: 1:{avg_ratio_good_meals:.0f}")
                
                # Generate recommendations
                if low_percentage > 30:
                    if len(good_meals) > 0:
                        recommended_ratio = good_meals['ratio'].mean()
                    else:
                        recommended_ratio = 15
                    
                    recommendations.append({
                        'type': 'CRITICAL',
                        'title': 'Carb Ratio Too Aggressive',
                        'message': f'You\'re going low after {low_percentage:.0f}% of meals',
                        'action': f'Consider changing carb ratio from 1:12 to 1:{recommended_ratio:.0f}',
                        'reasoning': 'Frequent post-meal lows indicate too much insulin per carb'
                    })
                
                elif low_percentage > 15:
                    recommendations.append({
                        'type': 'WARNING',
                        'title': 'Frequent Post-Meal Lows',
                        'message': f'Going low after {low_percentage:.0f}% of meals',
                        'action': 'Consider small carb ratio adjustment from 1:12 to 1:13 or 1:14',
                        'reasoning': 'Reduce insulin slightly to prevent lows while maintaining control'
                    })
        
        return {
            'insights': insights,
            'recommendations': recommendations,
            'total_meals_analyzed': total_meals_analyzed
        }
        
    except Exception as e:
        return f"Analysis error: {str(e)}"

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

def main():
    st.title("🧠 Bean's Bolus Brain")
    st.subheader("AI-Powered Diabetes Management Dashboard")
    
    # Add refresh button at the top
    if st.button("🔄 Refresh Dashboard", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    # Get current data
    glucose_data = get_dexcom_data()
    current_iob = calculate_iob()
    
    # Main dashboard layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Current glucose status
        st.markdown("### 🩸 Current Glucose")
        display_glucose_status(glucose_data)
        
        # IOB display
        st.markdown("### 💉 Insulin on Board")
        iob_color = "orange" if current_iob > 3 else "green"
        st.markdown(f"<h2 style='text-align: center; color: {iob_color};'>{current_iob:.1f} units</h2>", 
                   unsafe_allow_html=True)
        
        # Enhanced glucose chart with meal/insulin markers
        if st.session_state.glucose_readings:
            cutoff_time = datetime.now(eastern) - timedelta(hours=12)
            
            # Filter and ensure proper datetime objects
            recent_readings = []
            for r in st.session_state.glucose_readings:
                timestamp = r['timestamp']
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp)
                    if timestamp.tzinfo is None:
                        timestamp = eastern.localize(timestamp)
                elif timestamp.tzinfo is None:
                    timestamp = eastern.localize(timestamp)
                
                if timestamp >= cutoff_time:
                    recent_readings.append({
                        'timestamp': timestamp,
                        'value': r['value']
                    })
            
            if recent_readings:
                df = pd.DataFrame(recent_readings)
                fig = go.Figure()
                
                # Glucose line
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df['value'],
                    mode='lines+markers',
                    name='Glucose',
                    line=dict(color='blue', width=3)
                ))
                
                # Target range
                fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Target Range")
                fig.add_hline(y=130, line_dash="dash", line_color="green")
                
                # Add meal markers with proper datetime handling
                try:
                    for meal in st.session_state.meal_log:
                        meal_time = meal['timestamp']
                        if isinstance(meal_time, str):
                            meal_time = datetime.fromisoformat(meal_time)
                            if meal_time.tzinfo is None:
                                meal_time = eastern.localize(meal_time)
                        elif meal_time.tzinfo is None:
                            meal_time = eastern.localize(meal_time)
                        
                        if meal_time >= cutoff_time:
                            carbs_value = int(float(meal.get('carbs', 0)))
                            fig.add_vline(x=meal_time, line_dash="dot", line_color="orange",
                                         annotation_text=f"🍽️ {carbs_value}g")
                except Exception as e:
                    pass  # Skip meal markers if there are issues
                
                # Add insulin markers with proper datetime handling
                try:
                    for insulin in st.session_state.insulin_log:
                        insulin_time = insulin['timestamp']
                        if isinstance(insulin_time, str):
                            insulin_time = datetime.fromisoformat(insulin_time)
                            if insulin_time.tzinfo is None:
                                insulin_time = eastern.localize(insulin_time)
                        elif insulin_time.tzinfo is None:
                            insulin_time = eastern.localize(insulin_time)
                        
                        if insulin_time >= cutoff_time and insulin['type'] == 'bolus':
                            dose_value = float(insulin.get('dose', 0))
                            fig.add_vline(x=insulin_time, line_dash="dot", line_color="purple",
                                         annotation_text=f"💉 {dose_value}u")
                except Exception as e:
                    pass  # Skip insulin markers if there are issues
                
                fig.update_layout(
                    title="12-Hour Glucose Trend with Meals & Insulin",
                    xaxis_title="Time",
                    yaxis_title="Glucose (mg/dL)",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Enhanced today's summary with A1C
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
        
        # Calorie tracking with progress
        calorie_percentage = (total_calories / DAILY_CALORIE_GOAL) * 100
        st.metric("Calories", f"{total_calories}/{DAILY_CALORIE_GOAL}", 
                 delta=f"{calorie_percentage:.0f}% of goal")
        
        st.metric("Total Bolus", f"{total_insulin:.1f}u")
        
        # Time in range calculation
        if st.session_state.glucose_readings:
            today_glucose = [g for g in st.session_state.glucose_readings 
                           if g['timestamp'].date() == today]
            if today_glucose:
                in_range = sum(1 for g in today_glucose 
                             if GLUCOSE_RANGE[0] <= g['value'] <= GLUCOSE_RANGE[1])
                time_in_range = (in_range / len(today_glucose)) * 100
                st.metric("Time in Range", f"{time_in_range:.0f}%")
            
            # Estimated A1C calculation
            if len(st.session_state.glucose_readings) >= 5:
                all_glucose_values = [reading['value'] for reading in st.session_state.glucose_readings]
                avg_glucose = sum(all_glucose_values) / len(all_glucose_values)
                estimated_a1c = (avg_glucose + 46.7) / 28.7
                
                st.metric("Estimated A1C", f"{estimated_a1c:.1f}%", 
                         help=f"Based on {len(all_glucose_values)} glucose readings")
        else:
            st.info("Need glucose readings to calculate A1C")
    
    # Sidebar for logging (defaults to current date/time)
    with st.sidebar:
        st.header("📝 Quick Logging")
        
        # Refresh button in sidebar
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        
        # Current date/time as defaults
        now = datetime.now(eastern)
        current_date = now.date()
        current_time = now.time().replace(second=0, microsecond=0)
        
        # Manual glucose entry
        with st.expander("🩸 Manual Glucose Entry"):
            col1, col2 = st.columns([2, 1])
            with col1:
                manual_glucose = st.number_input("Glucose (mg/dL)", min_value=40, max_value=400, value=120)
            with col2:
                glucose_date = st.date_input("Date", value=current_date, key="glucose_date")
                glucose_time = st.time_input("Time", value=current_time, key="glucose_time")
            
            if st.button("Log Glucose"):
                glucose_datetime = datetime.combine(glucose_date, glucose_time)
                glucose_datetime = eastern.localize(glucose_datetime)
                
                glucose_entry = {
                    'value': manual_glucose,
                    'trend': 'Manual Entry',
                    'trend_arrow': '',
                    'timestamp': glucose_datetime
                }
                st.session_state.glucose_readings.append(glucose_entry)
                st.session_state.glucose_readings.sort(key=lambda x: x['timestamp'])
                save_data_to_file()
                st.success(f"Logged {manual_glucose} mg/dL")
                st.rerun()
        
        # Bolus logging
        with st.expander("💉 Log Bolus"):
            col1, col2 = st.columns([2, 1])
            with col1:
                bolus_dose = st.number_input("Bolus dose (units)", min_value=0.0, max_value=20.0, step=0.5)
                bolus_notes = st.text_input("Notes (optional)")
            with col2:
                bolus_date = st.date_input("Date", value=current_date, key="bolus_date")
                bolus_time = st.time_input("Time", value=current_time, key="bolus_time")
            
            if st.button("Log Bolus"):
                bolus_datetime = datetime.combine(bolus_date, bolus_time)
                bolus_datetime = eastern.localize(bolus_datetime)
                
                entry = {
                    'timestamp': bolus_datetime,
                    'type': 'bolus',
                    'dose': bolus_dose,
                    'notes': bolus_notes
                }
                st.session_state.insulin_log.append(entry)
                st.session_state.insulin_log.sort(key=lambda x: x['timestamp'])
                save_data_to_file()
                st.success(f"Logged {bolus_dose}u bolus")
                st.rerun()
        
        # Basal logging
        with st.expander("🕐 Log Basal"):
            col1, col2 = st.columns([2, 1])
            with col1:
                basal_dose = st.number_input("Basal dose (units)", min_value=0.0, max_value=30.0, step=0.5, value=float(st.session_state.basal_dose))
                basal_notes = st.text_input("Basal notes (optional)", key="basal_notes_input")
            with col2:
                basal_date = st.date_input("Date", value=current_date, key="basal_date")
                basal_time = st.time_input("Time", value=current_time, key="basal_time")
            
            if st.button("Log Basal"):
                basal_datetime = datetime.combine(basal_date, basal_time)
                basal_datetime = eastern.localize(basal_datetime)
                
                entry = {
                    'timestamp': basal_datetime,
                    'type': 'basal',
                    'dose': basal_dose,
                    'notes': basal_notes
                }
                st.session_state.insulin_log.append(entry)
                st.session_state.insulin_log.sort(key=lambda x: x['timestamp'])
                st.session_state.basal_dose = basal_dose  # Remember for next time
                save_data_to_file()
                st.success(f"Logged {basal_dose}u basal")
                st.rerun()

        # Meal logging with enhanced bolus calculator
        with st.expander("🍽️ Log Meal & Get Bolus Suggestion"):
            st.markdown("**📸 Photo Analysis with Claude AI**")
            
            uploaded_file = st.file_uploader("Upload meal photo", type=['png', 'jpg', 'jpeg'])
            
            if uploaded_file:
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("🤖 Analyze Photo"):
                        with st.spinner("Processing..."):
                            analysis = analyze_food_photo(uploaded_file)
                            if analysis.get('success'):
                                st.success("✅ Analysis complete!")
                                st.session_state.photo_analysis = analysis
                                
                                # Display Claude's analysis
                                if 'foods' in analysis:
                                    st.write("**🤖 Claude AI found:**")
                                    for food in analysis['foods']:
                                        st.write(f"• {food['name']}: {food['carbs']}g carbs, {food['protein']}g protein")
                                    st.write(f"**Total: {analysis['total_carbs']}g carbs, {analysis['total_protein']}g protein**")
                                    if 'confidence' in analysis:
                                        st.write(f"**Confidence:** {analysis['confidence']}")
                            else:
                                st.error(f"❌ {analysis.get('error', 'Analysis failed')}")
                
                with col2:
                    st.image(uploaded_file, width=150)
            
            # Manual entry with defaults from Claude AI analysis
            st.markdown("**📝 Manual Entry**")
            default_carbs = st.session_state.get('photo_analysis', {}).get('total_carbs', 30)
            default_protein = st.session_state.get('photo_analysis', {}).get('total_protein', 0)
            default_calories = st.session_state.get('photo_analysis', {}).get('total_calories', 0)
            
            col1, col2 = st.columns([2, 1])
            with col1:
                meal_carbs = st.number_input("Carbs (g)", min_value=0, max_value=200, value=int(default_carbs))
                meal_protein = st.number_input("Protein (g)", min_value=0, max_value=100, value=int(default_protein))
                meal_calories = st.number_input("Calories", min_value=0, max_value=2000, value=int(default_calories))
                meal_description = st.text_input("Meal description")
            
            with col2:
                meal_date = st.date_input("Date", value=current_date, key="meal_date")
                meal_time = st.time_input("Time", value=current_time, key="meal_time")
            
            # Enhanced bolus calculator
            if glucose_data:
                bolus_suggestion = calculate_bolus_suggestion(meal_carbs, meal_protein, glucose_data['value'], current_iob)
                
                st.markdown("**💊 Detailed Bolus Calculation:**")
                st.write(f"• Carb bolus: {bolus_suggestion['carb_bolus']}u ({meal_carbs}g ÷ {CARB_RATIO})")
                if bolus_suggestion['protein_bolus'] > 0:
                    st.write(f"• Protein bolus: {bolus_suggestion['protein_bolus']}u ({meal_protein}g × 10%)")
                if bolus_suggestion['correction_bolus'] > 0:
                    st.write(f"• Correction: {bolus_suggestion['correction_bolus']}u (glucose {glucose_data['value']} - IOB {current_iob:.1f})")
                st.markdown(f"**Total suggested: {bolus_suggestion['total_bolus']}u**")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Log Meal Only"):
                        meal_datetime = datetime.combine(meal_date, meal_time)
                        meal_datetime = eastern.localize(meal_datetime)
                        
                        entry = {
                            'timestamp': meal_datetime,
                            'carbs': meal_carbs,
                            'protein': meal_protein,
                            'calories': meal_calories,
                            'description': meal_description
                        }
                        st.session_state.meal_log.append(entry)
                        st.session_state.meal_log.sort(key=lambda x: x['timestamp'])
                        save_data_to_file()
                        st.success("Meal logged!")
                        st.rerun()
                
                with col2:
                    if st.button("Log Meal + Bolus"):
                        meal_datetime = datetime.combine(meal_date, meal_time)
                        meal_datetime = eastern.localize(meal_datetime)
                        
                        # Add meal
                        meal_entry = {
                            'timestamp': meal_datetime,
                            'carbs': meal_carbs,
                            'protein': meal_protein,
                            'calories': meal_calories,
                            'description': meal_description
                        }
                        st.session_state.meal_log.append(meal_entry)
                        
                        # Add bolus
                        bolus_entry = {
                            'timestamp': meal_datetime,
                            'type': 'bolus',
                            'dose': bolus_suggestion['total_bolus'],
                            'notes': f"Meal bolus: {meal_description}"
                        }
                        st.session_state.insulin_log.append(bolus_entry)
                        
                        # Sort both logs
                        st.session_state.meal_log.sort(key=lambda x: x['timestamp'])
                        st.session_state.insulin_log.sort(key=lambda x: x['timestamp'])
                        save_data_to_file()
                        st.success(f"Logged meal + {bolus_suggestion['total_bolus']}u bolus!")
                        st.rerun()
            else:
                st.info("Connect Dexcom for bolus suggestions")

        # Exercise logging section
        with st.expander("🏃‍♀️ Log Exercise"):
            # Exercise type selection
            exercise_type = st.selectbox(
                "Exercise Type:",
                [
                    "Cardio - Walking",
                    "Cardio - Running", 
                    "Cardio - Cycling",
                    "Cardio - Swimming",
                    "Strength Training - Upper Body",
                    "Strength Training - Lower Body", 
                    "Strength Training - Full Body",
                    "HIIT - High Intensity",
                    "HIIT - Tabata",
                    "Yoga - Gentle",
                    "Yoga - Power",
                    "Sports - Basketball",
                    "Sports - Tennis", 
                    "Sports - Soccer",
                    "Other"
                ],
                key="exercise_type_select"
            )
            
            # Duration and intensity
            col1, col2 = st.columns(2)
            with col1:
                duration = st.number_input(
                    "Duration (minutes):",
                    min_value=1,
                    max_value=240,
                    value=30,
                    step=5,
                    key="exercise_duration"
                )
            
            with col2:
                intensity = st.selectbox(
                    "Intensity:",
                    ["Light", "Moderate", "Vigorous", "Very High"],
                    index=1,  # Default to Moderate
                    key="exercise_intensity"
                )
            
            # Custom timestamp
            st.write("**Exercise Time:**")
            col1, col2 = st.columns(2)
            with col1:
                exercise_date = st.date_input(
                    "Date:",
                    value=current_date,
                    key="exercise_date"
                )
            with col2:
                exercise_time = st.time_input(
                    "Time:",
                    value=current_time,
                    key="exercise_time"
                )
            
            # Optional notes
            notes = st.text_area(
                "Notes (optional):",
                placeholder="How did you feel? Any glucose observations?",
                max_chars=200,
                key="exercise_notes"
            )
            
            # Predicted glucose effect
            if exercise_type.startswith("Cardio") or "HIIT" in exercise_type:
                st.info("⬇️ This exercise may lower your glucose 1-4 hours after completion")
                glucose_effect = "May lower glucose"
            elif "Strength" in exercise_type:
                st.info("⬆️ This exercise may temporarily raise your glucose")
                glucose_effect = "May raise glucose"
            else:
                st.info("📊 Monitor your glucose response to this activity")
                glucose_effect = "Variable effect"
            
            # Log exercise button
            if st.button("💪 Log Exercise", key="log_exercise_btn", use_container_width=True):
                # Create timestamp
                exercise_datetime = datetime.combine(exercise_date, exercise_time)
                exercise_timestamp = eastern.localize(exercise_datetime)
                
                # Create exercise entry
                exercise_entry = {
                    'timestamp': exercise_timestamp,
                    'type': exercise_type,
                    'duration': duration,
                    'intensity': intensity,
                    'notes': notes,
                    'predicted_effect': glucose_effect
                }
                
                # Add to log
                st.session_state.exercise_log.append(exercise_entry)
                
                # Sort by timestamp to maintain chronological order
                st.session_state.exercise_log.sort(key=lambda x: x['timestamp'])
                
                # Save data
                save_data_to_file()
                
                # Success message
                st.success(f"✅ Logged {exercise_type} for {duration} minutes!")
                
                # Show predicted effect
                if glucose_effect == "May lower glucose":
                    st.info("💡 Consider having a small snack if your glucose is <120 mg/dL")
                elif glucose_effect == "May raise glucose":
                    st.info("💡 Monitor for temporary glucose spike in next 1-2 hours")
                
                # Rerun to update the display
                st.rerun()
        
        # Enhanced correction calculator
        if glucose_data and glucose_data['value'] > 130:
            st.markdown("### 🎯 Correction Calculator")
            correction_bolus = max(0, (glucose_data['value'] - TARGET_GLUCOSE) / CORRECTION_FACTOR - current_iob)
            
            st.write(f"Current glucose: {glucose_data['value']} mg/dL")
            st.write(f"Target glucose: {TARGET_GLUCOSE} mg/dL")
            st.write(f"Correction needed: {(glucose_data['value'] - TARGET_GLUCOSE) / CORRECTION_FACTOR:.1f}u")
            st.write(f"Current IOB: {current_iob:.1f}u")
            
            if correction_bolus > 0.5:
                st.warning(f"**Suggested correction: {round(correction_bolus)}u**")
                if st.button("Log Correction"):
                    add_insulin_entry(round(correction_bolus), 'bolus', "Correction bolus")
                    st.success(f"Logged {round(correction_bolus)}u correction!")
                    st.rerun()
            else:
                st.info("No correction needed (IOB sufficient)")
        elif glucose_data:
            st.markdown("### 🎯 Correction Calculator")
            st.success(f"Glucose {glucose_data['value']} mg/dL - No correction needed")
    
    # Enhanced analysis section
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💉 Insulin Sensitivity Analysis"):
            with st.spinner("Analyzing your insulin effectiveness and ratios..."):
                analysis = run_insulin_sensitivity_analysis()
                
                if isinstance(analysis, str):
                    st.warning(analysis)
                else:
                    st.markdown("### 💉 Insulin Sensitivity Results")
                    
                    # Display insights
                    for insight in analysis['insights']:
                        st.info(insight)
                    
                    # Display recommendations with priority
                    if analysis['recommendations']:
                        st.markdown("### 🎯 Personalized Recommendations")
                        
                        for rec in analysis['recommendations']:
                            if rec['type'] == 'CRITICAL':
                                st.error(f"**🚨 {rec['title']}**")
                                st.error(f"**Issue:** {rec['message']}")
                                st.error(f"**Recommended Action:** {rec['action']}")
                                st.error(f"**Why:** {rec['reasoning']}")
                                st.markdown("---")
                            elif rec['type'] == 'WARNING':
                                st.warning(f"**⚠️ {rec['title']}**")
                                st.warning(f"**Issue:** {rec['message']}")
                                st.warning(f"**Recommended Action:** {rec['action']}")
                                st.info(f"**Why:** {rec['reasoning']}")
                                st.markdown("---")
                        
                        # Summary recommendation
                        critical_recs = [r for r in analysis['recommendations'] if r['type'] == 'CRITICAL']
                        if critical_recs:
                            st.markdown("### 🎯 **Priority Action Needed**")
                            st.error("**Your data shows frequent post-meal lows. Consider discussing a carb ratio adjustment with your healthcare provider.**")
                    else:
                        st.success("✅ Your insulin ratios appear to be working well based on available data!")
                        st.info("Continue monitoring and consider this analysis again as you collect more data.")
    
    with col2:
        if st.button("📊 Pattern Analysis"):
            with st.spinner("Analyzing diabetes patterns..."):
                if len(st.session_state.glucose_readings) < 10:
                    st.warning("Need more glucose data for pattern analysis")
                else:
                    st.markdown("### 📊 Pattern Analysis Results")
                    
                    # Time in range analysis - handle timezone properly
                    glucose_values = [reading['value'] for reading in st.session_state.glucose_readings]
                    in_range_count = sum(1 for value in glucose_values if 80 <= value <= 130)
                    in_range = (in_range_count / len(glucose_values)) * 100
                    st.info(f"📊 Your overall time in range is {in_range:.0f}% (goal: >70%)")
                    
                    # Best/worst times of day - fix timezone handling
                    try:
                        # Create clean data without timezone for pandas
                        hourly_data = {}
                        for reading in st.session_state.glucose_readings:
                            timestamp = reading['timestamp']
                            if isinstance(timestamp, str):
                                timestamp = datetime.fromisoformat(timestamp)
                            
                            # Extract hour without timezone conversion
                            hour = timestamp.hour
                            if hour not in hourly_data:
                                hourly_data[hour] = []
                            hourly_data[hour].append(reading['value'])
                        
                        # Calculate averages
                        hourly_averages = {hour: sum(values)/len(values) for hour, values in hourly_data.items()}
                        
                        if hourly_averages:
                            best_hour = min(hourly_averages.keys(), key=lambda h: abs(hourly_averages[h] - 115))
                            worst_hour = max(hourly_averages.keys(), key=lambda h: hourly_averages[h])
                            
                            st.info(f"🕐 Best glucose control: {best_hour}:00 (avg: {hourly_averages[best_hour]:.0f} mg/dL)")
                            st.info(f"⚠️ Highest average glucose: {worst_hour}:00 (avg: {hourly_averages[worst_hour]:.0f} mg/dL)")
                    except Exception as e:
                        st.info("📊 Time-of-day analysis needs more data points")
                    
                    # Meal analysis
                    if len(st.session_state.meal_log) >= 3:
                        total_carbs = sum(meal['carbs'] for meal in st.session_state.meal_log)
                        total_calories = sum(meal['calories'] for meal in st.session_state.meal_log)
                        avg_carbs = total_carbs / len(st.session_state.meal_log)
                        avg_calories = total_calories / len(st.session_state.meal_log)
                        
                        st.info(f"🍽️ Average meal: {avg_carbs:.0f}g carbs, {avg_calories:.0f} calories")
                        st.info(f"📊 Total meals logged: {len(st.session_state.meal_log)}")
                    else:
                        st.info("🍽️ Log more meals for detailed meal analysis")
    
    # Enhanced data tables with delete functionality
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Glucose History", "💉 Insulin History", "🍽️ Meal History", "🏃‍♀️ Exercise History"])
    
    with tab1:
        st.subheader("Recent Glucose Readings")
        if st.session_state.glucose_readings:
            # Show most recent first (reverse chronological)
            recent_readings = list(reversed(st.session_state.glucose_readings[-15:]))  # Last 15, newest first
            
            for i, reading in enumerate(recent_readings):
                # Calculate actual index in the original list
                actual_index = len(st.session_state.glucose_readings) - 1 - i
                
                col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
                
                with col1:
                    # Color code based on value
                    if reading['value'] < 70:
                        st.markdown(f"🔴 **{reading['value']} mg/dL**")
                    elif reading['value'] > 180:
                        st.markdown(f"🟠 **{reading['value']} mg/dL**")
                    else:
                        st.markdown(f"🟢 **{reading['value']} mg/dL**")
                
                with col2:
                    st.write(f"{reading['trend_arrow']} {reading.get('trend', '')}")
                
                with col3:
                    st.write(reading['timestamp'].strftime('%m/%d %I:%M %p'))
                
                with col4:
                    source = reading.get('source', 'Manual')
                    st.write(f"📱 {source}")
                
                with col5:
                    if st.button("🗑️", key=f"del_glucose_{actual_index}", help="Delete this reading"):
                        delete_entry('glucose', actual_index)
                
                st.divider()
        else:
            st.info("No glucose readings yet. Add one manually or wait for Dexcom data.")
    
    with tab2:
        st.subheader("Recent Insulin Doses")
        if st.session_state.insulin_log:
            # Show most recent first
            recent_insulin = list(reversed(st.session_state.insulin_log[-15:]))  # Last 15, newest first
            
            for i, dose in enumerate(recent_insulin):
                actual_index = len(st.session_state.insulin_log) - 1 - i
                
                col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
                
                with col1:
                    # Different icons for insulin types
                    icon = "💉" if dose['type'] == 'bolus' else "🕐"
                    st.markdown(f"{icon} **{dose['dose']} units**")
                
                with col2:
                    st.write(f"{dose['type']}")
                
                with col3:
                    st.write(dose['timestamp'].strftime('%m/%d %I:%M %p'))
                
                with col4:
                    # Show notes if any
                    notes = dose.get('notes', '')
                    if notes:
                        st.write(f"📝 {notes[:20]}...")
                    else:
                        st.write("—")
                
                with col5:
                    if st.button("🗑️", key=f"del_insulin_{actual_index}", help="Delete this dose"):
                        delete_entry('insulin', actual_index)
                
                st.divider()
        else:
            st.info("No insulin doses logged yet. Use the sidebar to log your first dose.")
    
    with tab3:
        st.subheader("Recent Meals")
        if st.session_state.meal_log:
            # Show most recent first
            recent_meals = list(reversed(st.session_state.meal_log[-15:]))  # Last 15, newest first
            
            for i, meal in enumerate(recent_meals):
                actual_index = len(st.session_state.meal_log) - 1 - i
                
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                
                with col1:
                    st.markdown(f"🍽️ **{meal.get('description', 'Unknown meal')}**")
                    # Show method if photo
                    if meal.get('method') == 'photo':
                        st.write("📸 Photo analyzed")
                
                with col2:
                    carbs = meal.get('carbs', 0)
                    protein = meal.get('protein', 0)
                    calories = meal.get('calories', 0)
                    st.write(f"🥖 {carbs}g carbs")
                    if protein > 0:
                        st.write(f"🥩 {protein}g protein")
                    if calories > 0:
                        st.write(f"🔥 {calories} cal")
                
                with col3:
                    st.write(meal['timestamp'].strftime('%m/%d %I:%M %p'))
                
                with col4:
                    if st.button("🗑️", key=f"del_meal_{actual_index}", help="Delete this meal"):
                        delete_entry('meal', actual_index)
                
                st.divider()
        else:
            st.info("No meals logged yet. Use the sidebar to log your first meal or take a photo.")
    
    with tab4:
        st.subheader("Recent Exercise")
        if st.session_state.exercise_log:
            # Show most recent first
            recent_exercise = list(reversed(st.session_state.exercise_log[-15:]))  # Last 15, newest first
            
            for i, exercise in enumerate(recent_exercise):
                actual_index = len(st.session_state.exercise_log) - 1 - i
                
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                
                with col1:
                    # Different icons for exercise types
                    if "Cardio" in exercise['type']:
                        icon = "🏃‍♀️"
                    elif "Strength" in exercise['type']:
                        icon = "💪"
                    elif "HIIT" in exercise['type']:
                        icon = "⚡"
                    elif "Yoga" in exercise['type']:
                        icon = "🧘‍♀️"
                    else:
                        icon = "🏃‍♀️"
                    
                    st.markdown(f"{icon} **{exercise['type']}**")
                
                with col2:
                    st.write(f"⏱️ {exercise['duration']} min")
                    st.write(f"💪 {exercise['intensity']}")
                
                with col3:
                    st.write(exercise['timestamp'].strftime('%m/%d %I:%M %p'))
                    # Show glucose effect prediction
                    if exercise['type'].startswith("Cardio") or "HIIT" in exercise['type']:
                        st.write("⬇️ May lower glucose")
                    elif "Strength" in exercise['type']:
                        st.write("⬆️ May raise glucose")
                
                with col4:
                    if st.button("🗑️", key=f"del_exercise_{actual_index}", help="Delete this exercise"):
                        delete_entry('exercise', actual_index)
                
                st.divider()
        else:
            st.info("No exercise logged yet. Use the sidebar to log your first workout.")

if __name__ == "__main__":
    main()
# Updated Sat Aug 16 19:17:38 EDT 2025
