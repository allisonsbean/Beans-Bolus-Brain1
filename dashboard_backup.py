import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pydexcom import Dexcom
import time

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
        margin-bottom: 1rem;
    }
    .current-bg {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .bg-normal { background-color: #d4edda; color: #155724; }
    .bg-high { background-color: #f8d7da; color: #721c24; }
    .bg-low { background-color: #fff3cd; color: #856404; }
    .bg-stale { background-color: #e2e3e5; color: #6c757d; }
    .manual-entry { background-color: #fff3cd; color: #856404; }
    .signal-loss { background-color: #f8d7da; color: #721c24; border: 2px dashed #dc3545; }
    .meal-summary {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

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

def create_glucose_chart(readings_df, manual_readings=None, meals=None):
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
    
    # Add target range
    fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Target Low")
    fig.add_hline(y=130, line_dash="dash", line_color="green", annotation_text="Target High")
    fig.add_hrect(y0=80, y1=130, fillcolor="green", opacity=0.1)
    
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
    time_diff = datetime.now() - reading_time.replace(tzinfo=None)
    return time_diff.total_seconds() > (max_minutes * 60)

def calculate_bolus_suggestion(current_glucose, carbs, protein, target_glucose=115, carb_ratio=12, correction_factor=50, protein_factor=10):
    """Calculate suggested bolus dose including protein"""
    # Carb bolus
    carb_bolus = carbs / carb_ratio
    
    # Protein bolus (10% of protein weight converted to carb equivalent)
    protein_carb_equivalent = protein * (protein_factor / 100)
    protein_bolus = protein_carb_equivalent / carb_ratio
    
    # Correction bolus
    correction_bolus = max(0, (current_glucose - target_glucose) / correction_factor)
    
    total_bolus = carb_bolus + protein_bolus + correction_bolus
    
    return {
        'carb_bolus': carb_bolus,
        'protein_bolus': protein_bolus,
        'protein_carb_equivalent': protein_carb_equivalent,
        'correction_bolus': correction_bolus,
        'total_bolus': total_bolus
    }

def main():
    st.markdown('<h1 class="main-header">🧠 Bean\'s Bolus Brain</h1>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'manual_readings' not in st.session_state:
        st.session_state.manual_readings = []
    if 'meals' not in st.session_state:
        st.session_state.meals = []
    
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
        
        st.header("✋ Manual Glucose")
        st.write("Use when Dexcom has signal loss")
        manual_glucose = st.number_input("Glucose (mg/dL)", min_value=40, max_value=400, value=100, key="manual_glucose")
        manual_time = st.time_input("Time", value=datetime.now().time(), key="manual_time")
        
        if st.button("Add Manual Reading"):
            manual_entry = {
                'datetime': datetime.combine(datetime.now().date(), manual_time),
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
        meal_time = st.time_input("Meal Time", value=datetime.now().time(), key="meal_time")
        
        if st.button("Log Meal"):
            if meal_name and (meal_carbs > 0 or meal_protein > 0):
                meal_entry = {
                    'datetime': datetime.combine(datetime.now().date(), meal_time),
                    'name': meal_name,
                    'carbs': meal_carbs,
                    'protein': meal_protein,
                    'calories': meal_calories
                }
                st.session_state.meals.append(meal_entry)
                st.success(f"Logged: {meal_name} - {meal_carbs}g carbs, {meal_protein}g protein")
            else:
                st.error("Please enter meal name and carbs or protein")
        
        # Quick meal buttons
        st.write("Quick Add:")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🍎 Apple (25g)"):
                quick_meal = {
                    'datetime': datetime.now(),
                    'name': 'Apple',
                    'carbs': 25,
                    'protein': 0,
                    'calories': 95
                }
                st.session_state.meals.append(quick_meal)
        
        with col2:
            if st.button("🥩 Chicken (25g)"):
                quick_meal = {
                    'datetime': datetime.now(),
                    'name': 'Chicken Breast (3oz)',
                    'carbs': 0,
                    'protein': 25,
                    'calories': 140
                }
                st.session_state.meals.append(quick_meal)
        
        # Clear data options
        if st.session_state.manual_readings or st.session_state.meals:
            st.header("🗑️ Clear Data")
            if st.button("Clear Manual Readings"):
                st.session_state.manual_readings = []
                st.success("Manual readings cleared")
            if st.button("Clear Meals"):
                st.session_state.meals = []
                st.success("Meals cleared")
    
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
    
    if current:
        display_glucose = current.value
        display_time = current.datetime
        display_source = f"Dexcom • {current.trend_description} {current.trend_arrow}"
        is_stale = is_reading_stale(current.datetime)
    elif readings and len(readings) > 0:
        latest = max(readings, key=lambda r: r.datetime)
        display_glucose = latest.value
        display_time = latest.datetime
        display_source = f"Last Dexcom • {latest.trend_description}"
        is_stale = is_reading_stale(latest.datetime)
    elif st.session_state.manual_readings:
        latest_manual = max(st.session_state.manual_readings, key=lambda r: r['datetime'])
        display_glucose = latest_manual['glucose']
        display_time = latest_manual['datetime']
        display_source = "Manual Entry"
        signal_loss = True
    else:
        signal_loss = True
    
    # Display current status
    if display_glucose:
        bg_class = get_bg_class(display_glucose, 
                               is_manual=(display_source == "Manual Entry"),
                               is_stale=is_stale,
                               signal_loss=signal_loss)
        
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
                <br><small>{time_str}</small>
                <br><small>{status_icon}</small>
                <br><small>{display_source}</small>
            </div>
        ''', unsafe_allow_html=True)
        
        # Show recent meals summary
        if st.session_state.meals:
            today_meals = [m for m in st.session_state.meals 
                          if m['datetime'].date() == datetime.now().date()]
            if today_meals:
                total_carbs_today = sum(m['carbs'] for m in today_meals)
                total_protein_today = sum(m.get('protein', 0) for m in today_meals)
                total_calories_today = sum(m['calories'] for m in today_meals)
                
                st.markdown(f'''
                    <div class="meal-summary">
                        <strong>🍽️ Today's Meals: {len(today_meals)} meals, {total_carbs_today}g carbs, {total_protein_today}g protein, {total_calories_today} calories</strong>
                    </div>
                ''', unsafe_allow_html=True)
        
        # Bolus calculator
        if st.session_state.meals and display_glucose:
            recent_meals = [m for m in st.session_state.meals 
                           if m['datetime'] > datetime.now() - timedelta(hours=2)]
            if recent_meals:
                recent_carbs = sum(m['carbs'] for m in recent_meals)
                recent_protein = sum(m.get('protein', 0) for m in recent_meals)
                bolus_calc = calculate_bolus_suggestion(display_glucose, recent_carbs, recent_protein)
                
                st.subheader("💉 Bolus Suggestion (Last 2 Hours)")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Carb Bolus", f"{bolus_calc['carb_bolus']:.1f} units", f"{recent_carbs}g carbs")
                with col2:
                    st.metric("Protein Bolus", f"{bolus_calc['protein_bolus']:.1f} units", f"{recent_protein}g protein")
                with col3:
                    st.metric("Correction", f"{bolus_calc['correction_bolus']:.1f} units", f"Target: 115 mg/dL")
                with col4:
                    st.metric("Total Suggested", f"{bolus_calc['total_bolus']:.1f} units", "Review before dosing")
                
                st.info(f"💡 Protein calculation: {recent_protein}g protein = {bolus_calc['protein_carb_equivalent']:.1f}g carb equivalent (10% rule)")
                st.warning("⚠️ This is a suggestion only. Always verify with your healthcare provider's guidelines.")
        
        # Metrics row
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if current and hasattr(current, 'trend_description'):
                st.metric("Trend", current.trend_description.title())
            else:
                st.metric("Trend", "Unknown")
        
        with col2:
            if readings:
                recent_readings = [r for r in readings if r.datetime.replace(tzinfo=None) > datetime.now() - timedelta(hours=24)]
                in_range = len([r for r in recent_readings if 80 <= r.value <= 130])
                time_in_range = (in_range / len(recent_readings) * 100) if recent_readings else 0
                st.metric("Time in Range (24h)", f"{time_in_range:.1f}%")
            else:
                st.metric("Time in Range (24h)", "No data")
        
        with col3:
            if st.session_state.meals:
                today_carbs = sum(m['carbs'] for m in st.session_state.meals if m['datetime'].date() == datetime.now().date())
                today_protein = sum(m.get('protein', 0) for m in st.session_state.meals if m['datetime'].date() == datetime.now().date())
                st.metric("Today's Intake", f"{today_carbs}g carbs", f"{today_protein}g protein")
            else:
                st.metric("Today's Intake", "0g carbs", "0g protein")
        
        # Create combined chart
        readings_df = pd.DataFrame()
        if readings:
            readings_data = []
            for reading in readings:
                readings_data.append({
                    'datetime': reading.datetime,
                    'glucose': reading.value,
                    'trend': reading.trend_description
                })
            readings_df = pd.DataFrame(readings_data)
            readings_df = readings_df.sort_values('datetime')
        
        # Filter meals to last 12 hours for chart
        recent_meals_for_chart = [m for m in st.session_state.meals 
                                 if m['datetime'] > datetime.now() - timedelta(hours=12)]
        
        if not readings_df.empty or st.session_state.manual_readings or recent_meals_for_chart:
            fig = create_glucose_chart(readings_df, st.session_state.manual_readings, recent_meals_for_chart)
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