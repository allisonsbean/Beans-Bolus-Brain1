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
import requests

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
    """Save session data to JSON file for persistence across deployments"""
    try:
        data = {
            'glucose_readings': [],
            'insulin_log': [],
            'meal_log': [],
            'exercise_log': [],
            'last_saved': datetime.now(eastern).isoformat()
        }
        
        # Convert glucose readings
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
                'notes': entry['notes'],
                'ratio_used': entry.get('ratio_used', 12),
                'reasoning': entry.get('reasoning', [])
            })
        
        # Convert meal log
        for entry in st.session_state.meal_log:
            data['meal_log'].append({
                'timestamp': entry['timestamp'].isoformat(),
                'carbs': entry['carbs'],
                'protein': entry['protein'],
                'calories': entry['calories'],
                'description': entry['description'],
                'context': entry.get('context', {}),
                'method': entry.get('method', 'manual')
            })
        
        # Convert exercise log
        for entry in st.session_state.exercise_log:
            data['exercise_log'].append({
                'timestamp': entry['timestamp'].isoformat(),
                'type': entry['type'],
                'duration': entry['duration'],
                'intensity': entry['intensity'],
                'notes': entry['notes'],
                'predicted_effect': entry.get('predicted_effect', '')
            })
        
        # Save to file
        with open('beans_data.json', 'w') as f:
            json.dump(data, f, indent=2)
            
        # Also store in session for backup download
        st.session_state.data_backup = json.dumps(data, indent=2)
            
    except Exception as e:
        st.error(f"Error saving data: {e}")

def load_data_from_file():
    """Load session data from JSON file"""
    try:
        # Try to load from file first
        if os.path.exists('beans_data.json'):
            with open('beans_data.json', 'r') as f:
                data = json.load(f)
        else:
            # No file exists, start fresh
            data = {
                'glucose_readings': [],
                'insulin_log': [],
                'meal_log': [],
                'exercise_log': []
            }
        
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
                'notes': entry['notes'],
                'ratio_used': entry.get('ratio_used', 12),
                'reasoning': entry.get('reasoning', [])
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
                'description': entry['description'],
                'context': entry.get('context', {}),
                'method': entry.get('method', 'manual')
            })
        
        # Load exercise log
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
                'notes': entry['notes'],
                'predicted_effect': entry.get('predicted_effect', '')
            })
                
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Initialize empty logs if loading fails
        st.session_state.glucose_readings = []
        st.session_state.insulin_log = []
        st.session_state.meal_log = []
        st.session_state.exercise_log = []

# Load data on startup
if 'data_loaded' not in st.session_state:
    load_data_from_file()
    st.session_state.data_loaded = True
@st.cache_data(ttl=60)
def get_dexcom_data():
    """Get real-time glucose data from Dexcom Share"""
    try:
        # Get credentials from secrets
        username = st.secrets.get("dexcom", {}).get("username", "allisonsbean@gmail.com")
        password = st.secrets.get("dexcom", {}).get("password", "Allison9")
        
        dexcom = Dexcom(username=username, password=password)
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
            
            # Check if we should add this reading (avoid duplicates)
            should_add = True
            if st.session_state.glucose_readings:
                last_reading = st.session_state.glucose_readings[-1]
                time_diff = abs((glucose_data['timestamp'] - last_reading['timestamp']).total_seconds())
                if time_diff < 60:  # Don't add if less than 1 minute apart
                    should_add = False

            if should_add:
                st.session_state.glucose_readings.append(glucose_data)
                # Keep only last 500 readings to prevent memory issues
                if len(st.session_state.glucose_readings) > 500:
                    st.session_state.glucose_readings = st.session_state.glucose_readings[-500:]
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
        
        # Only count bolus insulin for IOB
        if hours_elapsed < IOB_DURATION_HOURS and entry['type'] == 'bolus':
            # Linear decay over 4 hours
            remaining_fraction = max(0, (IOB_DURATION_HOURS - hours_elapsed) / IOB_DURATION_HOURS)
            total_iob += entry['dose'] * remaining_fraction
    
    return total_iob

def calculate_bolus_suggestion(carbs, protein, current_glucose, current_iob):
    """Calculate bolus suggestion with IOB adjustment"""
    # Carb bolus calculation
    carb_bolus = carbs / CARB_RATIO
    
    # Protein bolus calculation (10% rule)
    protein_carb_equivalent = protein * 0.1
    protein_bolus = protein_carb_equivalent / CARB_RATIO
    
    # Correction bolus calculation
    correction_needed = max(0, current_glucose - TARGET_GLUCOSE)
    correction_bolus = correction_needed / CORRECTION_FACTOR
    
    # Adjust correction for IOB
    adjusted_correction = max(0, correction_bolus - current_iob)
    
    # Total bolus (capped at max)
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
    
    # Display with proper formatting
    st.markdown(f"<h1 style='text-align: center; color: {color};'>{value} mg/dL {arrow}</h1>", 
                unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>{status}</h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;'>Trend: {trend}</p>", unsafe_allow_html=True)

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

def show_data_management():
    """Show data backup and restore options"""
    st.markdown("### 📦 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Create Backup"):
            save_data_to_file()  # Ensure current data is saved
            if 'data_backup' in st.session_state:
                st.download_button(
                    label="📥 Download JSON Backup",
                    data=st.session_state.data_backup,
                    file_name=f"beans_data_backup_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json"
                )
                st.success("✅ Backup ready for download!")
            else:
                st.info("No data to backup yet")
    
    with col2:
        uploaded_backup = st.file_uploader(
            "📤 Restore from Backup", 
            type=['json'],
            help="Upload a previously downloaded backup file"
        )
        
        if uploaded_backup:
            if st.button("🔄 Restore Data"):
                try:
                    backup_data = json.load(uploaded_backup)
                    
                    # Save backup data to file
                    with open('beans_data.json', 'w') as f:
                        json.dump(backup_data, f, indent=2)
                    
                    # Reload data from file
                    load_data_from_file()
                    st.success("✅ Data restored successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error restoring data: {e}")
def analyze_food_photo(image_file):
    """Analyze food photo using Claude AI"""
    try:
        # Get API key from Streamlit secrets
        try:
            api_key = st.secrets["claude"]["api_key"]
        except:
            # Fallback to old format if new format doesn't work
            api_key = st.secrets.get("CLAUDE_API_KEY")
        
        if not api_key:
            return {"error": "Claude AI API key not found in Streamlit secrets", "success": False}
        
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

def lookup_nutrition_by_barcode(barcode):
    """Look up nutrition information using OpenFoodFacts API with fixed calculations"""
    try:
        # OpenFoodFacts API call
        url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('status') == 1:  # Product found
                product = data['product']
                
                # Extract nutrition info
                product_name = product.get('product_name', 'Unknown Product')
                serving_size = product.get('serving_size', 'Check package')
                
                # Get nutrition per 100g
                nutriments = product.get('nutriments', {})
                carbs_100g = nutriments.get('carbohydrates_100g', 0)
                protein_100g = nutriments.get('proteins_100g', 0)
                calories_100g = nutriments.get('energy-kcal_100g', 0)
                
                # Store product data in session state to persist across reruns
                st.session_state.current_product = {
                    'name': product_name,
                    'serving_size': serving_size,
                    'carbs_100g': float(carbs_100g) if carbs_100g else 0,
                    'protein_100g': float(protein_100g) if protein_100g else 0,
                    'calories_100g': float(calories_100g) if calories_100g else 0,
                    'barcode': barcode
                }
                
                # Display product info
                st.success(f"✅ **{product_name}**")
                st.info(f"📏 **Serving Size:** {serving_size}")
                
                # Show per-100g nutrition in compact format
                st.write(f"**Per 100g:** 🍞 {carbs_100g}g carbs | 🥩 {protein_100g}g protein | 🔥 {calories_100g} cal")
                
            else:
                st.error("❌ Product not found in database")
                return
        else:
            st.error("❌ Error connecting to nutrition database")
            return
            
    except Exception as e:
        st.error(f"❌ Error looking up barcode: {str(e)}")
        return

def show_barcode_nutrition_calculator():
    """Show nutrition calculator for current product with FIXED calculations"""
    if 'current_product' not in st.session_state:
        st.info("👆 Enter a barcode above to get started")
        return
    
    product = st.session_state.current_product
    
    # FIXED: Better serving size extraction
    import re
    serving_grams = 30  # Default fallback
    
    try:
        serving_size = product['serving_size']
        # Look for numbers followed by 'g' in the serving_size string
        grams_match = re.search(r'(\d+\.?\d*)\s*g', serving_size, re.IGNORECASE)
        if grams_match:
            serving_grams = float(grams_match.group(1))
        else:
            # Look for other patterns like "30" in parentheses
            number_match = re.search(r'\(?(\d+\.?\d*)\)?', serving_size)
            if number_match:
                serving_grams = float(number_match.group(1))
    except:
        serving_grams = 30  # Keep default if parsing fails
    
    # Serving size input
    st.markdown("### 📊 Calculate Your Portion")
    num_servings = st.number_input(
        f"How many servings? (1 serving = {product['serving_size']})", 
        min_value=0.1, 
        max_value=10.0, 
        value=1.0, 
        step=0.5,
        key="barcode_servings"
    )
    
    # FIXED: Proper calculation implementation
    # Calculate nutrition for actual serving size
    carbs_per_serving = (product['carbs_100g'] * serving_grams / 100)
    protein_per_serving = (product['protein_100g'] * serving_grams / 100)
    calories_per_serving = (product['calories_100g'] * serving_grams / 100)
    
    # Calculate for user's portion
    actual_carbs = round(carbs_per_serving * num_servings, 1)
    actual_protein = round(protein_per_serving * num_servings, 1)
    actual_calories = round(calories_per_serving * num_servings)
    
    # Mobile-friendly compact display
    st.markdown(f"**Your portion ({num_servings} serving{'s' if num_servings != 1 else ''}):**")
    
    # Use columns for better mobile layout
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"🍞 **{actual_carbs}g**  \ncarbs")
    with col2:
        st.markdown(f"🥩 **{actual_protein}g**  \nprotein")
    with col3:
        st.markdown(f"🔥 **{actual_calories}**  \ncalories")
    
    # Show calculation for verification
    st.caption(f"Calculation: {product['carbs_100g']}g × {serving_grams}g ÷ 100g × {num_servings} = {actual_carbs}g carbs")
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📋 Transfer to Meal Logger", key="transfer_meal_barcode"):
            st.session_state.barcode_nutrition = {
                'carbs': actual_carbs,
                'protein': actual_protein,
                'calories': actual_calories,
                'description': product['name'],
                'method': 'barcode'
            }
            st.success(f"✅ Transferred: {actual_carbs}g carbs!")
            
    with col2:
        if st.button("💉 Quick Bolus Calc", key="quick_bolus_barcode"):
            # Calculate bolus suggestion
            carb_bolus = actual_carbs / CARB_RATIO  # Using CARB_RATIO constant
            protein_bolus = (actual_protein * 0.1) / CARB_RATIO
            total_bolus = round(carb_bolus + protein_bolus)
            
            st.success(f"💉 **{total_bolus} units suggested**")
            st.write(f"• {actual_carbs}g carbs = {carb_bolus:.1f}u")
            if protein_bolus > 0:
                st.write(f"• {actual_protein}g protein = {protein_bolus:.1f}u")
def detect_barcode_in_image(image):
    """Detect if image contains a barcode using simple heuristics"""
    try:
        import numpy as np
        
        # Ensure image is not too large
        max_size = 800
        if image.width > max_size or image.height > max_size:
            image = image.copy()
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        # Convert PIL image to numpy array
        img_array = np.array(image.convert('L'))  # Convert to grayscale
        
        # Check if array is valid size
        if img_array.size == 0:
            return False
            
        height, width = img_array.shape
        
        # Skip tiny images
        if height < 50 or width < 50:
            return False
        
        # Check for horizontal lines (barcode pattern)
        horizontal_edges = 0
        for row in img_array[height//3:2*height//3]:  # Check middle third
            edges = 0
            for i in range(1, len(row)):
                if abs(int(row[i]) - int(row[i-1])) > 50:  # Significant color change
                    edges += 1
            if edges > width // 4:  # Many edges suggests barcode
                horizontal_edges += 1
        
        # If many rows have lots of edges, probably a barcode
        barcode_score = horizontal_edges / (height // 3)
        
        return barcode_score > 0.3  # Threshold for barcode detection
        
    except Exception as e:
        return False  # If detection fails, assume it's food

def extract_barcode_from_image(image):
    """Try to extract barcode number from image using pyzbar"""
    try:
        # Try to import pyzbar
        from pyzbar import pyzbar
        import numpy as np
        
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Decode barcodes
        barcodes = pyzbar.decode(img_array)
        
        if barcodes:
            # Return the first barcode found
            barcode_data = barcodes[0].data.decode('utf-8')
            barcode_type = barcodes[0].type
            
            st.success(f"📱 **Automatically read barcode:** {barcode_data} (Type: {barcode_type})")
            return barcode_data
        else:
            return None
            
    except ImportError:
        st.warning("📱 Automatic barcode reading not available - install pyzbar library")
        return None
    except Exception as e:
        st.error(f"Error reading barcode: {str(e)}")
        return None

def smart_camera_interface():
    """Smart camera interface for both barcodes and food photos"""
    st.subheader("📸 Smart Camera - Barcodes & Food Photos")
    
    # File uploader for photos
    uploaded_file = st.file_uploader(
        "📱 Take photo or upload image", 
        type=['png', 'jpg', 'jpeg'],
        help="Point camera at barcode OR take a food photo - we'll detect which!"
    )
    
    if uploaded_file:
        # Display the image
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(uploaded_file, width=200, caption="Your photo")
        
        with col2:
            if st.button("🤖 Analyze Photo", key="smart_analyze"):
                with st.spinner("Detecting content type..."):
                    try:
                        # Load and resize image for mobile compatibility
                        image = Image.open(uploaded_file)
                        
                        # Resize large images to prevent memory issues
                        max_size = 1024
                        if image.width > max_size or image.height > max_size:
                            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                        
                        # Detect if barcode or food
                        is_barcode = detect_barcode_in_image(image)
                        
                    except Exception as e:
                        st.error(f"❌ Error processing image: {str(e)}")
                        st.info("💡 Try taking a smaller photo or using a different image format")
                        return
                    
                    if is_barcode:
                        st.info("📱 **Barcode detected!** Attempting to read...")
                        
                        # Try to extract barcode
                        barcode = extract_barcode_from_image(image)
                        
                        if barcode:
                            st.success(f"Found barcode: {barcode}")
                            lookup_nutrition_by_barcode(barcode)
                        else:
                            st.warning("📱 **Barcode detected but couldn't read the numbers automatically.**")
                            st.info("Please enter the barcode manually below:")
                    
                    else:
                        st.info("🍽️ **Food photo detected!** Analyzing with Claude AI...")
                        
                        # Use existing food photo analysis
                        analysis = analyze_food_photo(uploaded_file)
                        if analysis.get('success'):
                            st.success("✅ Food analysis complete!")
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
    
    # Manual barcode entry option
    st.markdown("---")
    st.markdown("**Or enter barcode manually:**")
    manual_barcode = st.text_input("UPC/Barcode:", placeholder="Enter barcode numbers", key="manual_barcode_input")
    
    if manual_barcode and st.button("🔍 Look Up Product", key="manual_lookup"):
        lookup_nutrition_by_barcode(manual_barcode)
    
    # Show nutrition calculator if we have a product
    show_barcode_nutrition_calculator()
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
        
        # Current date/time as defaults
        now = datetime.now(eastern)
        current_date = now.date()
        current_time = now.time().replace(second=0, microsecond=0)
        
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
                st.session_state.glucose_readings.sort(key=lambda x: x['timestamp'])
                save_data_to_file()
                st.success(f"Logged {manual_glucose} mg/dL")
                st.rerun()
        
        # Bolus logging
        with st.expander("💉 Log Bolus"):
            bolus_dose = st.number_input("Bolus dose (units)", min_value=0.0, max_value=20.0, step=0.5)
            bolus_notes = st.text_input("Notes (optional)")
            if st.button("Log Bolus"):
                entry = {
                    'timestamp': datetime.now(eastern),
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
            basal_dose = st.number_input("Basal dose (units)", min_value=0.0, max_value=30.0, step=0.5, value=float(st.session_state.basal_dose))
            basal_notes = st.text_input("Basal notes (optional)", key="basal_notes_input")
            if st.button("Log Basal"):
                entry = {
                    'timestamp': datetime.now(eastern),
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

        # Smart camera section
        with st.expander("📸 Smart Camera - Barcodes & Food Photos"):
            smart_camera_interface()

        # Enhanced meal logging
        with st.expander("🍽️ Smart Meal & Bolus Calculator"):
            
            # Use barcode data if available, otherwise photo analysis
            if 'barcode_nutrition' in st.session_state:
                default_carbs = st.session_state.barcode_nutrition.get('carbs', 30)
                default_protein = st.session_state.barcode_nutrition.get('protein', 0)
                default_calories = st.session_state.barcode_nutrition.get('calories', 0)
                default_description = st.session_state.barcode_nutrition.get('description', '')
                meal_method = st.session_state.barcode_nutrition.get('method', 'barcode')
            elif 'photo_analysis' in st.session_state:
                default_carbs = st.session_state.photo_analysis.get('total_carbs', 30)
                default_protein = st.session_state.photo_analysis.get('total_protein', 0)
                default_calories = st.session_state.photo_analysis.get('total_calories', 0)
                default_description = 'Claude AI analyzed meal'
                meal_method = 'photo'
            else:
                default_carbs = 30
                default_protein = 0
                default_calories = 0
                default_description = ''
                meal_method = 'manual'
                
            meal_carbs = st.number_input("Carbs (g)", min_value=0, max_value=200, value=int(default_carbs), key="smart_meal_carbs")
            meal_protein = st.number_input("Protein (g)", min_value=0, max_value=100, value=int(default_protein), key="smart_meal_protein")
            meal_calories = st.number_input("Calories", min_value=0, max_value=2000, value=int(default_calories), key="smart_meal_calories")
            meal_description = st.text_input("Meal description", value=default_description, key="smart_meal_description")
            
            # Get current glucose for bolus calculation
            try:
                current_glucose_data = get_dexcom_data()
                if not current_glucose_data and st.session_state.glucose_readings:
                    current_glucose_data = st.session_state.glucose_readings[-1]
            except:
                current_glucose_data = None
            
            # Manual glucose input if Dexcom not available
            if not current_glucose_data:
                st.warning("⚠️ No Dexcom data available")
                manual_glucose = st.number_input("Current glucose (mg/dL):", min_value=40, max_value=400, value=120, key="manual_glucose_for_bolus")
                current_glucose_data = {'value': manual_glucose}
            
            if current_glucose_data:
                # Calculate bolus suggestion
                current_iob = calculate_iob()
                bolus_suggestion = calculate_bolus_suggestion(meal_carbs, meal_protein, current_glucose_data['value'], current_iob)
                
                st.markdown("### 💊 Bolus Suggestion")
                st.info(f"**Total Suggested:** {bolus_suggestion['total_bolus']}u")
                st.write(f"• Carbs: {bolus_suggestion['carb_bolus']}u")
                if bolus_suggestion['protein_bolus'] > 0:
                    st.write(f"• Protein: {bolus_suggestion['protein_bolus']}u")
                if bolus_suggestion['correction_bolus'] > 0:
                    st.write(f"• Correction: {bolus_suggestion['correction_bolus']}u")
                
                # Action buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📝 Log Meal Only", key="log_smart_meal_only"):
                        entry = {
                            'timestamp': datetime.now(eastern),
                            'carbs': meal_carbs,
                            'protein': meal_protein,
                            'calories': meal_calories,
                            'description': meal_description,
                            'method': meal_method
                        }
                        st.session_state.meal_log.append(entry)
                        st.session_state.meal_log.sort(key=lambda x: x['timestamp'])
                        save_data_to_file()
                        st.success("✅ Meal logged!")
                        # Clear data after use
                        if 'barcode_nutrition' in st.session_state:
                            del st.session_state.barcode_nutrition
                        if 'photo_analysis' in st.session_state:
                            del st.session_state.photo_analysis
                        st.rerun()
                
                with col2:
                    if st.button("💉 Log Meal + Bolus", key="log_meal_and_bolus"):
                        meal_time = datetime.now(eastern)
                        
                        # Add meal
                        meal_entry = {
                            'timestamp': meal_time,
                            'carbs': meal_carbs,
                            'protein': meal_protein,
                            'calories': meal_calories,
                            'description': meal_description,
                            'method': meal_method
                        }
                        st.session_state.meal_log.append(meal_entry)
                        
                        # Add bolus
                        bolus_entry = {
                            'timestamp': meal_time,
                            'type': 'bolus',
                            'dose': bolus_suggestion['total_bolus'],
                            'notes': f"Meal bolus: {meal_description}",
                            'ratio_used': CARB_RATIO
                        }
                        st.session_state.insulin_log.append(bolus_entry)
                        
                        # Sort both logs
                        st.session_state.meal_log.sort(key=lambda x: x['timestamp'])
                        st.session_state.insulin_log.sort(key=lambda x: x['timestamp'])
                        save_data_to_file()
                        st.success(f"✅ Logged meal + {bolus_suggestion['total_bolus']}u bolus!")
                        # Clear data after use
                        if 'barcode_nutrition' in st.session_state:
                            del st.session_state.barcode_nutrition
                        if 'photo_analysis' in st.session_state:
                            del st.session_state.photo_analysis
                        st.rerun()
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
                    index=1,
                    key="exercise_intensity"
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
                # Create exercise entry
                exercise_entry = {
                    'timestamp': datetime.now(eastern),
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

        # Data backup section
        with st.expander("📦 Data Backup & Restore"):
            show_data_management()
# Enhanced data tables with delete functionality
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Glucose History", "💉 Insulin History", "🍽️ Meal History", "🏃‍♀️ Exercise History"])
    
    with tab1:
        st.subheader("Recent Glucose Readings")
        if st.session_state.glucose_readings:
            recent_readings = list(reversed(st.session_state.glucose_readings[-15:]))
            
            for i, reading in enumerate(recent_readings):
                actual_index = len(st.session_state.glucose_readings) - 1 - i
                
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                
                with col1:
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
                    if st.button("🗑️", key=f"del_glucose_{actual_index}", help="Delete this reading"):
                        delete_entry('glucose', actual_index)
                
                st.divider()
        else:
            st.info("No glucose readings yet. Add one manually or wait for Dexcom data.")
    
    with tab2:
        st.subheader("Recent Insulin Doses")
        if st.session_state.insulin_log:
            recent_insulin = list(reversed(st.session_state.insulin_log[-15:]))
            
            for i, dose in enumerate(recent_insulin):
                actual_index = len(st.session_state.insulin_log) - 1 - i
                
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                
                with col1:
                    icon = "💉" if dose['type'] == 'bolus' else "🕐"
                    st.markdown(f"{icon} **{dose['dose']} units**")
                
                with col2:
                    st.write(f"{dose['type']}")
                
                with col3:
                    st.write(dose['timestamp'].strftime('%m/%d %I:%M %p'))
                
                with col4:
                    if st.button("🗑️", key=f"del_insulin_{actual_index}", help="Delete this dose"):
                        delete_entry('insulin', actual_index)
                
                st.divider()
        else:
            st.info("No insulin doses logged yet. Use the sidebar to log your first dose.")
    
    with tab3:
        st.subheader("Recent Meals")
        if st.session_state.meal_log:
            recent_meals = list(reversed(st.session_state.meal_log[-15:]))
            
            for i, meal in enumerate(recent_meals):
                actual_index = len(st.session_state.meal_log) - 1 - i
                
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                with col1:
                    st.markdown(f"🍽️ **{meal.get('description', 'Unknown meal')}**")
                    method = meal.get('method', 'manual')
                    if method == 'photo':
                        st.write("📸 Photo analyzed")
                    elif method == 'barcode':
                        st.write("📱 Barcode scanned")
                
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
            recent_exercise = list(reversed(st.session_state.exercise_log[-15:]))
            
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
