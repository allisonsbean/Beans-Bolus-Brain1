import requests
import json
from datetime import datetime, timedelta
import uuid

class DexcomShare:
    def __init__(self, username, password, server="US"):
        self.username = username
        self.password = password
        self.server = server
        self.session_id = None
        self.account_id = None
        
        # API endpoints
        if server.upper() == "US":
            self.base_url = "https://shareous1.dexcom.com/ShareWebServices/Services"
        else:
            self.base_url = "https://shareous1.dexcom.com/ShareWebServices/Services"
            
    def login(self):
        """Authenticate with Dexcom Share API"""
        login_url = f"{self.base_url}/General/LoginPublisherAccountByName"
        
        # Generate application ID (required for newer API versions)
        app_id = "d89443d2-327c-4a6f-89e5-496bbb0317db"
        
        login_data = {
            "password": self.password,
            "username": self.username,
            "applicationId": app_id
        }
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Dexcom Share/3.0.2.11 CFNetwork/711.2.23 Darwin/14.0.0"
        }
        
        try:
            response = requests.post(login_url, 
                                   data=json.dumps(login_data), 
                                   headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                if result and result != "00000000-0000-0000-0000-000000000000":
                    self.session_id = result.replace('"', '')  # Remove quotes if present
                    print("✅ Successfully logged into Dexcom Share!")
                    return True
                else:
                    print("❌ Login failed: Invalid credentials")
                    return False
            else:
                print(f"❌ Login failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Login error: {e}")
            return False
    
    def get_glucose_readings(self, minutes=60, max_count=1):
        """Get recent glucose readings"""
        if not self.session_id:
            print("❌ Not logged in. Please login first.")
            return None
            
        readings_url = f"{self.base_url}/Publisher/ReadPublisherLatestGlucoseValues"
        
        params = {
            "sessionId": self.session_id,
            "minutes": minutes,
            "maxCount": max_count
        }
        
        headers = {
            "User-Agent": "Dexcom Share/3.0.2.11 CFNetwork/711.2.23 Darwin/14.0.0"
        }
        
        try:
            response = requests.post(readings_url, params=params, headers=headers)
            
            print(f"Response status: {response.status_code}")
            print(f"Response length: {len(response.text)}")
            print(f"Raw response: {response.text}")
            
            if response.status_code == 200:
                if response.text.strip() == "":
                    print("❌ Empty response from Dexcom")
                    return None
                    
                readings = response.json()
                
                if not readings:
                    print("❌ No readings found")
                    return None
                    
                # Parse readings
                parsed_readings = []
                for reading in readings:
                    try:
                        # Parse timestamp (Dexcom uses .NET DateTime format)
                        timestamp_str = reading.get('ST')
                        if timestamp_str:
                            # Convert from .NET timestamp format
                            timestamp_ms = int(timestamp_str.split('(')[1].split(')')[0])
                            timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
                        else:
                            timestamp = datetime.now()
                        
                        glucose_value = reading.get('Value', 0)
                        trend = reading.get('Trend', 0)
                        
                        # Convert trend number to arrow
                        trend_arrows = {
                            0: "↗↗",    # DoubleUp
                            1: "↗",     # SingleUp  
                            2: "↗",     # FortyFiveUp
                            3: "→",     # Flat
                            4: "↘",     # FortyFiveDown
                            5: "↘",     # SingleDown
                            6: "↘↘",    # DoubleDown
                            7: "?",     # NotComputable
                            8: "?"      # RateOutOfRange
                        }
                        
                        parsed_reading = {
                            'timestamp': timestamp,
                            'glucose': glucose_value,
                            'trend': trend,
                            'trend_arrow': trend_arrows.get(trend, "?")
                        }
                        parsed_readings.append(parsed_reading)
                        
                    except Exception as e:
                        print(f"⚠️  Error parsing reading: {e}")
                        continue
                
                return parsed_readings
                
            else:
                print(f"❌ Failed to get readings: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error getting readings: {e}")
            return None

def test_dexcom_connection():
    """Test function to verify Dexcom connection"""
    print("🧠 Bean's Bolus Brain - Testing Dexcom Connection...")
    
    # Get credentials
    username = input("Enter your Dexcom Share username: ")
    password = input("Enter your Dexcom Share password: ")
    
    # Create Dexcom connection
    dexcom = DexcomShare(username, password, "US")
    
    # Test login
    if dexcom.login():
        print("🎉 Login successful! Getting recent readings...")
        
        # Get recent readings
        readings = dexcom.get_glucose_readings(minutes=180, max_count=5)
        
        if readings:
            print(f"\n📊 Found {len(readings)} recent readings:")
            for reading in readings:
                print(f"  • {reading['glucose']} mg/dL {reading['trend_arrow']} at {reading['timestamp'].strftime('%I:%M %p')}")
        else:
            print("❌ No readings retrieved")
            print("\n🔍 Troubleshooting tips:")
            print("  • Make sure Share is enabled in your Dexcom app")
            print("  • Ensure you have at least one follower set up")
            print("  • Check that 'Send to Followers' is ON")
            print("  • Wait 5-10 minutes after making changes")
    else:
        print("❌ Login failed")

if __name__ == "__main__":
    test_dexcom_connection()