from pydexcom import Dexcom

def test_pydexcom():
    print("🧠 Testing with PyDexcom library...")
    
    username = input("Enter your Dexcom Share username: ")
    password = input("Enter your Dexcom Share password: ")
    
    try:
        # Create Dexcom instance using legacy username format
        dexcom = Dexcom(username=username, password=password)
        
        # Get current glucose reading
        bg = dexcom.get_current_glucose_reading()
        
        if bg:
            print(f"✅ Success! Current glucose: {bg.value} mg/dL")
            print(f"📅 Time: {bg.datetime}")
            print(f"📈 Trend: {bg.trend_description}")
            
            # Get recent readings
            readings = dexcom.get_glucose_readings()
            print(f"\n📊 Recent readings ({len(readings)} found):")
            for reading in readings[:5]:  # Show last 5
                print(f"  • {reading.value} mg/dL at {reading.datetime.strftime('%I:%M %p')}")
        else:
            print("❌ No glucose reading found")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_pydexcom()