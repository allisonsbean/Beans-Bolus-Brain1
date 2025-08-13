import requests
from bs4 import BeautifulSoup
from datetime import datetime, date, timedelta
import re

def test_mfp_web_connection():
    """Test MyFitnessPal connection using web scraping"""
    print("🍎 Bean's Bolus Brain - Testing MyFitnessPal Web Connection...")
    
    # Get credentials
    username = input("Enter your MyFitnessPal username: ")
    password = input("Enter your MyFitnessPal password: ")
    
    try:
        # Create session
        session = requests.Session()
        
        # Get login page
        login_page = session.get('https://www.myfitnesspal.com/account/login')
        soup = BeautifulSoup(login_page.content, 'html.parser')
        
        # Find authenticity token
        auth_token = soup.find('input', {'name': 'authenticity_token'})
        if auth_token:
            auth_token = auth_token.get('value')
        else:
            print("❌ Could not find authentication token")
            return False
        
        # Login data
        login_data = {
            'username': username,
            'password': password,
            'authenticity_token': auth_token,
            'utf8': '✓'
        }
        
        # Submit login
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        login_response = session.post(
            'https://www.myfitnesspal.com/account/login',
            data=login_data,
            headers=headers,
            allow_redirects=True
        )
        
        # Check if login was successful
        if 'invalid' in login_response.text.lower() or login_response.status_code != 200:
            print("❌ Login failed - check username and password")
            return False
        
        print("✅ Successfully logged into MyFitnessPal!")
        
        # Get today's food diary
        today = date.today()
        diary_url = f'https://www.myfitnesspal.com/food/diary/{username}?date={today}'
        
        print(f"📅 Getting food diary for {today.strftime('%B %d, %Y')}...")
        
        diary_response = session.get(diary_url, headers=headers)
        
        if diary_response.status_code != 200:
            print("❌ Could not access food diary")
            return False
        
        # Parse the diary page
        soup = BeautifulSoup(diary_response.content, 'html.parser')
        
        # Look for meal sections
        meals = soup.find_all('div', class_='meal-container') or soup.find_all('table', class_='table0')
        
        if not meals:
            # Try alternative parsing
            print("💡 Trying alternative parsing method...")
            # Look for any tables or divs that might contain food data
            food_elements = soup.find_all(text=re.compile(r'\d+.*carb', re.IGNORECASE))
            if food_elements:
                print("✅ Found some food data!")
                for element in food_elements[:5]:  # Show first 5
                    print(f"   • {element.strip()}")
            else:
                print("❌ No food data found")
                print("💡 Make sure you have meals logged in MyFitnessPal today")
                # Let's see what we got
                print(f"Page title: {soup.title.string if soup.title else 'No title'}")
        else:
            print(f"✅ Found {len(meals)} meal sections!")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_mfp_web_connection()
