import myfitnesspal
from datetime import datetime, timedelta

def test_myfitnesspal_connection():
    """Test MyFitnessPal connection and get recent meals"""
    print("🍎 Bean's Bolus Brain - Testing MyFitnessPal Connection...")
    
    # Get credentials
    username = input("Enter your MyFitnessPal username: ")
    password = input("Enter your MyFitnessPal password: ")
    
    try:
        # Create MyFitnessPal client
        client = myfitnesspal.Client(username, password)
        
        print("✅ Successfully logged into MyFitnessPal!")
        
        # Get today's food diary
        today = datetime.now().date()
        print(f"📅 Getting food diary for {today.strftime('%B %d, %Y')}...")
        
        day = client.get_date(today)
        
        print(f"\n🍽️ Today's Meals:")
        total_carbs = 0
        total_calories = 0
        meal_count = 0
        
        for meal in day.meals:
            if meal.entries:  # Only show meals that have food entries
                meal_count += 1
                meal_carbs = sum(food.carbohydrates or 0 for food in meal.entries)
                meal_calories = sum(food.calories or 0 for food in meal.entries)
                
                print(f"\n  📍 {meal.name}:")
                print(f"     Carbs: {meal_carbs:.1f}g | Calories: {meal_calories:.0f}")
                
                for food in meal.entries:
                    carbs = food.carbohydrates or 0
                    calories = food.calories or 0
                    print(f"     • {food.name} - {carbs:.1f}g carbs, {calories:.0f} cal")
                
                total_carbs += meal_carbs
                total_calories += meal_calories
        
        print(f"\n📊 Daily Totals:")
        print(f"   Total Carbs: {total_carbs:.1f}g")
        print(f"   Total Calories: {total_calories:.0f}")
        print(f"   Meals logged: {meal_count}")
        
        if meal_count == 0:
            print("\n💡 No meals found for today. Try logging some food in MyFitnessPal first!")
        
        # Test getting yesterday's data too
        yesterday = today - timedelta(days=1)
        print(f"\n📅 Testing yesterday ({yesterday.strftime('%B %d, %Y')})...")
        
        try:
            yesterday_day = client.get_date(yesterday)
            yesterday_carbs = 0
            yesterday_meals = 0
            
            for meal in yesterday_day.meals:
                if meal.entries:
                    yesterday_meals += 1
                    yesterday_carbs += sum(food.carbohydrates or 0 for food in meal.entries)
            
            print(f"   Yesterday: {yesterday_carbs:.1f}g carbs, {yesterday_meals} meals")
            
        except Exception as e:
            print(f"   ⚠️ Couldn't get yesterday's data: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ MyFitnessPal connection failed: {e}")
        print("\n🔍 Common issues:")
        print("   • Make sure username/password are correct")
        print("   • Use your MyFitnessPal website credentials (not app login)")
        print("   • Try logging into myfitnesspal.com first to verify")
        print("   • Make sure you have some meals logged in MyFitnessPal")
        return False

if __name__ == "__main__":
    test_myfitnesspal_connection()
