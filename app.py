import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict
from werkzeug.security import generate_password_hash, check_password_hash
import logging
from contextlib import contextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DB_PATH = 'nutriusher.db'
ASSETS_DIR = Path('assets')
LOGO_PATH = ASSETS_DIR / 'logo.png'
NUTRITION_IMG_PATH = ASSETS_DIR / 'nutrition_food.jpg'

# Custom Exceptions
class NutriUsherError(Exception):
    """Base exception for NutriUsher application"""
    pass

class DatabaseError(NutriUsherError):
    """Database related errors"""
    pass

class AuthenticationError(NutriUsherError):
    """Authentication related errors"""
    pass

# Data Models
@dataclass
class User:
    id: Optional[int]
    email: str
    username: str
    password: str  # Stored as hash

@dataclass
class UserDetails:
    id: Optional[int]
    user_id: int
    gender: str
    age: int
    height: float
    weight: float
    activity_level: str
    condition: str
    diet: str

@dataclass
class DietPlan:
    id: Optional[int]
    user_id: int
    plan: str
    created_at: datetime

@dataclass
class Meal:
    food: str
    sugars: float
    carbs: float
    proteins: float
    fats: float
    calories: float
    sodium: float

@dataclass
class DailyMeals:
    breakfast: List[Meal]
    lunch: List[Meal]
    dinner: List[Meal]
# Database Manager
class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Initialize database with required tables"""
        with self.get_connection() as conn:
            c = conn.cursor()
            c.executescript('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE,
                    username TEXT UNIQUE,
                    password TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS user_details (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    gender TEXT,
                    age INTEGER,
                    height REAL,
                    weight REAL,
                    activity_level TEXT,
                    condition TEXT,
                    diet TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                    );
            ''')
            conn.commit()

# Authentication Manager
class AuthManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def register_user(self, email: str, username: str, password: str) -> User:
        """Register a new user"""
        if len(password) < 8:
            raise AuthenticationError("Password must be at least 8 characters long")

        hashed_password = generate_password_hash(password)
        with self.db_manager.get_connection() as conn:
            try:
                c = conn.cursor()
                c.execute(
                    "INSERT INTO users (email, username, password) VALUES (?, ?, ?)",
                    (email, username, hashed_password)
                )
                conn.commit()
                return User(
                    id=c.lastrowid,
                    email=email,
                    username=username,
                    password=hashed_password
                )
            except sqlite3.IntegrityError:
                raise AuthenticationError("Username or email already exists")

    def login_user(self, username: str, password: str) -> User:
        """Authenticate and login a user"""
        with self.db_manager.get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE username = ?", (username,))
            user_data = c.fetchone()
            
            if not user_data or not check_password_hash(user_data[3], password):
                raise AuthenticationError("Invalid username or password")

            return User(
                id=user_data[0],
                email=user_data[1],
                username=user_data[2],
                password=user_data[3]
            )

# UI Components
class UIComponents:
    @staticmethod
    def set_page_config():
        st.set_page_config(
                page_title="NutriUsher",
                page_icon="🥗",
                layout="wide"
            )

        st.markdown("""
            <style>
            .main { padding: 0rem 1rem; }
            .stButton>button { width: 100%; }
            .stTextInput>div>div>input { color: #4F8BF9; }
            .login-box {
                max-width: 400px;
                margin: auto;
                padding: 2rem;
                border: 2px solid #46A017;
                border-radius: 10px;
                background-color: #1c1e21;
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
                color: #f0f0f0;
            }
            </style>
        """, unsafe_allow_html=True)
    @staticmethod
    def show_navigation():
        menu_items = {
            'Home': '🏠',
            'Login': '🔑',
            'Sign Up': '📝',
            'Enter Details': '📋',
            'Generate Plan': '🍽️',
            'View Plans': '📊',
            'Profile': '👤'
        }

        # Only show certain menu items when logged in
        if 'user_id' in st.session_state:
            del menu_items['Login']
            del menu_items['Sign Up']
        else:
            menu_items = {k: v for k, v in menu_items.items() 
                if k in ['Home', 'Login', 'Sign Up']}

        col1, col2, *cols = st.columns([2, 8, *[1]*len(menu_items)])
    
        with col1:
            st.markdown("# 🥗 NutriUsher")
    
        for item, (page, icon) in zip(cols, menu_items.items()):
            with item:
                if st.button(f"{icon} {page}"):
                    st.session_state.page = page
                    st.rerun()

    @staticmethod
    def show_home():
        st.markdown(
            "<h1 style='text-align: center; color: #46A017;'>"
            "NutriUsher - Personalized Diet Planning"
            "</h1>",
            unsafe_allow_html=True
        )
        
        if NUTRITION_IMG_PATH.exists():
            st.image(str(NUTRITION_IMG_PATH), use_container_width=True)
        
        st.markdown("""
            <div style='text-align: center; margin: 20px 0;'>
                <h2>Your Personal Diet Assistant</h2>
                <p>Customized meal plans based on your:</p>
                <ul style='list-style-type: none;'>
                    <li>🎯 Health goals</li>
                    <li>🥗 Dietary preferences</li>
                    <li>💪 Fitness level</li>
                    <li>🏥 Medical conditions</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Get Started", use_container_width=True):
                st.session_state.page = "Sign Up"

# Main Application
class NutriUsher:
    def __init__(self):
        self.db_manager = DatabaseManager(DB_PATH)
        self.auth_manager = AuthManager(self.db_manager)
        self.ui = UIComponents()
        self.diet_plan_manager = DietPlanManager(self.db_manager)

    def run(self):
        """Main application entry point"""
        UIComponents.set_page_config()

        if 'page' not in st.session_state:
            st.session_state.page = 'Home'

        # Update page mapping to include all valid pages
        self.page_mapping = {
            'Home': self.page_home,
            'Login': self.page_login,
            'Sign Up': self.page_sign_up,
            'Enter Details': self.page_enter_details,
            'Generate Plan': self.page_generate_plan,
            'View Plans': self.page_view_plans,  # Changed from 'My Diet Plans'
            'Profile': self.page_profile
        }

        self.ui.show_navigation()

        try:
            current_page = st.session_state.page
            if current_page in self.page_mapping:
                self.page_mapping[current_page]()
            else:
                logger.error(f"Page not found: {current_page}")
                st.error("Page not found")
                st.session_state.page = 'Home'
                self.page_home()
        except Exception as e:
            logger.error(f"Error occurred: {str(e)}")
            st.error("An unexpected error occurred. Please try again later.")

    def page_home(self):
        """Render home page"""
        self.ui.show_home()

    def page_login(self):
        """Render login page"""
        with st.form("login_form"):
            st.markdown("<h2 style='text-align: center;'>Login</h2>", unsafe_allow_html=True)
            username = st.text_input("👤 Username")
            password = st.text_input("🔒 Password", type="password")
            submitted = st.form_submit_button("🔓 Login")

            if submitted:
                try:
                    user = self.auth_manager.login_user(username, password)
                    st.session_state.user_id = user.id
                    st.session_state.username = user.username
                    st.success("Login successful!")
                    st.session_state.page = "Enter Details"
                    st.rerun()
                except AuthenticationError as e:
                    st.error(str(e))

    def page_sign_up(self):
        """Render sign up page"""
        with st.form("signup_form"):
            st.markdown("<h2 style='text-align: center;'>Sign Up</h2>", unsafe_allow_html=True)
            email = st.text_input("📧 Email")
            username = st.text_input("👤 Username")
            password = st.text_input("🔒 Password", type="password")
            confirm_password = st.text_input("🔒 Confirm Password", type="password")
            submitted = st.form_submit_button("📈 Register")

            if submitted:
                if not all([email, username, password, confirm_password]):
                    st.error("Please fill in all fields")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    try:
                        self.auth_manager.register_user(email, username, password)
                        st.success("Registration successful! Please login.")
                        st.session_state.page = "Login"
                        st.rerun()
                    except AuthenticationError as e:
                        st.error(str(e))

    def page_enter_details(self):
        """Render user details entry page"""
        st.title("Enter Your Details")

        if 'user_id' not in st.session_state:
            st.warning("Please log in to enter your details.")
            return

        with self.db_manager.get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM user_details WHERE user_id = ?", (st.session_state.user_id,))
            existing_details = c.fetchone()

        with st.form("user_details_form"):
            gender = st.selectbox("Gender", ["Male", "Female"], 
                            index=0 if not existing_details else ["Male", "Female"].index(existing_details[2]))
            age = st.number_input("Age", min_value=1, max_value=120, 
                            value=existing_details[3] if existing_details else 30)
            height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, 
                               value=existing_details[4] if existing_details else 170.0)
            weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, 
                               value=existing_details[5] if existing_details else 70.0)
            activity_level = st.selectbox("Activity Level", 
                                    ["Sedentary", "Light", "Moderate", "Very Active", "Extra Active"],
                                    index=0 if not existing_details else ["Sedentary", "Light", "Moderate", "Very Active", "Extra Active"].index(existing_details[6]))
            condition = st.selectbox("Health Goal", 
                               ["Weight Loss", "Weight Gain", "Maintain Weight"],
                               index=0 if not existing_details else ["Weight Loss", "Weight Gain", "Maintain Weight"].index(existing_details[7]))
            diet = st.selectbox("Diet Preference", 
                          ["Veg", "Non-veg", "Vegan"],
                          index=0 if not existing_details else ["Veg", "Non-veg", "Vegan"].index(existing_details[8]))

            submitted = st.form_submit_button("Save Details")

            if submitted:
                try:
                    with self.db_manager.get_connection() as conn:
                        c = conn.cursor()
                        if existing_details:
                            c.execute("""
                            UPDATE user_details 
                            SET gender=?, age=?, height=?, weight=?, activity_level=?, condition=?, diet=?, updated_at=?
                            WHERE user_id=?
                        """, (gender, age, height, weight, activity_level, condition, diet, datetime.now(), st.session_state.user_id))
                        else:
                            c.execute("""
                            INSERT INTO user_details (user_id, gender, age, height, weight, activity_level, condition, diet, updated_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (st.session_state.user_id, gender, age, height, weight, activity_level, condition, diet, datetime.now()))
                        conn.commit()
                    st.success("Details saved successfully!")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    def page_generate_plan(self):
        """Render diet plan generation page"""
        st.title("Generate Your Personalized Diet Plan")

        if 'user_id' not in st.session_state:
            st.warning("Please log in to generate a diet plan.")
            return

        # Load user details
    def page_profile(self):
        """Render user profile page"""
        st.title("Your Profile")

        if 'user_id' not in st.session_state:
            st.warning("Please log in to view your profile.")
            return

        # Fetch user details
        with self.db_manager.get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE id = ?", (st.session_state.user_id,))
            user_data = c.fetchone()
            c.execute("SELECT * FROM user_details WHERE user_id = ?", (st.session_state.user_id,))
            user_details = c.fetchone()

        if user_data:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Account Information")
                st.write(f"Username: {user_data[2]}")
                st.write(f"Email: {user_data[1]}")
                st.write(f"Account Created: {datetime.fromisoformat(user_data[4]).strftime('%Y-%m-%d')}")

            with col2:
                st.subheader("Health Details")
                if user_details:
                    st.write(f"Age: {user_details[3]} years")
                    st.write(f"Height: {user_details[4]} cm")
                    st.write(f"Weight: {user_details[5]} kg")
                    st.write(f"Activity Level: {user_details[6]}")
                    st.write(f"Health Goal: {user_details[7]}")
                    st.write(f"Diet Preference: {user_details[8]}")

                else:
                    st.warning("No health details found. Please enter your details.")
                    if st.button("Enter Details"):
                        st.session_state.page = "Enter Details"
                        st.rerun()

            # Account Management section
            st.subheader("Account Management")
            col3, col4 = st.columns(2)
        
            with col3:
                if st.button("Delete Account", type="secondary"):
                    st.warning("Are you sure you want to delete your account? This action cannot be undone.")
                    if st.button("Yes, Delete My Account", key="confirm_delete"):
                        try:
                            with self.db_manager.get_connection() as conn:
                                c = conn.cursor()
                                # Delete user details first (foreign key constraint)
                                c.execute("DELETE FROM user_details WHERE user_id = ?", (st.session_state.user_id,))
                                # Delete diet plans
                                c.execute("DELETE FROM diet_plans WHERE user_id = ?", (st.session_state.user_id,))
                                # Delete user
                                c.execute("DELETE FROM users WHERE id = ?", (st.session_state.user_id,))
                                conn.commit()
                            # Clear session and redirect to home
                            st.session_state.clear()
                            st.success("Account deleted successfully.")
                            st.session_state.page = "Home"
                            st.rerun()
                        except Exception as e:
                            logger.error(f"Error deleting account: {str(e)}")
                            st.error("Failed to delete account. Please try again.")

            with col4:
                if st.button("Logout", type="primary"):
                    # Clear session state
                    st.session_state.clear()
                    st.success("You have been logged out successfully.")
                    # Redirect to home page
                    st.session_state.page = "Home"
                    st.rerun()

        else:
            st.error("User data not found.")
    def page_view_plans(self):
        """Render page to view saved diet plans"""
        st.title("Your Saved Diet Plans")

        if 'user_id' not in st.session_state:
            st.warning("Please log in to view your diet plans.")
            return

        plans = self.diet_plan_manager.get_user_plans(st.session_state.user_id)

        if not plans:
            st.info("You haven't saved any diet plans yet.")
            if st.button("Generate New Plan"):
                st.session_state.page = "Generate Plan"
                st.rerun()
        else:
            for plan in plans:
                with st.expander(f"Diet Plan - {plan.created_at.strftime('%Y-%m-%d %H:%M')}"):
                    # Convert string representation of plan back to dictionary
                    try:
                        plan_dict = eval(plan.plan)  # Note: Using eval for demonstration; in production, use proper serialization
                        for day, daily_meals in plan_dict.items():
                            st.subheader(day)
                            for meal_type, meals in daily_meals.__dict__.items():
                                st.write(f"**{meal_type.capitalize()}:**")
                                for meal in meals:
                                    st.write(f"- {meal.food}")
                                    st.write(f"  Calories: {meal.calories} kcal")
                                    st.write(f"  Nutrients: Carbs: {meal.carbs}g, Proteins: {meal.proteins}g, Fats: {meal.fats}g")
                                st.write("")
                    except Exception as e:
                        logger.error(f"Error displaying plan: {str(e)}")
                        st.error("Error displaying this plan")

    def page_profile(self):
        """Render user profile page"""
        st.title("Your Profile")

        if 'user_id' not in st.session_state:
            st.warning("Please log in to view your profile.")
            return

    # Fetch user details
        with self.db_manager.get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM users WHERE id = ?", (st.session_state.user_id,))
            user_data = c.fetchone()
            c.execute("SELECT * FROM user_details WHERE user_id = ?", (st.session_state.user_id,))
            user_details = c.fetchone()

        if user_data:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Account Information")
                st.write(f"Username: {user_data[2]}")
                st.write(f"Email: {user_data[1]}")
                st.write(f"Account Created: {datetime.fromisoformat(user_data[4]).strftime('%Y-%m-%d')}")

            with col2:
                st.subheader("Health Details")
                if user_details:
                    st.write(f"Age: {user_details[3]} years")
                    st.write(f"Height: {user_details[4]} cm")
                    st.write(f"Weight: {user_details[5]} kg")
                    st.write(f"Activity Level: {user_details[6]}")
                    st.write(f"Health Goal: {user_details[7]}")
                    st.write(f"Diet Preference: {user_details[8]}")
                    st.write(f"Allergies: {user_details[9] if len(user_details) > 9 and user_details[9] else 'None'}")
                else:
                    st.warning("No health details found. Please enter your details.")
                    if st.button("Enter Details "):
                        st.session_state.page = "Enter Details"
                        st.rerun()

        # Account Management section with better styling
            st.markdown("---")  # Add a separator line
            st.subheader("Account Management")
        
        # Use columns for button layout
            col_space1, col_logout, col_middle, col_delete, col_space2 = st.columns([1, 2, 1, 2, 1])
        
            with col_logout:
                if st.button("🚪 Logout", type="primary", use_container_width=True):
                    st.session_state.clear()
                    st.success("You have been logged out successfully.")
                    st.session_state.page = "Home"
                    st.rerun()
        
            with col_delete:
                if st.button("🗑️ Delete Account", type="secondary", use_container_width=True):
                    st.warning("Are you sure you want to delete your account? This action cannot be undone.")
                    if st.button("Yes, Delete My Account", key="confirm_delete", type="secondary"):
                        try:
                            with self.db_manager.get_connection() as conn:
                                c = conn.cursor()
                                # Delete user details first (foreign key constraint)
                                c.execute("DELETE FROM user_details WHERE user_id = ?", (st.session_state.user_id,))
                                # Delete diet plans if the table exists
                                c.execute("DELETE FROM diet_plans WHERE user_id = ?", (st.session_state.user_id,))
                                # Delete user
                                c.execute("DELETE FROM users WHERE id = ?", (st.session_state.user_id,))
                                conn.commit()
                            st.session_state.clear()
                            st.success("Account deleted successfully.")
                            st.session_state.page = "Home"
                            st.rerun()
                        except Exception as e:
                            logger.error(f"Error deleting account: {str(e)}")
                            st.error("Failed to delete account. Please try again.")
        else:
            st.error("User data not found.")

class DietPlanManager:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        
    def load_food_data(self) -> pd.DataFrame:
        """Load food data from the database or CSV file"""
        try:
            df = pd.read_csv('food.csv')
            # Ensure required columns exist, add if missing
            required_columns = ['Food', 'Type', 'Allergens', 'Sugars', 'Carbs', 
                              'Proteins', 'Fats', 'Calories', 'Sodium']
            for col in required_columns:
                if col not in df.columns:
                    df[col] = ''  # Add empty column if missing
            return df
        except FileNotFoundError:
            st.error("Food database not found. Please ensure food.csv exists in the application directory.")
            return pd.DataFrame(columns=['Food', 'Type', 'Allergens', 'Sugars', 'Carbs', 
                                      'Proteins', 'Fats', 'Calories', 'Sodium'])

    def calculate_daily_requirements(self, user_details: UserDetails) -> Dict[str, float]:
        """Calculate daily nutritional requirements based on user details"""
        # Basic BMR calculation using Harris-Benedict equation
        if user_details.gender.lower() == 'male':
            bmr = 88.362 + (13.397 * user_details.weight) + (4.799 * user_details.height) - (5.677 * user_details.age)
        else:
            bmr = 447.593 + (9.247 * user_details.weight) + (3.098 * user_details.height) - (4.330 * user_details.age)

        # Activity level multipliers
        activity_multipliers = {
            'sedentary': 1.2,
            'light': 1.375,
            'moderate': 1.55,
            'very_active': 1.725,
            'extra_active': 1.9
        }
        
        multiplier = activity_multipliers.get(user_details.activity_level.lower(), 1.2)
        daily_calories = bmr * multiplier

        # Adjust based on condition
        if user_details.condition.lower() == 'weight loss':
            daily_calories *= 0.85
        elif user_details.condition.lower() == 'weight gain':
            daily_calories *= 1.15

        return {
            'calories': daily_calories,
            'proteins': (daily_calories * 0.25) / 4,  # 25% of calories from protein
            'carbs': (daily_calories * 0.5) / 4,      # 50% of calories from carbs
            'fats': (daily_calories * 0.25) / 9       # 25% of calories from fats
        }

    def generate_meal_plan(self, user_details: UserDetails, food_data: pd.DataFrame) -> Dict[str, DailyMeals]:
        """Generate a weekly meal plan based on user details and requirements"""
        requirements = self.calculate_daily_requirements(user_details)
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        # Ensure food_data has required columns
        if food_data.empty or 'Type' not in food_data.columns:
            st.error("Food data is missing required columns. Please check the food database.")
            return {}

        # Filter food data based on diet preference
        if user_details.diet.lower() == 'veg':
            food_data = food_data[food_data['Type'].str.lower() == 'vegetarian']
        elif user_details.diet.lower() == 'vegan':
            food_data = food_data[food_data['Type'].str.lower() == 'vegan']

        # If no foods match the diet preference, use all foods
        if food_data.empty:
            st.warning("No foods found for the selected diet preference. Using all available foods.")
            food_data = self.load_food_data()

        # Filter out allergens
        if user_details.allergies:
            for allergen in user_details.allergies.split(','):
                food_data = food_data[~food_data['Allergens'].str.contains(allergen.strip(), na=False, case=False)]

        weekly_plan = {}
        for day in days:
            # Distribute calories across meals
            breakfast_cals = requirements['calories'] * 0.3
            lunch_cals = requirements['calories'] * 0.4
            dinner_cals = requirements['calories'] * 0.3

            # Generate meals for each day
            breakfast = self.generate_meal(food_data, breakfast_cals, 'Breakfast')
            lunch = self.generate_meal(food_data, lunch_cals, 'Lunch')
            dinner = self.generate_meal(food_data, dinner_cals, 'Dinner')

            weekly_plan[day] = DailyMeals(
                breakfast=breakfast,
                lunch=lunch,
                dinner=dinner
            )

        return weekly_plan


    def generate_meal(self, food_data: pd.DataFrame, target_calories: float, meal_type: str) -> List[Meal]:
        """Generate a single meal based on target calories"""
        meal_items = []
        current_calories = 0
        
        while current_calories < target_calories and len(meal_items) < 4:
            item = food_data.sample(n=1).iloc[0]
            meal = Meal(
                food=item['Food'],
                sugars=item['Sugars'],
                carbs=item['Carbs'],
                proteins=item['Proteins'],
                fats=item['Fats'],
                calories=item['Calories'],
                sodium=item['Sodium']
            )
            meal_items.append(meal)
            current_calories += meal.calories

        return meal_items

    def save_diet_plan(self, user_id: int, plan: Dict[str, DailyMeals]) -> None:
        """Save the generated diet plan to the database"""
        plan_json = str(plan)  # Convert plan to string representation
        with self.db_manager.get_connection() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO diet_plans (user_id, plan, created_at) VALUES (?, ?, ?)",
                (user_id, plan_json, datetime.now())
            )
            conn.commit()

    def get_user_plans(self, user_id: int) -> List[DietPlan]:
        """Retrieve all diet plans for a user"""
        with self.db_manager.get_connection() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM diet_plans WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
            return [DietPlan(id=row[0], user_id=row[1], plan=row[2], created_at=datetime.fromisoformat(row[3]))
                   for row in c.fetchall()]

if __name__ == "__main__":
    app = NutriUsher()
    app.run()
