import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load the food data (food.csv)
food_data = pd.read_csv('/food.csv')
# Load the user_nutritional_data.csv file
user_nutritional_data = pd.read_csv('/user_nutritional_data.csv')

print("Food Data:")
print(food_data.head())

print("\nUser Nutritional Data:")
print(user_nutritional_data.head())

# Set the option to display all columns
pd.options.display.max_columns = None

# Check for missing values
print(food_data.isnull().sum())
print(user_nutritional_data.isnull().sum())

food_data = food_data.dropna()
user_nutritional_data = user_nutritional_data.dropna()
print(food_data)
print(user_nutritional_data)
print(food_data.describe())
print(user_nutritional_data.describe())

food_data.fillna(0, inplace=True)

Q1 = user_nutritional_data.quantile(0.25)
Q3 = user_nutritional_data.quantile(0.75)
IQR = Q3 - Q1
user_nutritional_data = user_nutritional_data[~((user_nutritional_data < (Q1 - 1.5 * IQR)) | (user_nutritional_data > (Q3 + 1.5 * IQR))).any(axis=1)]

user_nutritional_data['Gender'] = user_nutritional_data['Gender'].astype('category')
user_nutritional_data['Age'] = user_nutritional_data['Age'].astype(int)  # Assuming age is an integer
user_nutritional_data['Height'] = user_nutritional_data['Height'].astype(float)
user_nutritional_data['Weight'] = user_nutritional_data['Weight'].astype(float)
user_nutritional_data['Calories'] = user_nutritional_data['Calories'].astype(float)

# Step 1: Define feature columns (X) and target column (y)
# Example: Predict 'Calories' based on other features like 'Age', 'Weight', 'Height', 'Gender'
X = user_nutritional_data[['Gender', 'Age', 'Height', 'Weight', 'BMR', 'Carbs', 'Proteins', 'Fats']]
y = user_nutritional_data['Calories']

# Step 2: Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=user_nutritional_data['Gender'])

# Step 3: Standardize/normalize features (optional but often useful for some models like neural networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train a RandomForestRegressor (as an example model)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Step 5: Make predictions and evaluate the model
y_pred = model.predict(X_test_scaled)

# Step 6: Evaluate the model using metrics like RMSE and R2 score
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2 Score): {r2}")

model = Sequential()

# Input layer and first hidden layer
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # 64 neurons, input_dim matches the number of features

# Second hidden layer
model.add(Dense(32, activation='relu'))  # 32 neurons in this hidden layer

# Output layer
model.add(Dense(1))  # One neuron since this is a regression task (predicting Calories)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Summary of the model
model.summary()

# Train the model
history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=100, batch_size=32)

# Evaluate the model on the test set
loss = model.evaluate(X_test_scaled, y_test)
print(f"Test Loss (Mean Squared Error): {loss}")

# Split the data into features (X) and target (y) for prediction (e.g., predicting 'Calories')
X = user_nutritional_data[['Gender', 'Age', 'Height', 'Weight', 'BMR', 'Carbs', 'Proteins', 'Fats']]
y = user_nutritional_data['Calories']

# Split into training and testing datasets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=user_nutritional_data['Gender'])

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the neural network model
model = Sequential()

# Input and first hidden layer
model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))  # 64 neurons, input_dim matches the number of features

# Second hidden layer
model.add(Dense(32, activation='relu'))  # 32 neurons in this hidden layer

# Output layer (for regression task)
model.add(Dense(1))  # One neuron since this is a regression task (predicting Calories)

# Compile the model with the Adam optimizer and mean squared error loss
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Display the model structure
model.summary()

# Train the model (Preliminary testing)
history = model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=10, batch_size=32)

# Evaluate the model on the test dataset
test_loss = model.evaluate(X_test_scaled, y_test)
print(f"Test Loss (Mean Squared Error): {test_loss}")

# Plot training and validation loss over epochs (Optional: for visual analysis)
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error Loss')
plt.legend()
plt.show()

# Rename 'Food_items' to 'Food' if it contains food names
food_data = food_data.rename(columns={'Food_items': 'Food'})  # Assuming 'Food_items' contains food names

# Adding a 'Category' column based on existing columns (Breakfast, Lunch, Dinner)
food_data['Category'] = food_data[['Breakfast', 'Lunch', 'Dinner']].idxmax(axis=1)

Gender = input("Enter Gender (Male/Female): ")
Age = int(input("Enter Age: "))
Height = float(input("Enter Height (in cm): "))
Weight = float(input("Enter Weight (in kg): "))
BMR = int(input("Enter Basal Metabolic Rate: "))
Condition = input("Enter Conditions Listed Below \nWeight Loss \nWeight Gain \nDiabetes \nHeart Condition \nBlood Pressure): ")
Diet = input("Enter Diet (Veg/Non-Veg): ")

# Calculate BMI
# Formula: BMI = weight (kg) / (height (m))^2
height_in_meters = Height / 100  # Convert cm to m
BMI = round(Weight / (height_in_meters ** 2), 2)

user_input = {
    'Gender': Gender,
    'Age': Age,
    'Height': Height,
    'Weight': Weight,
    'BMR': BMR,
    'Condition': Condition,
    'Diet': Diet,
    'BMI': BMI  # Add calculated BMI to the dictionary
}

print(f"Calculated BMI: {BMI}")
print("User input:", user_input)

# Function to recommend food based on health condition and diet preference
def recommend_food(user_input, food_data, meal_category):
    condition = user_input['Condition']
    diet_preference = user_input['Diet']

    # Filter by meal category
    category_filtered = food_data[food_data['Category'] == meal_category]

    # Apply diet preference filter (assuming 'Veg' column exists: 1 for veg, 0 for non-veg)
    if diet_preference == 'Veg':
        category_filtered = category_filtered[category_filtered['Diet'] == 0]
    elif diet_preference == 'Non-Veg':
        category_filtered = category_filtered[category_filtered['Diet'] == 1]

    # Condition-based filters
    if condition == 'Diabetes':
        recommended_foods = category_filtered[category_filtered['Sugars'] < 5]
    elif condition == 'Heart Condition':
        recommended_foods = category_filtered[category_filtered['Fats'] < 10]
    elif condition == 'Weight Loss':
        recommended_foods = category_filtered[category_filtered['Calories'] < 400]  # Low-calorie foods
    elif condition == 'Weight Gain':
        recommended_foods = category_filtered[category_filtered['Calories'] > 600]  # High-calorie foods
    elif condition == 'Blood Pressure':
        if 'Sodium' in category_filtered.columns:
            recommended_foods = category_filtered[category_filtered['Sodium'] < 150]  # Low-sodium foods
        else:
            recommended_foods = pd.DataFrame()  # No recommendation if 'Sodium' data isn't available
    else:
        recommended_foods = category_filtered[category_filtered['Calories'] <= user_input['BMR'] / 3]

    return recommended_foods

# Function to recommend meals for an entire week (3 times a day, with at least 3 items per meal)
def recommend_weekly_meals(user_input, food_data):
    meals = ['Breakfast', 'Lunch', 'Dinner']
    weekly_menu = {}

    # Loop through 7 days of the week
    for day in range(1, 8):
        day_menu = {}

        # Recommend meals for breakfast, lunch, and dinner
        for meal in meals:
            meal_recommendations = recommend_food(user_input, food_data, meal)

            if not meal_recommendations.empty:
                # Select up to 3 items randomly for each meal (or as many as available)
                recommended_meals = meal_recommendations.sample(n=min(3, len(meal_recommendations)), random_state=day)
                day_menu[meal] = [
                    {
                        'Food': meal_item['Food'],
                        'Sugars': meal_item.get('Sugars', 'N/A'),
                        'Carbs': meal_item.get('Carbs', 'N/A'),
                        'Proteins': meal_item.get('Proteins', 'N/A'),
                        'Fats': meal_item.get('Fats', 'N/A'),
                        'Calories': meal_item.get('Calories', 'N/A'),
                        'Sodium': meal_item.get('Sodium', 'N/A') if 'Sodium' in meal_item else 'N/A'
                    }
                    for _, meal_item in recommended_meals.iterrows()
                ]
            else:
                day_menu[meal] = 'No suitable meal found'

        # Add the day menu to the weekly menu
        weekly_menu[f'Day {day}'] = day_menu

    return weekly_menu

# Get recommendations for the week
weekly_meals = recommend_weekly_meals(user_input, food_data)

# Display the weekly meal plan
for day, meals in weekly_meals.items():
    print(f"\n{day}'s Meal Plan:")
    for meal, details in meals.items():
        if isinstance(details, list):
            print(f"{meal}:")
            for idx, item in enumerate(details, start=1):
                print(f"  {idx}. {item['Food']} (Sugars: {item['Sugars']}g, Carbs: {item['Carbs']}g, Proteins: {item['Proteins']}g, Fats: {item['Fats']}g, Calories: {item['Calories']} kcal, Sodium: {item['Sodium']}mg)")
        else:
            print(f"{meal}: {details}")