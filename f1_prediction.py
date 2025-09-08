import pandas as pd
import joblib

# Load trained model
best_rf = joblib.load('model/random_forest_model.pkl')

# Load encoders
le_driver = joblib.load('model/label_encoder_driver.pkl')
le_constructor = joblib.load('model/label_encoder_constructor.pkl')
le_gp = joblib.load('model/label_encoder_gp.pkl')

print("✅ Model and Encoders loaded successfully!")

#Load cleaned dataset
cleaned_data=pd.read_csv(r'/home/rachit/cleaned_data.csv')
#Predictor
print("\n--- Predict Finishing Position ---")

#Get unique values from the dataset
available_drivers = cleaned_data['driver'].unique()
available_constructors = cleaned_data['constructor'].unique()
available_gps = cleaned_data['GP_name'].unique()

# Display dropdown-like options
print("\nAvailable Drivers:")
for i, d in enumerate(available_drivers):
    print(f"{i}. {d}")
driver_idx = int(input("Select Driver (enter index): "))
input_driver = available_drivers[driver_idx]

print("\nAvailable Constructors:")
for i, c in enumerate(available_constructors):
    print(f"{i}. {c}")
constructor_idx = int(input("Select Constructor (enter index): "))
input_constructor = available_constructors[constructor_idx]

print("\nAvailable GP Names:")
for i, g in enumerate(available_gps):
    print(f"{i}. {g}")
gp_idx = int(input("Select Grand Prix (enter index): "))
input_gp_name = available_gps[gp_idx]

input_year = int(input("\nEnter Year: "))
input_qualifying_pos = int(input("Enter Qualifying Position: "))

# Encode categorical values
input_driver_encoded = le_driver.transform([input_driver])[0]
input_constructor_encoded = le_constructor.transform([input_constructor])[0]
input_gp_name_encoded = le_gp.transform([input_gp_name])[0]

# Retrieve average driver_confidence for selected driver
driver_conf = cleaned_data[cleaned_data['driver'] == input_driver]['driver_confidence'].mean()

# Retrieve average constructor_reliability for selected constructor
constructor_reliability = cleaned_data[cleaned_data['constructor'] == input_constructor]['constructor_relaiblity'].mean()


# Prepare input DataFrame
input_df = pd.DataFrame([{
    'GP_name': input_gp_name_encoded,
    'quali_pos': input_qualifying_pos,
    'constructor': input_constructor_encoded,
    'driver': input_driver_encoded,
    'driver_confidence': driver_conf,
    'constructor_relaiblity':constructor_reliability
}])

# Predict using the trained model
predicted_category = best_rf.predict(input_df)[0]

# Map category to label
category_map = {
    1: "Top 3",
    2: "Top 10",
    3: "Finished",
}


print(f"\n🏁 Predicted Finishing Position: {category_map.get(predicted_category, 'Unknown')}")

