import pandas as pd


results=pd.read_csv(r'/home/rachit/results.csv')
races=pd.read_csv(r'/home/rachit/races.csv')
drivers=pd.read_csv(r'/home/rachit/drivers.csv')
constructors=pd.read_csv(r'/home/rachit/constructors.csv')
circuit=pd.read_csv(r'/home/rachit/circuits.csv')
quali=pd.read_csv(r'/home/rachit/qualifying.csv')


df1 = pd.merge(races,results,how='inner',on=['raceId'])
df2 = pd.merge(df1,quali,how='inner',on=['raceId','driverId','constructorId'])
df3 = pd.merge(df2,drivers,how='inner',on=['driverId'])
df4 = pd.merge(df3,constructors,how='inner',on=['constructorId'])
df4=df4.drop(columns=['url'])
df5 = pd.merge(df4,circuit,how='inner',on=['circuitId'])


data=df5.drop(['raceId','circuitId','url_x','time_x','driverId','resultId','name_x','constructorId',
              'number_x','position_x','positionText', 'positionOrder','time_y', 'milliseconds', 'fastestLap',
             'rank', 'fastestLapTime', 'fastestLapSpeed','number_y','driverRef', 'number', 'code',
              'url_y', 'constructorRef','circuitRef','location', 'lat', 'lng', 'alt','url'
             ,'q1', 'q2', 'q3','round','fp1_date', 'fp1_time', 'fp2_date', 'fp2_time', 'fp3_date', 'fp3_time', 'quali_date', 'quali_time',
              'sprint_date','sprint_time', 'qualifyId'],axis=1)


data = data[(data['year']>=1990) & (data['year']<=2024)]


data.rename(columns={'name':'GP_name',
                     'position_y':'position',
                     'grid':'quali_pos',
                     'name_y':'constructor',
                     'nationality_x':'driver_nationality',
                     'nationality_y':'constructor_nationality'},
            inplace=True)
data['driver'] = data['forename']+' '+data['surname']
data['date'] = pd.to_datetime(data['date'])
data['dob'] = pd.to_datetime(data['dob'])
data.drop(['forename','surname'],axis=1,inplace=True)

data["age_at_gp_in_days"]=abs(data['dob']-data['date'])
data['age_at_gp_in_days'] = data['age_at_gp_in_days'].apply(lambda x: str(x).split(' ')[0])

#changing constructor's names
data['constructor'] = data['constructor'].apply(lambda x: 'Aston Martin' if x == 'Force India' else x)
data['constructor'] = data['constructor'].apply(lambda x: 'Aston Martin' if x == 'Racing Point' else x)
data['constructor'] = data['constructor'].apply(lambda x: 'Sauber' if x == 'Alfa Romeo' else x)
data['constructor'] = data['constructor'].apply(lambda x: 'Alpine' if x == 'Lotus F1' else x)
data['constructor'] = data['constructor'].apply(lambda x: 'Alpine' if x == 'Renault' else x)
data['constructor'] = data['constructor'].apply(lambda x: 'AlphaTauri' if x == 'Toro Rosso' else x)

#changing country's name to the same and shortening name of country
data['driver_nationality'] = data['driver_nationality'].apply(lambda x: str(x)[:3])
data['constructor_nationality'] = data['constructor_nationality'].apply(lambda x: str(x)[:3])
data['country'] = data['country'].apply(lambda x: 'Bri' if x=='UK' else x)
data['country'] = data['country'].apply(lambda x: 'Ame' if x=='USA' else x)
data['country'] = data['country'].apply(lambda x: 'Fre' if x=='Fra' else x)
data['country'] = data['country'].apply(lambda x: str(x)[:3])

#seeing if driver and constructor nationality match the race country
data['driver_home'] = data['driver_nationality'] == data['country']
data['constructor_home'] = data['constructor_nationality'] == data['country']
data['driver_home'] = data['driver_home'].apply(lambda x: int(x))
data['constructor_home'] = data['constructor_home'].apply(lambda x: int(x))


data['driver_dnf'] = data['statusId'].apply(lambda x: 1 if x in [3,4,20,29,31,41,68,73,81,97,82,104,107,130,137] else 0)
data['constructor_dnf'] = data['statusId'].apply(lambda x: 1 if x not in [3,4,20,29,31,41,68,73,81,97,82,104,107,130,137,1] else 0)

#driver confidence
dnf_by_driver = data.groupby('driver')['driver_dnf'].sum()
driver_race_entered = data.groupby('driver')['driver_dnf'].count()
driver_dnf_ratio = (dnf_by_driver/driver_race_entered)
driver_confidence = 1-driver_dnf_ratio
driver_confidence_dict = dict(zip(driver_confidence.index,driver_confidence))

#constructor reliability
dnf_by_constructor = data.groupby('constructor')['constructor_dnf'].sum()
constructor_race_entered = data.groupby('constructor')['constructor_dnf'].count()
constructor_dnf_ratio = (dnf_by_constructor/constructor_race_entered)
constructor_relaiblity = 1-constructor_dnf_ratio
constructor_relaiblity_dict = dict(zip(constructor_relaiblity.index,constructor_relaiblity))

data['driver_confidence'] = data['driver'].apply(lambda x:driver_confidence_dict[x])
data['constructor_relaiblity'] = data['constructor'].apply(lambda x:constructor_relaiblity_dict[x])

#removing retired drivers and constructors
active_constructors = ['Alpine', 'Williams', 'McLaren', 'Ferrari', 'Mercedes',
                       'AlphaTauri', 'Aston Martin', 'Sauber', 'Red Bull',
                       'Haas F1 Team']
active_drivers = ['Max Verstappen','Sergio Perez','Lewis Hamilton','George Russel','Charles Leclerc',
                  'Carlos Sainz','Lando Norris','Oscar Piastri','Fernando Alonso','Lance Stroll',
                  'Esteban Ocon','Pierre Gasly','Kevin Magnussen','Nico Hulkenberg','Alexander Albon',
                'Logan Sargeant','Valtteri Bottas','Zhou Guanyu','Liam Lawson','Yuki Tsunoda']
data['active_driver'] = data['driver'].apply(lambda x: int(x in active_drivers))
data['active_constructor'] = data['constructor'].apply(lambda x: int(x in active_constructors))

#cleaned and pre processed data
cleaned_data = data[['GP_name','quali_pos','constructor','driver','position','driver_confidence','constructor_relaiblity','active_driver','active_constructor','dob']]
#filtering data of current constructors and drivers
cleaned_data = cleaned_data[(cleaned_data['active_driver']==1)&(cleaned_data['active_constructor']==1)]

desktop_path ='/home/rachit/cleaned_data.csv'
cleaned_data.to_csv(desktop_path, index=True)

print(f"Cleaned data has been saved to: {desktop_path}")


x=cleaned_data


def position_index(x):
    if x<4:
      return 1
    if x>10:
        return 3
    else:
        return 2


constructor_name = ['Alpine', 'Williams', 'McLaren', 'Ferrari', 'Mercedes',
                       'AlphaTauri', 'Aston Martin', 'Sauber', 'Red Bull',
                       'Haas F1 Team']
for name in constructor_name:
    reliability = cleaned_data.loc[cleaned_data['constructor'] == name, 'constructor_relaiblity'].values[0]



drivers = ['Max Verstappen','Sergio Perez','Lewis Hamilton','George Russel','Charles Leclerc',
                  'Carlos Sainz','Lando Norris','Oscar Piastri','Fernando Alonso','Lance Stroll',
                  'Esteban Ocon','Pierre Gasly','Kevin Magnussen','Nico Hulkenberg','Alexander Albon',
                'Logan Sargeant','Valtteri Bottas','Zhou Guanyu','Liam Lawson','Yuki Tsunoda']


for driver in drivers:
    driver_data = cleaned_data[cleaned_data['driver'] == driver]
    if len(driver_data) > 0:
        driver_confidence_dict[driver] = driver_data.iloc[0]['driver_confidence']
    else:
        driver_confidence_dict[driver] = 0.8



from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


#Model
x = cleaned_data.copy()

le_driver = LabelEncoder()
le_constructor = LabelEncoder()
le_gp = LabelEncoder()

x['driver'] = le_driver.fit_transform(x['driver'])
x['constructor'] = le_constructor.fit_transform(x['constructor'])
x['GP_name'] = le_gp.fit_transform(x['GP_name'])


x['position_category'] = x['position'].apply(position_index)

# Drop unused columns
X = x.drop(['position','dob','active_driver','active_constructor','position_category'], axis=1)
y = x['position_category']

print(X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200,300],
    'max_depth': [None,10, 20],
    'min_samples_split': [2, 5,10],
    'min_samples_leaf': [1, 2,4],
    'max_features': ['sqrt', 'log2']
}

# Initiate the model
rf_model = RandomForestClassifier(random_state=42)

# GridSearchCV setup
grid_search = GridSearchCV(estimator=rf_model,
                           param_grid=param_grid,
                           cv=5,
                           scoring='accuracy',
                           n_jobs=-1,
                           verbose=2)

# Fit to the training data
grid_search.fit(X_train, y_train)

# Get the best parameters
print("Best Hyperparameters:", grid_search.best_params_)

# Use the best estimator
best_rf = grid_search.best_estimator_



#Training the RandomForest
best_rf.fit(X_train, y_train)

y_pred = best_rf.predict(X_test)

print("Feature names during training:", X_train.columns.tolist())


accuracy=accuracy_score(y_test, y_pred)
print(f'\n Test Set Accuracy: {accuracy:.4f}')
print("\nClassification Report:\n", classification_report(y_test, y_pred))

X.to_csv('model/X.csv', index=False)

#Saving files required for further computation
import joblib
import os

# Creating a directory for the model
os.makedirs('model', exist_ok=True)

# Save the trained model
joblib.dump(best_rf, 'model/random_forest_model.pkl')

# Save the LabelEncoders
joblib.dump(le_driver, 'model/label_encoder_driver.pkl')
joblib.dump(le_constructor, 'model/label_encoder_constructor.pkl')
joblib.dump(le_gp, 'model/label_encoder_gp.pkl')

print("✅ Model and LabelEncoders saved successfully in 'model/' folder")

# Save X_test and y_test
joblib.dump(X_test, 'model/X_test.pkl')
joblib.dump(X_train, 'model/X_train.pkl')
joblib.dump(y_pred, 'model/y_pred.pkl')
joblib.dump(y_test, 'model/y_test.pkl')

print("✅ X_test,X_train,y_test,y_pred saved!")

cv_results_df = pd.DataFrame(grid_search.cv_results_)
cv_results_df.to_csv('model/gridsearch_results.csv', index=False)

print("✅ Cross-validation results saved at 'model/gridsearch_results.csv'")













