from django.shortcuts import render
#from django.http import HttpResponse  # should be placed with other imports


# Create your views here.

# Module all requirment
import requests # This libarary helps us to feach data from API
import pandas as pd # for handaling and analysing data
import numpy as np #for numberical operation
from sklearn.model_selection import train_test_split #to split data training and testing sets
from sklearn.preprocessing import LabelEncoder # to convert categorical data into numerical values
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # model for classification and regration 
from sklearn.metrics import mean_squared_error #to measure the accuracy our prediction
from datetime import datetime, timedelta # to handle date and time
import pytz
import os

# ApI Call
API_KEY = '7776198be7c14f2911c050555dbec626' # My Api
BASE_URL = 'https://api.openweathermap.org/data/2.5/' # base url for making API request

# 3 Featch Current weather Data
def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    # Check for error in the response
    
    return {
        'city': data['name'],
        'current_temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'temp_min': round(data['main']['temp_min']),
        'temp_max': round(data['main']['temp_max']),
        'humidity': round(data['main']['humidity']),
        'description': data['weather'][0]['description'],
        'country': data['sys']['country'],
        'wind_gust_dir': data['wind']['deg'],
        'pressure': data['main']['pressure'],
        'wind_Gust_Speed': data['wind']['speed'],
        'clouds': data['clouds']['all'],
        'Visibility': data['visibility']
    }

# 4 Read Historical Data
def read_historical_data(filename):
    df = pd.read_csv(filename) # load csv file dataFrame
    df = df.dropna() #remove row with missing values
    df = df.drop_duplicates()
    return df

#5 Preparing Data
def prepare_data(data):
    le = LabelEncoder() #create a LabelEncoder instance
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])

    # Define the featue variable and target variable
    x = data[['MinTemp','MaxTemp','WindGustDir','WindGustSpeed','Humidity','Pressure','Temp']]
    y = data['RainTomorrow']# target variable

    return x, y, le #return feature variable, target variable and the lable encoder

# 6.Rain Random Forest 
def train_rain_model(x,y):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train,y_train) #train the model
    y_pred = model.predict(x_test) #to make predicttion on test set
    print("Mean squared Error for Rain Model")
    print(mean_squared_error(y_test,y_pred))

    return model 

# 7. Preparing Regation Data
def prepare_regration_data(data,feature):
    x,y = [],[] #initalize list for the feature and target values
    for i in range(len(data)-1):
        x.append(data[feature].iloc[i])

        y.append(data[feature].iloc[i+1])
    x = np.array(x).reshape(-1,1)
    y = np.array(y)
    return x,y

#8. train regration Model
def train_regration_model(x,y):
    model = RandomForestRegressor(n_estimators = 100, random_state=42)
    model.fit(x,y)
    return model

# 9. Predict Feature
def predict_future(model,current_value):
    predictions = [current_value]

    for i in range(5):
        next_value = model.predict(np.array([[predictions[-1]]]))

        predictions.append(next_value[0])

    return predictions[1:]

#10. Weather Analytic Function
def weather_view(request):
    if request.method == 'POST':
        city = request.POST.get('city')
        current_weather = get_current_weather(city)

        #Load historical data
        csv_path = os.path.join('E:\\ML Project\\weather.csv')
        historical_data = read_historical_data(csv_path)

        #prepare and train the rain prediction model

        x,y,le = prepare_data(historical_data)

        rain_model = train_rain_model(x,y)

        #map wind direction to campass points
        wind_deg = current_weather['wind_gust_dir']% 360
        compass_points = [
        ("N",0,11.25),("NNE",11.25,33.75),("NE",33.75,56.25),
        ("ENE",56.25,78.75),("E",78.75,101.25),("ESE",101.25,123.75),
        ("SE",123.75,146.25),("SSE",146.25,168.75),("S",168.75,191.25),
        ("SSW",191.25,213.75),("SW",213.75,236.25),("WSW",236.25,258.75),
        ("W",258.75,281.25),("WNW",285.25,303.75),("NW",303.75,326.25),
        ("NNW",326.25,348.75)
        ]

        compass_direction = next(point for point, start,end in compass_points if start <= wind_deg < end)
        
        if compass_direction in le.classes_:
            compass_direction_encoded = le.transform([compass_direction])[0]
        else:
            compass_direction_encoded = 0  # or use mode class


        current_data = {
            'MinTemp':current_weather['temp_min'],
            'MaxTemp':current_weather['temp_max'],
            #'WindGustDir': compass_direction_encoded['wind_gust_dir'],
            'WindGustDir': compass_direction_encoded,
            'WindGustSpeed':current_weather['wind_Gust_Speed'],
            'Humidity': current_weather['humidity'],
            'Pressure':current_weather['pressure'],
            'Temp': current_weather['current_temp'],
            
        }
        current_df = pd.DataFrame([current_data])

        #rain Prediction
        rain_prediction = rain_model.predict(current_df)[0]

        #prepare regration model for temperature and humidity

        x_temp,y_temp = prepare_regration_data(historical_data,'Temp')

        x_hum,y_hum = prepare_regration_data(historical_data,'Humidity')

        temp_model = train_regration_model(x_temp,y_temp)

        hum_model = train_regration_model(x_hum,y_hum)

        #predict future temperature and humidity

        future_temp = predict_future(temp_model,current_weather['temp_min'])

        future_humidity = predict_future(hum_model,current_weather['humidity'])

        #Prepare time for future prediction

        timezone = pytz.timezone('Asia/Kolkata')
        now = datetime.now(timezone)
        next_hour = now + timedelta(hours=1)

        future_times = [(next_hour + timedelta(hours=i)).strftime("%H:00") for i in range(5)]

        # Store each value seprate

        time1, time2,time3, time4, time5 = future_times
        temp1, temp2, temp3, temp4, temp5 = future_temp
        hum1, hum2, hum3, hum4, hum5 = future_humidity
        

        # pass data to temperature
        context = {
            'location':city,
            'current_temp': current_weather['current_temp'],
            'MinTemp':current_weather['temp_min'],
            'MaxTemp': current_weather['temp_max'],
            'feels_like': current_weather['feels_like'],
            'humidity': current_weather['humidity'],
            'clouds': current_weather['clouds'],
            'description':current_weather['description'],
            'city':current_weather['city'],
            'country':current_weather['country'],

            'time': datetime.now(),
            'date': datetime.now().strftime("%B %d, %y"),

            'wind': current_weather['wind_Gust_Speed'],
            'pressure': current_weather['pressure'],
            'Visibility': current_weather['Visibility'],

            'time1': time1,
            'time2': time2,
            'time3': time3,
            'time4': time4,
            'time5': time5,

            'temp1': f"{round(temp1,1)}",
            'temp2': f"{round(temp2,1)}",
            'temp3': f"{round(temp3,1)}",
            'temp4': f"{round(temp4,1)}",
            'temp5': f"{round(temp5,1)}",

            'hum1': f"{round(hum1, 1)}",
            'hum2': f"{round(hum2, 1)}",
            'hum3': f"{round(hum3, 1)}",
            'hum4': f"{round(hum4, 1)}",
            'hum5': f"{round(hum5, 1)}",
            
        }
        return render(request, 'weather.html',context)
    return render(request, 'weather.html')



#weather_view()


#from django.http import HttpResponse

#def weather_view(request):
   # return HttpResponse('<h1> Weather Prediction App </h1>')
