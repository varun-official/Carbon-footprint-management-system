# Import Libraries
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, confusion_matrix, accuracy_score, classification_report


# CO Sub-Index calculation


def get_CO_subindex(x):
    if x <= 1:
        return x * 50 / 1
    elif x <= 2:
        return 50 + (x - 1) * 50 / 1
    elif x <= 10:
        return 100 + (x - 2) * 100 / 8
    elif x <= 17:
        return 200 + (x - 10) * 100 / 7
    elif x <= 34:
        return 300 + (x - 17) * 100 / 17
    elif x > 34:
        return 400 + (x - 34) * 100 / 17
    else:
        return 0



# CO2 Sub-Index calculation


def get_CO2_subindex(x):
    if x >= 250 and x <= 400:
        return x * 50 / 1
    elif x <= 1000:
        return 50 + (x - 400) * 50 / 600
    elif x <= 1500:
        return 100 + (x - 1000) * 100 / 500
    elif x <= 2000:
        return 200 + (x - 1500) * 100 / 500
    elif x <= 5000:
        return 300 + (x - 2000) * 100 / 3000
    elif x > 5000:
        return 400 + (x - 5000) * 100 / 3000
    else:
        return 0



# Humidity Sub-Index calculation


def get_Humidity_subindex(x):
    if x <= 40:
        return x * 50 / 1
    elif x <= 50:
        return 50 + (x - 40) * 50 / 10
    elif x <= 60:
        return 100 + (x - 50) * 100 / 10
    elif x <= 70:
        return 200 + (x - 60) * 100 / 10
    elif x <= 80:
        return 300 + (x - 70) * 100 / 10
    elif x > 80:
        return 400 + (x - 80) * 100 / 10
    else:
        return 0



# Temperature Sub-Index calculation


def get_Temperature_subindex(x):
    if x <= 24:
        return x * 50 / 1
    elif x <= 28:
        return 50 + (x - 24) * 50 / 4
    elif x <= 32:
        return 100 + (x - 28) * 100 / 4
    elif x <= 36:
        return 200 + (x - 32) * 100 / 4
    elif x <= 40:
        return 300 + (x - 36) * 100 / 4
    elif x > 40:
        return 400 + (x - 40) * 100 / 4
    else:
        return 0


# AQI bucketing


def get_AQI_bucket(x):
    if x <= 50:
        return "Good"
    elif x <= 100:
        return "Satisfactory"
    elif x <= 200:
        return "Moderate"
    elif x <= 300:
        return "Poor"
    elif x <= 400:
        return "Very Poor"
    elif x > 400:
        return "Severe"
    else:
        return np.NaN


def get_AQI_bucket_desc(x):
    if x <= 50:
        return "Minimal Impact"
    elif x <= 100:
        return "Minor Breathing Discomfort To Sensitive People"
    elif x <= 200:
        return "Minor Breathing Discomfort To People With Lung, Heart Disease, Children And Other Adults"
    elif x <= 300:
        return "Breathing Discomfort To People On Prolonged Exposure"
    elif x <= 400:
        return "Respiratory Illness To The People On Prolonged Exposure"
    elif x > 400:
        return "Respiratory Effects On Even Healthy People"
    else:
        return np.NaN


def predictAQI(dataset):
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    imputer = imputer.fit(dataset.iloc[:])
    dataset.iloc[:] = imputer.transform(dataset.iloc[:])
    dataset["CO2_SubIndex"] = dataset["CO2"].apply(lambda x: get_CO2_subindex(x))

    dataset["CO_SubIndex"] = dataset["CO"].apply(lambda x: get_CO_subindex(x))

    dataset["Humidity_SubIndex"] = dataset["Humidity"].apply(
        lambda x: get_Humidity_subindex(x))

    dataset["Temperature_SubIndex"] = dataset["Temperature"].apply(
        lambda x: get_Temperature_subindex(x))

    dataset["Checks"] = (dataset["CO_SubIndex"] > 0).astype(int) + (dataset["CO2_SubIndex"] > 0).astype(
        int) + (dataset["Humidity_SubIndex"] > 0).astype(int) + (dataset["Temperature_SubIndex"] > 0).astype(int)

    dataset["AQI"] = round(dataset[["CO_SubIndex", "CO2_SubIndex",
                        "Humidity_SubIndex", "Temperature_SubIndex"]].max(axis=1))
    dataset.loc[dataset["CO_SubIndex"] + dataset["CO2_SubIndex"] +
                dataset["Humidity_SubIndex"] + dataset["Temperature_SubIndex"] <= 0, "AQI"] = np.NaN

    dataset["AQI_Bucket"] = dataset["AQI"].apply(lambda x: get_AQI_bucket(x))
    dataset["AQI_Description"] = dataset["AQI"].apply(
        lambda x: get_AQI_bucket_desc(x))

    x = dataset.iloc[:, :-2]
    y = dataset.iloc[:, -2]

    label_encoder_y = preprocessing.LabelEncoder()
    dataset.iloc[:, -2] = label_encoder_y.fit_transform(dataset.iloc[:, -2])

    le_name_mapping = dict(zip(label_encoder_y.classes_,
                        label_encoder_y.transform(label_encoder_y.classes_)))


    X_train, X_test, y_train, y_test = train_test_split(
        dataset.iloc[:, :-2], dataset.iloc[:, -2], test_size=0.30, random_state=0)


    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    rfc = RandomForestClassifier(n_estimators=100, criterion="gini",
                                max_depth=None, min_samples_split=2, random_state=0)
    rfc.fit(X_train, y_train)

    y_pred = rfc.predict(X_test)

    confusion_matrix(y_test, y_pred)

    cross_val_score(rfc, X_train, y_train, cv=10)
    return dataset[['AQI','AQI_Description']].tail(5).values

