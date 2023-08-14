import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

df = pd.read_csv('https://raw.githubusercontent.com/rocksdxebec/data_analyst_project/main/Sleep/data/sleepdata_1.csv', delimiter=';')

st.write("""
## The Importance of Sleep

Sleep is a fundamental human need that greatly impacts both physical and mental well-being. According to the National Sleep Foundation, a consistent sleep schedule reinforces the body's natural circadian rhythm, enhancing overall sleep quality. This, in turn, promotes better hormonal balance and aids critical bodily functions like tissue repair.

Moreover, the Harvard Medical School emphasizes that sleep plays a pivotal role in cognitive processes. Lack of sleep impairs attention, alertness, problem-solving, and decision-making skills. It can also lead to mood disturbances such as irritability, anxiety, or even depression. This is largely attributed to the brain's processing and consolidation of the day's experiences during rest, facilitating memory and learning.

Additionally, chronic sleep deprivation has been linked to a range of serious health conditions. The Centers for Disease Control and Prevention (CDC) notes that it elevates the risk of chronic ailments like obesity, diabetes, cardiovascular diseases, and even a weakened immune system. Therefore, prioritizing sleep is not just about mental sharpness; it's a cornerstone of comprehensive health.

""")

st.image('https://raw.githubusercontent.com/rocksdxebec/data_analyst_project/main/Sleep/img/good_sleep.jpg', caption='Benefits of a good sleep', use_column_width=True)

st.write("""
# 1. Pre-processing

#### 1.1 Importing the CSV dataset obtained

The dataset herewith will be analyzed, cleaned and pre-processed as needed so as to make useful conclusions form therein and gain useful insights about makes a quality sleep and how best to attain it.

""")

st.dataframe(df[:10])

st.write("""

All the info regarding the dataframe is shown below:

""")

st.write(df.info())

st.write("""

All the info regarding N/A values within the dataframe is shown below:

""")

st.write(df.isna().sum())

st.write("""

All the info regarding NULL values within the dataframe is shown below:

""")

st.write(df.isnull().sum())

st.write("""
#### 1.2 Dropping the null and N/A values

The dataset has quite a few null and N/A values for the "Wake up" column, "Sleep notes" column and "Heart rate" column. They will all be dropped form the dataset and then further steps will be taken to make an analysis.


""")

df = df.dropna()

st.write("""
#### 1.3 Converting row values into useful values

Convert the value data-type of the "Time in bed" from object to float, data-type of Sleep quality from object to percentage (b/w 0 and 1)

""")

# Convert "sleep quality" to percentages
df["Sleep quality"] = df["Sleep quality"].astype(str).str.replace('%','').astype(float) / 100

# Convert "Time in bed" to hours
df["Time in bed"] = df["Time in bed"].apply(lambda x: pd.to_timedelta(x + ":00").seconds / 3600)

st.write("""
#### 1.4 Extracting Day of Week and Month from Start and End.

Extracting the Day of Week and Month from Start and End to analyze sleeping patterns as per weekdays/ weekends or across which may serve as useful information for performing further analysis.

""")

df['Start'] = pd.to_datetime(df['Start'])
df['End'] = pd.to_datetime(df['End'])

df['Day of week'] = df['Start'].dt.dayofweek
df['Month'] = df['Start'].dt.month

df['Start hour'] = df['Start'].dt.hour + (df['Start'].dt.minute / 60) + (df['Start'].dt.second / 3600)
df['End hour'] = df['End'].dt.hour + (df['End'].dt.minute / 60) + (df['End'].dt.second / 3600)

st.write("""
#### 1.5 Appropriating the "Wake Up" column data

After observing the elements in the column let's conver the emojis into practical numerical values. The subjective feeling of waking up can be valued as follows:

:) => 2

:| => 1

:( => 0

""")

df['Wake up'] = df['Wake up'].replace({':)':2, ':|':1, ':(':0}).astype('int')

st.write("""
## 2 Exploratory Data Analysis (EDA)

#### 2.1 First we find the corelation b/w all the factors

In this section we will employ a variety of visualization methids to obtain a better picture to visualize data which will also help us perform corelation analysis and help find out the most important factors which will impact the "Sleep Quality" of an individual.

""")

df_new = df.drop('Sleep Notes', axis=1)
st.write(df_new.corr()['Sleep quality'].sort_values(ascending=False))

fig = plt.figure(figsize = (15,15))
out = sns.heatmap(df_new.corr(),cmap='coolwarm')
out.set_title('Correlation b/w all the different factors')

st.pyplot(fig)

st.write("""
#### 2.2 Ploting pairplot b/w all the factors

Pairplot will give us a better idea about the distribution of all the factors affecting the quality sleep

""")

pairplot = sns.pairplot(df_new, hue='Wake up')
st.pyplot(pairplot)

st.write("""
#### 2.3 Ploting scatterplot b/w Time in bed and sleep quality

The scatterplot will give us a better visual estimation b/w the relationship of time in bed and sleep quality
""")

fig_scatter1 = plt.figure()
ax_scatter1 = fig_scatter1.gca()

scatter_1 = sns.scatterplot(data=df_new, x='Time in bed', y='Sleep quality', hue='Wake up', style='Wake up', palette="deep", ax=ax_scatter1)
plt.title("Scatter Plot: Time in bed vs. Sleep quality")
plt.xlabel("Time in bed")
plt.ylabel("Sleep quality")
st.pyplot(fig_scatter1)

st.write("""
#### 2.4 Ploting scatterplot b/w Start hour and sleep quality

The scatterplot will give us a better visual estimation b/w the relationship of Start hour and sleep quality
""")

fig_scatter2 = plt.figure()
ax_scatter2 = fig_scatter2.gca()

scatter_2 = sns.scatterplot(data=df_new, x='Start hour', y='Sleep quality', hue='Wake up', style='Wake up', palette="deep", ax=ax_scatter2)
plt.title("Scatter Plot: Start Hour vs. Sleep quality")
plt.xlabel("Start Hour")
plt.ylabel("Sleep quality")
st.pyplot(fig_scatter2)

st.write("""
#### 2.5 Ploting scatterplot b/w End hour and sleep quality

The scatterplot will give us a better visual estimation b/w the relationship of End hour and sleep quality
""")

fig_scatter3 = plt.figure()
ax_scatter3 = fig_scatter3.gca()

scatter_3 = sns.scatterplot(data=df_new, x='End hour', y='Sleep quality', hue='Wake up', style='Wake up', palette="deep", ax=ax_scatter3)
plt.title("Scatter Plot: End Hour vs. Sleep quality")
plt.xlabel("End Hour")
plt.ylabel("Sleep quality")
st.pyplot(fig_scatter3)

st.write("""
#### 2.6 Ploting scatterplot b/w Heart Rate and sleep quality

The scatterplot will give us a better visual estimation b/w the relationship of Heart and sleep quality
""")

fig_scatter4 = plt.figure()
ax_scatter4 = fig_scatter4.gca()

scatter_4 = sns.scatterplot(data=df_new, x='Heart rate', y='Sleep quality', hue='Wake up', style='Wake up', palette="deep", ax=ax_scatter4)
plt.title("Scatter Plot: Heart Rate vs. Sleep quality")
plt.xlabel("Heart Rate")
plt.ylabel("Sleep quality")
st.pyplot(fig_scatter4)

st.write("""
#### 2.7 Making a jointplot between "Time in bed" and "Sleep quality"

""")

jointplot1 = sns.jointplot(x=df['Time in bed'], y=df['Sleep quality'], data = df, kind='reg')
plt.xlabel("Time in Bed")
plt.ylabel("Sleep quality")
st.pyplot(jointplot1.fig)

st.write("""
#### 2.8 Making a jointplot between "Start hour" and "Sleep quality"

""")

jointplot2 = sns.jointplot(x=df['Start hour'], y=df['Sleep quality'], data = df, kind='hex')
plt.xlabel("Start Hour")
plt.ylabel("Sleep quality")
st.pyplot(jointplot2.fig)

st.write("""
#### 2.9 Making a jointplot between "End hour" and "Sleep quality"

""")

jointplot3 = sns.jointplot(x=df['End hour'], y=df['Sleep quality'], data = df, kind='hex')
plt.xlabel("End Hour")
plt.ylabel("Sleep quality")
st.pyplot(jointplot3.fig)

st.write("""
## 3 Predictive modelling on sleep quality

#### 3.1 We split the entire dataset into Training and Testing dataset

""")

from sklearn.model_selection import train_test_split

X = df.drop(columns=['Sleep quality', 'Start', 'End', 'Sleep Notes'])
y = df['Sleep quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

st.write("""
#### 3.2 Using LinearRegression Model

""")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

st.write("""
#### 3.3 Using RandomForest Model

""")

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

st.write("""
#### 3.4 Using Logistic Regression Model

""")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Convert 'y' into binary categories 
y_bin = y.apply(lambda x: 1 if x > 0.5 else 0)  # We assume > 50% is 'Good'

X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=0.2, random_state=42)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

y_pred_logreg = logreg.predict(X_test)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)

st.write("""
#### 3.5 Using Decision Tree Model

""")

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

st.write("""
#### Model Performance comparison

""")

st.write("Model Performance:")
st.write("-" * 30)
st.write(f"Linear Regression:")
st.write(f"MSE: {mse_lr}")
st.write(f"R2 Score: {r2_lr}")
st.write("")
st.write(f"Random Forest:")
st.write(f"MSE: {mse_rf}")
st.write(f"R2 Score: {r2_rf}")
st.write("")
st.write(f"Logistic Regression (Accuracy): {accuracy_logreg}")
st.write("")
st.write(f"Decision Tree:")
st.write(f"MSE: {mse_dt}")
st.write(f"R2 Score: {r2_dt}")

st.write("""
## From all the above Analysis and Visualization we can cleary conclude that, A Person Who Sleeps Better(i.e, in between 6 to 8 hours) and has possitive self affirmations (which may also include drinking coffee) will have a higher sleep quality. Additionally the notion of "Early to bed, early to rise; makes you wealthy and wise" turns to be true as the best sleep quality if for those who slept between 9PM and 5AM

""")
