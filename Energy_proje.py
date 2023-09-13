# Import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
import geopandas as gpd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler,LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 80)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
warnings.simplefilter(action='ignore')

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x : "%.5f" % x)
pd.set_option('display.expand_frame_repr', False)

df=pd.read_csv('ad/energyy.csv')
df.head()
# There is a missing columns for solar energy


# merging solar part
df5=pd.read_csv('ad/15 share-electricity-solar.csv')
df5.drop('Code',axis=1,inplace=True)
df5.rename(columns={'Entity':'Country'},inplace=True)

# info
df.info()
df5.info()


# Dropped regions for checking countries
df["Country"].unique()
drop_1 =['Africa', 'Asia Pacific', 'CIS','North America', 'Other Europe', 'Other South & Central America','South & Central America','World']

df2 = df[~df["Country"].isin(drop_1)]
df2.info()

# fill some na values for creating a graph/table
fill_na_value = ["Electricity from hydro (TWh)","Electricity from wind (TWh)","Electricity from solar (TWh)",
                 "Other renewables including bioenergy (TWh)","Wind (% electricity)","Renewables (% electricity)","Hydro (% electricity)"]

df2[fill_na_value]= df2[fill_na_value].fillna(df2[fill_na_value].mean())
df2.isna().sum()


df2.describe().T

# final value creating without regions
df3 = df[df["Country"].isin(drop_1)]
df3.isna().sum()

df3.groupby('Country')['Electricity from hydro (TWh)'].sum().sort_values(ascending=False)
df3.groupby('Country')['Electricity from solar (TWh)' ].sum().sort_values(ascending=False)
df3.groupby('Country')['Electricity from wind (TWh)'].sum().sort_values(ascending=False)

#Graphs for Analyzing#

#1
#piechart yenılenebılır enerjı 2020

# Filter renewable consumption dataframe for the World record in 2020
world = df3[(df3.Year == 2020) & (df3.Country == 'World')]
# Drop all columns not containing consumption data so we can create a pie chart
world.drop(['Country', 'Year','Unnamed: 0'], axis=1, inplace=True)
world.columns
pie_world = world[['Solar Generation - TWh','Wind Generation - TWh','Hydro Generation - TWh','Geo Biomass Other - TWh']]
plt.style.use("seaborn-pastel")
plt.figure(figsize = (7,7))
plt.title('Production of Worldwide Renewable Energy Consumption 2020')

plt.pie(pie_world.iloc[0], labels = ['Solar Generation - TWh','Wind Generation - TWh','Hydro Generation - TWh','Geo Biomass Other - TWh'],
        autopct='%1.1f%%', wedgeprops = {'edgecolor':'black'})
plt.show(block=True)


#2
#yenılenebılır enerjı kaynaklarının yıllara göre uretımı


renewable_consumption =["Solar Generation - TWh","Wind Generation - TWh",
                                                  "Hydro Generation - TWh",
                                                  "Biofuels Production - TWh - Total"]
energy_df = df2[renewable_consumption]
year_range = df2["Year"].isin(range(1990,2021))
energy_df = energy_df[year_range]
bar = df2.groupby('Year')[renewable_consumption].sum()
bar.reset_index(level=0, inplace=True)

bar.head()

bar.plot(x='Year', ylabel='Terrawatt Hour', kind='bar', stacked=True,figsize=(20,6),color=['black', 'red', 'green', 'blue'],
        title='Global Renewable Consumption 1990 - 2020')
plt.show()


# In[ ]:



#3
#
income_df=pd.read_csv("ad/Income_category_of_Countries.csv")
income_df.rename(columns={"Country_name": "Country"},inplace=True)
income_df.columns

df4=df2.merge(income_df,how="left",on="Country")
df4.drop("S_no",axis=1,inplace=True)
df4["Country"].unique()
df4.columns



df4=df4.merge(df5,how='left',on=['Country','Year'])
df4.loc[df4['Country'] == 'Russia', 'Income_group'] = 'Upper middle income'
df4.drop('Unnamed: 0',axis=1,inplace=True)
df4['Solar (% electricity)']=df4['Solar (% electricity)'].fillna(0)
df4.isna().sum()
df4.head()


#4


# using list comprehension
# to get string with substring
economies = ['High income',
 'Upper middle income',
 'Lower middle income',
 'Low income']
print(economies)
df_economy = df4[df4['Income_group'].isin(economies)]
df_economy = df_economy[df_economy['Year']> 1990]

targets = ['Wind (% electricity)', 'Renewables (% electricity)','Hydro (% electricity)','Solar (% electricity)']


def plot_timeseries(df):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    sns.lineplot(ax=axes[0, 0], data=df, x="Year", y=targets[0], hue="Income_group").set(title=targets[0])

    sns.lineplot(ax=axes[0, 1], data=df, x="Year", y=targets[1], hue="Income_group").set(title=targets[1])

    sns.lineplot(ax=axes[1, 0], data=df, x="Year", y=targets[2], hue="Income_group").set(title=targets[2])

    sns.lineplot(ax=axes[1, 1], data=df, x="Year", y=targets[3], hue="Income_group").set(title=targets[3])
    #plt.close()
    plt.show()
    return fig


# , ax.set_xticklabels(df["Entity"], rotation=45, fontsize=10)
plot_timeseries(df4)




#5

#Focusing on columns with data on annual percentage change in energy sources consumption starting year 1990

consumption=df3[df3['Year']>=1990]
consumption["df_geo_percentage"] =(consumption["Geothermal Capacity"]/consumption["Geothermal Capacity"].sum())*100



fig = px.bar(consumption,
             x='Country', y='Geothermal Capacity',
             color='Country',
             animation_frame='Year',
             animation_group="Country",
             #range_y=[0, 5],
             labels={'Geothermal Capacity ': 'Geothermal Capacity change, %'},
             title='Regional Changes in Geothermal Capacity from 1990, %')


fig.update_layout(showlegend=False)
fig.add_vrect(x0=11.5, x1=10.5)
# X ekseni altındaki metni özelleştirme
fig.update_xaxes(title_text='Country', title_standoff=0)  # title_standoff ile kaydırma yapabilirsiniz
plt.tight_layout()
fig.show()





#6. grafık




consumptionpersource=df2[df2['Year']>=2000]
consumptionpersource['year']=pd.to_datetime(consumptionpersource['Year'], format='%Y')
consumptionpersource['year']=consumptionpersource['year'].dt.year


fig = px.bar(consumptionpersource,
             x="Country", y=["Solar Generation - TWh",
                                                  "Wind Generation - TWh",
                                                  "Hydro Generation - TWh",
                                                  "Biofuels Production - TWh - Total"],
             title="Consumption Profiles per Countries",
             color_discrete_map={
                'Solar Generation - TWh':'black',
                'Wind Generation - TWh': '#eeee00',
                'Hydro Generation - TWh': "#B8860B",
                'Biofuels Production - TWh - Total': "#0000FF",
             },
             animation_frame="year",
             animation_group="Country",
             #range_y=[0, 200000]
             )
plt.tight_layout()
# X ekseni altındaki metni özelleştirme
fig.update_xaxes(title_text='Country', title_standoff=0)  # title_standoff ile kaydırma yapabilirsiniz
fig.show()


# In[61]:


#7.grafik
"""Renewables (% electricity"""

# Function to plot features on world map
def plot_world_map(column_name):
    fig = go.Figure()
    for year in range(2000, 2021):
        # Filter the data for the current year
        filtered_df = df2[df2['Year'] == year]

        # Create a choropleth trace for the current year
        trace = go.Choropleth(
            locations=filtered_df['Country'],
            z=filtered_df[column_name],
            locationmode='country names',
            colorscale='Jet',  
            colorbar=dict(title=column_name),
            zmin=df[column_name].min(),
            zmax=df[column_name].max(),
            visible=False  
        )

        # Add the trace to the figure
        fig.add_trace(trace)

    # Set the first trace to visible
    fig.data[0].visible = True

    # Create animation steps
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method='update',
            args=[{'visible': [False] * len(fig.data)}, 
                  {'title_text': f'{column_name} Map - {2000 + i}', 'frame': {'duration': 1000, 'redraw': True}}],
            label=str(2000 + i)  
        )
        step['args'][0]['visible'][i] = True 
        steps.append(step)

        # Create the slider
    sliders = [dict(
        active=0,
        steps=steps,
        currentvalue={"prefix": "Year: ", "font": {"size": 14}},  # Increase font size for slider label
    )]

    
    fig.update_layout(
        title_text=f'{column_name} Map with slider',  
        title_font_size=24,  
        title_x=0.5,  
        geo=dict(
            showframe=True,
            showcoastlines=True,
            projection_type='robinson'
        ),
        sliders=sliders,
        height=500,  
        width=1000,  
        font=dict(family='Arial', size=12),  
        margin=dict(t=80, l=50, r=50, b=50),
        template='plotly_dark',
    )

    # Show the figure
    fig.show()

df2.columns
column_name = 'Renewables (% electricity)'
plot_world_map(column_name)


#8


"""Using Electricity from wind (TWh) Over Years'"""
def plot_map(df, column, title):
    """
    Create an animated choropleth map with specified data and parameters.
    
    Parameters:
        df (DataFrame): The DataFrame containing the data.
        column (str): The name of the column to be used as the color metric.
        title (str): The title of the choropleth map.
        
    Returns:
        fig: The Plotly figure object representing the choropleth map.
    """
    
    # Create a choropleth map using Plotly Express
    fig = px.choropleth(
        df,
        locations = 'Country',
        locationmode = 'country names',
        color = column,
        hover_name = 'Country',
        color_continuous_scale = 'RdYlGn',
        animation_frame = 'Year',
        range_color = [0, 100])

    # Update geographic features
    fig.update_geos(
        showcoastlines = True,
        coastlinecolor = "Black",
        showland = True,
        landcolor = "white",
        showcountries = True,
        showocean = True,
        oceancolor = "LightBlue")
    
    # Update the layout of the figure
    fig.update_layout(
        title_text = title,
        geo = dict(
            showframe = False,
            showcoastlines = False,
            projection_type = 'equirectangular',
            showland = True,
            landcolor = "white",
            showcountries = True,
            showocean = True,
            oceancolor = "LightBlue"),
        width = 1000,
        height = 850,
        dragmode = 'pan',
        hovermode = 'closest',
        coloraxis_colorbar = dict(
            title = column,
            title_font_size = 14,
            title_side = 'right',
            lenmode = 'pixels',
            len = 300,
            thicknessmode = 'pixels',
            thickness = 15),
        updatemenus = [
            {"type": "buttons", "showactive": False, "x": 0.1, "y": 0.9, "buttons": [{"label": "Play", "method": "animate"}]},
            {"type": "buttons", "showactive": False, "x": 0.18, "y": 0.9, "buttons": [{"label": "Pause", "method": "animate"}]},
            {"type": "buttons", "showactive": False, "x": 0.26, "y": 0.9, "buttons": [{"label": "Stop", "method": "animate"}]}],
        sliders = [{"yanchor": "top", "xanchor": "left", "currentvalue": {"font": {"size": 20}}, "steps": []}])

    # Create slider steps for animation
    slider_steps = []

    for year in df['Year'].unique():
        step = {
            "args": [
                [year],
                {"frame": {"duration": 300, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
            "label": str(year),
            "method": "animate"}
        slider_steps.append(step)

    # Assign slider steps to the figure layout
    fig.layout.updatemenus[0].buttons[0].args[1]['steps'] = slider_steps

    return fig




plot_map(df,'Electricity from wind (TWh)',' Using Electricity from wind (TWh) Over Years')


# Machine Learning

data=pd.read_csv('energy_solar.csv')
data.drop('Unnamed: 0',axis=1,inplace=True)
data.head()

"""Kategorik, Numerik, Kardinal değişkenler"""
def grab_col_names(dataframe, cat_th=20, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.
    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri
    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi
    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))
    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı
    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(data, cat_th=4, car_th=35)

"""Categorical Analysis"""
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(data, col, plot=False)

"""Missing Value Analysis"""
def missing_values_tabl(df):
    na_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = df[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (df[na_columns].isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    missing_df = pd.DataFrame(missing_df)
    return missing_df


missing_values_tabl(data)

data['Income_group'].fillna('Unknown',inplace=True)
for i in num_cols:
    data[i].fillna(data[i].mean(),inplace=True)

"""Outlier Analysis"""
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    print(col, check_outlier(data, col))
    if check_outlier(data, col):
        replace_with_thresholds(data, col)

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

"""Encoder Labels"""
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
data["Income_group"] = labelencoder.fit_transform(data["Income_group"])
data.head()

data = pd.get_dummies(data, columns=['Country'],drop_first=True)
data.head()

# Dependent and Undependent
X = data[['Electricity from hydro (TWh)', 'Wind Generation - TWh', 'Solar Generation - TWh']]
y = data['Income_group']

# train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Label Encocder
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Dexision Tree
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train_encoded)

# Modeli test verileri üzerinde değerlendirme
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test_encoded, y_pred)
conf_matrix = confusion_matrix(y_test_encoded, y_pred)
class_report = classification_report(y_test_encoded, y_pred)

# Results
print("Doğruluk Oranı:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Sınıflandırma Raporu:\n", class_report)

"""Pipeline"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
# Pipeline
pipelines = {
    'Decision Tree': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', DecisionTreeClassifier(random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=42))
    ]),
    'Support Vector Classifier': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='linear', random_state=42))
    ])
}


results = {}
for model_name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train_encoded)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test_encoded, y_pred)
    classification_rep = classification_report(y_test_encoded, y_pred, output_dict=True)
    results[model_name] = {
        'Accuracy': accuracy,
        'Classification Report': classification_rep
    }

# Results
for model_name, result in results.items():
    print(f"Model: {model_name}")
    print(f"Accuracy: {result['Accuracy']:.2f}")
    print("Classification Report:")
    print(pd.DataFrame(result['Classification Report']).transpose())
    print("\n")