import streamlit as st
df=pd.read_csv('ad/energyy.csv')



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



