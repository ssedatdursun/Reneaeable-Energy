import pandas as pd
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 80)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
warnings.simplefilter(action='ignore')
pd.set_option('display.expand_frame_repr', False)

df = pd.read_csv('energyy.csv')
df.head()
df_solar=pd.read_csv('energy_solar.csv')
df_solar.drop('Unnamed: 0',axis=1,inplace=True)


# Dropped regions for checking countries
df["Country"].unique()
drop_1 = ['Africa', 'Asia Pacific', 'CIS', 'North America', 'Other Europe', 'Other South & Central America',
          'South & Central America', 'World']


# fill some na values for creating a graph/table
fill_na_value = ["Electricity from hydro (TWh)", "Electricity from wind (TWh)", "Electricity from solar (TWh)",
                 "Other renewables including bioenergy (TWh)", "Wind (% electricity)", "Renewables (% electricity)",
                 "Hydro (% electricity)"]

df_solar[fill_na_value] = df_solar[fill_na_value].fillna(df_solar[fill_na_value].mean())
df_solar.isna().sum()

df2.describe().T

# final value creating without regions
df_region = df[df["Country"].isin(drop_1)]
df_region.isna().sum()

df_region.groupby('Country')['Electricity from hydro (TWh)'].sum().sort_values(ascending=False)
df_region.groupby('Country')['Electricity from solar (TWh)'].sum().sort_values(ascending=False)
df_region.groupby('Country')['Electricity from wind (TWh)'].sum().sort_values(ascending=False)

# Graphs for Analyzing#

# 1
# piechart yenılenebılır enerjı 2020

# Filter renewable consumption dataframe for the World record in 2020
world = df_region[(df_region.Year == 2020) & (df_region.Country == 'World')]
# Drop all columns not containing consumption data so we can create a pie chart
world.drop(['Country', 'Year', 'Unnamed: 0'], axis=1, inplace=True)
world.columns
pie_world = world[
    ['Solar Generation - TWh', 'Wind Generation - TWh', 'Hydro Generation - TWh', 'Geo Biomass Other - TWh']]
plt.style.use("seaborn-pastel")
plt.figure(figsize=(7, 7))
plt.title('Production of Worldwide Renewable Energy Consumption 2020')

plt.pie(pie_world.iloc[0],
        labels=['Solar Generation - TWh', 'Wind Generation - TWh', 'Hydro Generation - TWh', 'Geo Biomass Other - TWh'],
        autopct='%1.1f%%', wedgeprops={'edgecolor': 'black'})
plt.show(block=True)

# 2
# Renewable Energy per year


renewable_consumption = ["Solar Generation - TWh", "Wind Generation - TWh",
                         "Hydro Generation - TWh",
                         "Biofuels Production - TWh - Total"]
energy_df = df_solar[renewable_consumption]
year_range = df_solar["Year"].isin(range(1990, 2021))
energy_df = energy_df[year_range]
bar = df_solar.groupby('Year')[renewable_consumption].sum()
bar.reset_index(level=0, inplace=True)

bar.head()

bar.plot(x='Year', ylabel='Terrawatt Hour', kind='bar', stacked=True, figsize=(20, 6),
         color=['black', 'red', 'green', 'blue'],
         title='Global Renewable Consumption 1990 - 2020')
plt.show()




#
df_solar['Solar (% electricity)'] = df_solar['Solar (% electricity)'].fillna(0)
df_solar.isna().sum()
df_solar.head()

# 4
# using list comprehension to get string with substring
economies = ['High income',
             'Upper middle income',
             'Lower middle income',
             'Low income']
print(economies)
df_economy = df_solar[df_solar['Income_group'].isin(economies)]
df_economy = df_economy[df_economy['Year'] > 1990]

targets = ['Wind (% electricity)', 'Renewables (% electricity)', 'Hydro (% electricity)', 'Solar (% electricity)']


def plot_timeseries(df):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    sns.lineplot(ax=axes[0, 0], data=df, x="Year", y=targets[0], hue="Income_group").set(title=targets[0])

    sns.lineplot(ax=axes[0, 1], data=df, x="Year", y=targets[1], hue="Income_group").set(title=targets[1])

    sns.lineplot(ax=axes[1, 0], data=df, x="Year", y=targets[2], hue="Income_group").set(title=targets[2])

    sns.lineplot(ax=axes[1, 1], data=df, x="Year", y=targets[3], hue="Income_group").set(title=targets[3])
    # plt.close()
    plt.show()
    return fig


# , ax.set_xticklabels(df["Entity"], rotation=45, fontsize=10)
plot_timeseries(df_solar)

# 5

# Focusing on columns with data on annual percentage change in energy sources consumption starting year 1990

consumption = df_region[df_region['Year'] >= 1990]
consumption["df_geo_percentage"] = (consumption["Geothermal Capacity"] / consumption["Geothermal Capacity"].sum()) * 100

fig = px.bar(consumption,
             x='Country', y='Geothermal Capacity',
             color='Country',
             animation_frame='Year',
             animation_group="Country",
             # range_y=[0, 5],
             labels={'Geothermal Capacity ': 'Geothermal Capacity change, %'},
             title='Regional Changes in Geothermal Capacity from 1990, %')

fig.update_layout(showlegend=False)
fig.add_vrect(x0=11.5, x1=10.5)
# X ekseni altındaki metni özelleştirme
fig.update_xaxes(title_text='Country', title_standoff=0)  # title_standoff ile kaydırma yapabilirsiniz
plt.tight_layout()
fig.show()

# 6. grafık


consumptionpersource = df_solar[df_solar['Year'] >= 2000]
consumptionpersource['year'] = pd.to_datetime(consumptionpersource['Year'], format='%Y')
consumptionpersource['year'] = consumptionpersource['year'].dt.year

fig = px.bar(consumptionpersource,
             x="Country", y=["Solar Generation - TWh",
                             "Wind Generation - TWh",
                             "Hydro Generation - TWh",
                             "Biofuels Production - TWh - Total"],
             title="Consumption Profiles per Countries",
             color_discrete_map={
                 'Solar Generation - TWh': 'black',
                 'Wind Generation - TWh': '#eeee00',
                 'Hydro Generation - TWh': "#B8860B",
                 'Biofuels Production - TWh - Total': "#0000FF",
             },
             animation_frame="year",
             animation_group="Country",
             # range_y=[0, 200000]
             )
plt.tight_layout()
# X ekseni altındaki metni özelleştirme
fig.update_xaxes(title_text='Country', title_standoff=0)  # title_standoff ile kaydırma yapabilirsiniz
fig.show()


# 7.grafik
"""Renewables (% electricity"""


# Function to plot features on world map
def plot_world_map(column_name):
    fig = go.Figure()
    for year in range(2000, 2021):
        # Filter the data for the current year
        filtered_df = df_solar[df_solar['Year'] == year]

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


column_name = 'Renewables (% electricity)'
plot_world_map(column_name)
