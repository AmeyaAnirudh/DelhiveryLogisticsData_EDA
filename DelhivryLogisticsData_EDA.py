# %% [markdown]
# Delhivery is a prominent logistics and supply chain services company in India, known for its extensive reach and efficient delivery solutions.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
dataset = pd.read_csv("C:\\Data Science\\projects\\Delhivery\\delhivery.csv")
df = pd.DataFrame(dataset)
print(df.head().to_string())

# %% [markdown]
# data: Tells whether the data is testing or training data.
# 
# trip_creation_time: Timestamp of trip creation
# 
# route_schedule_uuid: Unique Id for a particular route schedule.
# 
# route_type: Transportation type FTL – Full Truck Load: FTL shipments get to the destination sooner, as the truck is making no other pickups or drop-offs along the way 
# Carting: Handling system consisting of small vehicles (carts)
# 
# trip_uuid: Unique ID given to a particular trip (A trip may include different source and destination centers).
# 
# source_center: Source ID of trip origin
# 
# source_name; Source Name of trip origin
# 
# destination_center: Destination ID
# 
# destination_name: Destination Name
# 
# od_start_time: Trip start time
# 
# od_end_time: Trip end time
# 
# start_scan_to_end_scan: Time taken to deliver from source to destination
# 
# is_cutoff: Unknown field
# 
# cutoff_factor: Unknown field
# 
# cutoff_timestamp: Unknown field
# 
# actual_distance_to_destination: Distance in Kms between source and destination warehouse
# 
# actual_time: Actual time taken to complete the delivery (Cumulative)
# 
# osrm_time
# An open-source routing engine time calculator which computes the shortest path between points in a given map (Includes usual traffic, distance through major and minor roads) and gives the time (Cumulative)
# 
# osrm_distance
# An open-source routing engine which computes the shortest path between points in a given map (Includes usual traffic, distance through major and minor roads) (Cumulative)
# 
# factor
# Unknown field
# 
# segment_actual_time
# This is a segment time. Time taken by the subset of the package delivery
# 
# segment_osrm_time
# This is the OSRM segment time. Time taken by the subset of the package delivery
# 
# segment_osrm_distance
# This is the OSRM distance. Distance covered by subset of the package delivery
# 
# segment_factor
# Unknown field
# 
# 
# 

# %%
df.dtypes

# %%
df['route_type'].apply(type).value_counts()

# %%
df.info()

# %% [markdown]
# osrm_time:
# 
# OSRM - open source routing machine
# 
# OSRM is a routing engine that provide time estimates for routes based on road network data.
# 
# OSRM (Open Source Routing Machine) is a powerful and fast routing engine that provides optimized routes, time estimates, and distances based on road network data. It is commonly used for transportation and logistics applications to calculate the best possible routes and to estimate the travel time and distance between two locations.
# 
# What it Means for OSRM to Calculate Time for Delivery or a Segment:
# 
# 	1.	Route Calculation:
# 	•	OSRM can compute the optimal route between two points (such as a source location and a destination) using road networks, taking into account factors like road types    (highways, local roads), traffic conditions (if available), speed limits, and other geographical data.
# 	2.	Time Calculation for Delivery:
# 	•	Delivery Time Calculation: Based on the chosen route, OSRM estimates how long it will take for a vehicle or person to travel between two locations.
# 	•	This calculation takes into account:
# 	•	Distance between the locations.
# 	•	Expected speed on each road segment (highways vs. smaller streets).
# 	•	Traffic data (if provided to OSRM).
# 	•	OSRM produces a time estimate for the whole delivery route.
# 	3.	Time Calculation for a Segment:
# 	•	Segment Time Calculation: If the entire delivery route is broken into segments (smaller parts of the route), OSRM can also calculate the estimated time for each segment.
# 	•	For example, if the delivery goes through multiple cities, OSRM can provide:
# 	•	Estimated time from City A to City B (Segment 1).
# 	•	Estimated time from City B to City C (Segment 2).
# 	•	This is useful for understanding how long different portions of the route will take and whether delays may occur in certain segments.
# 
# Key Uses of OSRM Time Calculation:
# 
# 	•	Predicting Delivery Time (ETA): OSRM helps in calculating the Estimated Time of Arrival (ETA) based on the road conditions, distance, and vehicle speed.
# 	•	Comparing Actual vs. Estimated Times: OSRM’s calculated times can be compared with the actual delivery times to analyze route efficiency and identify delays.
# 	•	Optimizing Routes: By calculating the most efficient route, OSRM helps reduce travel time and fuel costs.

# %%
df.describe()

# %% [markdown]
# Dropping irrelevant Columns
# 
# factor : because data is unknown
# 
# segmant_factor : " " "
# 
# is_cutoff: Unknown field
# 
# cutoff_factor: Unknown field
# 
# cutoff_timestamp: Unknown field
# 
# 

# %%
# df.drop(['factor','segment_factor','is_cutoff','cutoff_factor','cutoff_timestamp'],axis=1)
col_to_drop = ['factor','segment_factor','is_cutoff','cutoff_factor','cutoff_timestamp']
col_in_df = [col for col in col_to_drop if col in df.columns]

if col_in_df:
    df.drop(col_in_df,axis=1,inplace=True)
    print(f"dropped columns: {col_in_df}")
else:
    print("No column to drop.")

# %% [markdown]
# Rename
# 
# source_center: Source ID of trip origin
# 
# destination_center: Destination ID
# 
# od_start_time: Trip start time
# 
# od_end_time: Trip end time
# 
# start_scan_to_end_scan: Time taken to deliver from source to destination
# 
# actual_time: Actual time taken to complete the delivery (Cumulative)
# 

# %%
df.rename(columns={'source_center':'source_centre_ID','destination_center':'destination_ID','od_start_time':'Trip_start_time','od_end_time':'trip_end_time','start_scan_to_end_scan':'s_to_d_deliveryTime','actual_time':'actual_full_deliveryTime'},inplace=True)
print(df.to_string())
# df.to_csv('cleaned_delhivery_logistics.csv', index=False')

# %%
df.rename(columns={'s_to_d_deliveryTime':'start_scan_to_end_scan'},inplace=True)
print(df.to_string())

# %% [markdown]
# Dropping duplicate rows

# %%
df.duplicated()

# %%
df.count()

# %%
df.drop_duplicates()

# %%
df.count()

# %% [markdown]
# Removing null values
# 
# df.count() shows there arenull values present in the dataset. So we need to remove the missing values or null values

# %%
df.isnull().sum()

# %%
df.dropna(inplace=True)

# %%
df.isnull().sum()

# %%
df.count()

# %% [markdown]
# Previously there were 144867 rows
# 
# After removing the null values : 144316 rows
# 
# source_name had 293 null values.
# destination_name 261 null values.
# 
# Which means that 551 rows were removed after applying df.dropna(inplace=True)

# %%
df.dtypes

# %% [markdown]
# trip_creation_time                 object
# 
# Trip_start_time                      object
# 
# trip_end_time                        object
# 
# We need to convert it into datetime format.

# %%
#removing any whitespace in the column name

# df.columns = df.columns.str.strip()
df['trip_creation_time'] = pd.to_datetime(df['trip_creation_time'])
# print(df.dtypes)
print(df['trip_creation_time'].head())


# %% [markdown]
# od_start_time, start_scan_to_end_scan, actual_full_deliveryTime and od_end_time should be converted to datetime.

# %%
df['trip_creation_time']= df['trip_creation_time'].astype(str)

# %%
df.dtypes


# %%
# print(df.to_string())
df['Trip_start_time']=pd.to_datetime(df['Trip_start_time'])
df['trip_end_time']=pd.to_datetime(df['trip_end_time'])


# %%
df.dtypes

# %%
# print(df.to_string()).head()
df.head()

# %% [markdown]
# Applying floor() to the trip_creation_time, Trip_start_time and	trip_end_time
# 
# 

# %%
df['trip_creation_time'] = pd.to_datetime(df['trip_creation_time']).dt.floor('S')
df['Trip_start_time'] = pd.to_datetime(df['Trip_start_time']).dt.floor('S')
df['trip_end_time'] = pd.to_datetime(df['trip_end_time']).dt.floor('S')

# %%
df.head()

# %% [markdown]
# Let's split the source_name into city and state

# %%
df['source_state'] = df['source_name'].str.split(' ').str[1].str.lstrip('(').str.rstrip(')')
print(df['source_state'])
# df['source_name'].str.split(' ').str[1].str.lstrip('(').str.rstrip(')').value_counts()

# %%
df['source_city'] = df['source_name'].str.split(' ').str[0]
print(df['source_city'])

# %%
df['source_name'].str.split(' ').str[1].str.lstrip('(').str.rstrip(')').value_counts()

# %% [markdown]
# Highest number of orders were dispatched from Haryana=27408. 

# %%
df['source_name'].str.split(' ').str[0].value_counts()
# print(df['source_name'])   

# %% [markdown]
# Highest numebr of dispatches happened from Gurgaon_Bilaspur_HB=23267
# 
# In which state is this place located?
# Gurgaon_Bilaspur_HB        23267

# %%
df[df['source_name'].str.startswith('Gurgaon_Bilaspur_HB')]                 #Haryana

# %%
#splitting destination_name into city and state.
print("destination_state")
df['destination_state'] = df['destination_name'].str.split(' ').str[1].str.lstrip('(').str.rstrip(')')
print(df['destination_state'])
print("\n")
print("destination'_city")
df['destination_city'] = df['destination_name'].str.split(' ').str[0]
print(df['destination_city'])

# %%
df['destination_name'].str.split(' ').str[1].str.lstrip('(').str.rstrip(')').value_counts()

# %%
df['destination_name'].str.split(' ').str[0].value_counts()

# %% [markdown]
# Highest number of orders were recieved from karnataka state.
# 
# Anyway highest number of ordersin terms of city were recieved from Gurgaon_Bilaspur_HB     15192 in Haryana.

# %%
# #route type


# df['route_type'].str.startswith('Carting').value_counts()

# print("\n")
# df['route_type'].str.count('FTL').value_counts()

# a = []
# for i in df['route_type']:
#     if i=='Carting':
#         a.append(i)
# print(len(a))

# count = df['route_type'].get("Carting",0).value_counts()
# count = df['route_type'].value_counts().get("Carting",0)
# print("Carting ", count)
print(df['route_type'].unique())

print("\n")

df['route_type'].str.startswith('Carting').value_counts()

# %% [markdown]
# Carting route_type = 45185
# 
# route_type FTL(Full truck load) = 99132   
# 
# 

# %%
df.count()

# %%
#Categorizing time field "start_scan_to_end_scan" 

df["start_scan_to_end_scan"].max()
# df['start_scan_to_end_scan'].idxmax()


# %%
df['start_scan_to_end_scan'].min()

# %% [markdown]
# Max value of stat_scan_to_end_scan is 7898 and minimum value is 20

# %%
#converting dtype of field start_scan_to_end_scan

df['start_scan_to_end_scan'] = df["start_scan_to_end_scan"].astype('float')

# %%
def time_to_cat(val):
    if(val>=2 and val<=1000):
        return('2_to_1000')
    elif(val>1000 and val<=2000):
        return('1000_to_2000')
    elif(val>2000 and val<=3000):
        return('2000_to_3000')
    elif(val>3000 and val<=4000):
        return('3000_to_4000')
    elif(val>4000 and val<=5000):
        return('4000_to_5000')    
    elif(val>5000 and val<=6000):
        return('5000_to_6000')
    elif(val>6000 and val<=7000):
        return('6000_to_7000')
    elif(val>7000 and val<=8000):
        return('7000_to_8000')

df['start_scan_to_end_scan'] = df['start_scan_to_end_scan'].apply(time_to_cat)
print(df['start_scan_to_end_scan'])


# %%
df['start_scan_to_end_scan'].value_counts()

# %%
df['start_scan_to_end_scan'].value_counts().reset_index().rename(columns={'index':'start_scan_to_end_scan','start_scn_to_end_scn':'count'})

# %%
print(df.to_string())

# %%
cols_to_round = ['actual_full_deliveryTime','osrm_time','segment_actual_time','segment_osrm_time']
df[cols_to_round] = df[cols_to_round].apply(lambda x:x.round(2))

# %%
dist_cols_to_round = ['actual_distance_to_destination','osrm_distance','segment_osrm_distance']
df[dist_cols_to_round] = df[dist_cols_to_round].apply(lambda x:x.round(2))

# %%
print(df.to_string())

# %%
df['actual_full_deliveryTime'].max()

# %%
df['actual_full_deliveryTime'].min()

# %%
plt.scatter(df['start_scan_to_end_scan'],df['actual_full_deliveryTime'],alpha=0.5)
plt.xticks(ticks=None, labels=None, rotation=90, fontsize=None)
plt.xlabel("start_scan_to_end_scan")
plt.ylabel("actual_full_deliveryTime")
plt.show()


# %% [markdown]
# Max delhivery start scan to end scan happened between the range of 4000 - 5000 range of time.
# 
# Least delhivery time taken count was from 2 - 1000

# %%
#Finding delay based on osrm time and actual delivery time  
# (actual_full_deliveryTime vs osrm_time)

#computing delay = osrm_time - actual_full_deliveryTime
df['delay'] = df['actual_full_deliveryTime'] -df['osrm_time']
# print(df['delay'])
print(df.to_string())

#if actual time is > osrm prediction time, delay willl br positive




# %%
#grouping based on source and destination

df['Source'] = df['source_city']+"(" + df['source_state'] + ")"
df['Destination'] = df['destination_city'] + "(" + df['destination_state'] + ")"
df['route'] = df['Source'] + " -> " + df['Destination']

#Group and aggregate 
#Group by route to calculate average delay and number of deliveries for each route

dealy_analysis = (df.groupby('route')['delay'].agg(['mean','count']).reset_index().sort_values(by='mean',ascending=False).rename(columns= {'mean':'average_delay', 'count': 'delivery_count'}))


#Visualizing the top 10 routes with the highest average delay

top_delays = dealy_analysis.head(10)

plt.figure(figsize=(12,8))
plt.barh(top_delays['route'],top_delays['average_delay'], color='orange')
plt.xlabel('Average Delay')
plt.ylabel('Route')
plt.title('Top 10 routes with highest average delay')
plt.gca().invert_yaxis()        #get curent axis
plt.show()

 


# %%
#Analyzing Distance impact
#delays vs. distance

for distance_column in ['actual_distance_to_destination']:
    if distance_column in df.columns:
        plt.figure(figsize=(8,6))
        plt.scatter(df[distance_column],df['delay'],alpha=0.5)
        plt.title(f'Delay vs {distance_column.replace("_"," ").title()}')
        plt.xlabel(f'{distance_column.replace("_"," ").title()}')
        plt.ylabel('Delay')
        plt.grid(True)
        plt.show()

        #calculating correlation
        correlation = df[[distance_column,'delay']].corr().loc[distance_column, 'delay']
        print(f"Correlation between{distance_column} and delay: {correlation}")

# delay_analysis.to_csv('delay_analysis.csv', index=False)




