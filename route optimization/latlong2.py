import pandas as pd
from math import radians, cos, sin, sqrt, atan2

# Haversine formula to calculate distance
def haversine(lat1, lon1, lat2, lon2):
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    # Radius of earth in kilometers. Use 6371 for kilometers
    r = 6371
    return c * r

# Function to calculate distance for a given BookingID
def calculate_distance_for_bookingID(file_path, bookingID):
    # Load the dataset
    df = pd.read_excel(file_path)
    
    # Find the row with the given BookingID
    row = df[df['BookingID'] == bookingID]
    
    if row.empty:
        return "BookingID not found."
    
    # Extract latitude and longitude
    org_lat, org_lon = map(float, row['Org_lat_lon'].values[0].split(','))
    des_lat, des_lon = map(float, row['Des_lat_lon'].values[0].split(','))
    
    # Calculate the distance
    distance = haversine(org_lat, org_lon, des_lat, des_lon)
    return distance

# User input for BookingID
bookingID_input = input("Enter BookingID: ")

# Calculate distance for the input BookingID
file_path = 'C:\\Users\\USER\\Desktop\\som-tsp-master\\dataset.xlsx' # Change this to your actual file path
distance = calculate_distance_for_bookingID(file_path, bookingID_input)
print(f"The distance for BookingID {bookingID_input} is: {distance} kilometers")
