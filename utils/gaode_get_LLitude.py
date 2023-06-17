import csv
import requests
import time

def geocode_address(address):
    api_key = '2f294705f67cb1bd7d3bf0f22df89bf2'
    geocode_url = 'https://restapi.amap.com/v3/geocode/geo'
    params = {
        'address': address,
        'output': 'json',
        'key': api_key,
    }

    response = requests.get(geocode_url, params=params)

    if response.status_code == 200:
        data = response.json()
        if data['status'] == '1':
            location = data['geocodes'][0]['location']
            longitude, latitude = location.split(',')
            return latitude, longitude
    return None, None


with open('../Spider/data/data.csv', 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    fieldnames = reader.fieldnames + ['Latitude', 'Longitude']
    
    with open('../Spider/data/data_with_coordinates.csv', 'a', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            address = '上海市' + row['Community Address']
            latitude, longitude = geocode_address(address)

            # 将经度和纬度添加到行中
            row['Latitude'] = latitude
            row['Longitude'] = longitude

            writer.writerow(row)

            time.sleep(1)
