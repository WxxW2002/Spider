import pandas as pd
pd.set_option('display.max_columns', None)
from geopy.distance import geodesic
from tqdm import tqdm
from pypinyin import lazy_pinyin

def parse_chinese_number(chinese_number):
  chinese_number = chinese_number.replace('两', '二')
  number_mapping = {
    '零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
    '十': 10, '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15, '十六': 16, '十七': 17, '十八': 18, '十九': 19,
    '二十': 20, '二十一': 21, '二十二': 22, '二十三': 23, '二十四': 24, '二十五': 25, '二十六': 26, '二十七': 27, '二十八': 28, '二十九': 29,
    '三十': 30, '三十一': 31, '三十二': 32, '三十三': 33, '三十四': 34, '三十五': 35, '三十六': 36, '三十七': 37, '三十八': 38, '三十九': 39,
    '四十': 40, '四十一': 41, '四十二': 42, '四十三': 43, '四十四': 44, '四十五': 45, '四十六': 46, '四十七': 47, '四十八': 48, '四十九': 49,
    '五十': 50, '五十一': 51, '五十二': 52, '五十三': 53, '五十四': 54, '五十五': 55, '五十六': 56, '五十七': 57, '五十八': 58, '五十九': 59,
    '六十': 60, '六十一': 61, '六十二': 62, '六十三': 63, '六十四': 64, '六十五': 65, '六十六': 66, '六十七': 67, '六十八': 68, '六十九': 69,
    '七十': 70, '七十一': 71, '七十二': 72, '七十三': 73, '七十四': 74, '七十五': 75, '七十六': 76, '七十七': 77, '七十八': 78, '七十九': 79,
    '八十': 80, '八十一': 81, '八十二': 82, '八十三': 83, '八十四': 84, '八十五': 85, '八十六': 86, '八十七': 87, '八十八': 88, '八十九': 89,
    '九十': 90, '九十一': 91, '九十二': 92, '九十三': 93, '九十四': 94, '九十五': 95, '九十六': 96, '九十七': 97, '九十八': 98, '九十九': 99,
    '一百一十三': 113, '一百一十四': 114,
  }
  return number_mapping[chinese_number]

def calculate_distance(housing_lat, housing_lon, subway_lat, subway_lon):
    housing_coord = (housing_lat, housing_lon)
    subway_coord = (subway_lat, subway_lon)
    return geodesic(housing_coord, subway_coord).meters

if __name__ == '__main__':
  data = pd.read_csv('data/house.csv')
  df_subway = pd.read_csv('data/subway.csv')
  data = data.dropna()

  columns_to_drop = ['URL', 'Community URL', 'Community', 'Community Address']
  data = data.drop(columns=columns_to_drop)

  columns_to_encode = ['District', 'House Structure', 'Building Type', 'Building Structure', 'Decoration Degree', 'Housing Age']
  data = pd.get_dummies(data, columns=columns_to_encode)

  data['East'] = 0
  data['South'] = 0
  data['West'] = 0
  data['North'] = 0

  for index, row in data.iterrows():
      orientation = row['Orientation']
      if '东' in orientation:
          data.at[index, 'East'] = 1
      if '南' in orientation:
          data.at[index, 'South'] = 1
      if '西' in orientation:
          data.at[index, 'West'] = 1
      if '北' in orientation:
          data.at[index, 'North'] = 1
  data = data.drop(columns=['Orientation'])

  data[['Room', 'Hall', 'Kitchen', 'Bathroom']] = data['House Type'].str.extract('(\d+)室(\d+)厅(\d+)厨(\d+)卫')
  data = data.drop(columns=['House Type'])

  data[['Ladder', 'Door']] = data['Ladder Ratio'].str.extract('(\w+)梯(\w+)户')
  data = data.drop(columns=['Ladder Ratio'])
  data['Ladder'] = data['Ladder'].apply(parse_chinese_number)
  data['Door'] = data['Door'].apply(parse_chinese_number)

  data[['Floor Type', 'Totle Floor']] = data['Floor'].str.extract('(\w+) \(共(\d+)层\)')
  data = data.drop(columns=['Floor'])
  data = pd.get_dummies(data, columns=['Floor Type'])

  data = pd.get_dummies(data, columns=['Has Elevator'])

  # 创建一个新的列，用于存储房屋到最近地铁站的距离
  data['Subway_Dist'] = None

  # 对于每个房屋，计算距离最近地铁站的距离
  for index, row in tqdm(data.iterrows(), total=len(data)):
      housing_lat = row['Latitude']
      housing_lon = row['Longitude']
      min_distance = float('inf')  # 初始最小距离设为无穷大

      # 遍历地铁站列表，计算距离
      for _, subway_row in df_subway.iterrows():
          subway_lat = subway_row['Latitude']
          subway_lon = subway_row['Longitude']
          distance = calculate_distance(housing_lat, housing_lon, subway_lat, subway_lon)
          if distance < min_distance:
              min_distance = distance

      # 更新房屋数据中的最近地铁站距离列
      data.at[index, 'Subway_Dist'] = min_distance
  
  new_column_names = [col.replace(' ', '_') for col in data.columns]
  new_column_names
  new_columns = []
  for column in new_column_names:
      pinyin = ''.join(lazy_pinyin(column))
      new_columns.append(pinyin)
  data.columns = new_columns

  data.to_csv('data/house_processed.csv', index=False)