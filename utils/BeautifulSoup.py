import requests
from bs4 import BeautifulSoup
import re
import csv

class Crawldata:
    def __init__(self):
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        # self.proxies = {"http": "http://8.219.67.35:80", "https": "107.173.250.178:3000"}
        self.data = []

        # Create a CSV file and write header
        self.csv_header_written = False
        self.csv_file = open('../Spider/data/BSoup_data.csv', 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=[
            'URL', 'Title', 'Subtitle', 'Total', 'Average', 'District', 'Community', 'Community URL', 'House Type',
            'Floor', 'Area', 'House Structure', 'Building Type', 'Orientation', 'Building Structure', 'Decoration Degree',
            'Ladder Ratio', 'Has Elevator', 'Build Time', 'Housing Age', 'Community Address', 'Total Buildings',
            'Total Houses', 'Average Property Cost'
        ])

    def start_requests(self):
        with open('../Spider/data/valid_htmls.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                url = row[0]
                self.parse(url)

    def get_text_safe(self,soup_element, default=''):
        try:
            return soup_element.get_text(strip=True)
        except AttributeError:
            return default

    def get_href_safe(self, soup_element, default=''):
        if soup_element is None:
            return default
        try:        
            return soup_element['href']
        except (AttributeError, KeyError):
            return default

    def parse(self, url):
        response = requests.get(url, headers=self.headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        title = self.get_text_safe(soup.select_one('body > div.sellDetailHeader > div > div > div.title > h1'))
        subtitle = self.get_text_safe(soup.select_one('body > div.sellDetailHeader > div > div > div.title > div'))
        total_price = self.get_text_safe(soup.select_one('body > div.overview > div.content > div.price-container > div > span.total'))
        average_price_text = self.get_text_safe(soup.select_one('body > div.overview > div.content > div.price-container > div > div.text > div.unitPrice > span'))
        average_price = ''
        if average_price_text:
            average_price = re.search(r'\d+', average_price_text).group()
        district = self.get_text_safe(soup.select_one('body > div.overview > div.content > div.aroundInfo > div.areaName > span.info > a:nth-child(1)'))
        community = self.get_text_safe(soup.select_one('body > div.overview > div.content > div.aroundInfo > div.communityName > a.info'))
        community_url = "https://sh.lianjia.com" + self.get_href_safe(soup.select_one('body > div.overview > div.content > div.aroundInfo > div.communityName > a.info'))
        house_type = self.get_text_safe(soup.select_one('#introduction > div > div > div.base > div.content > ul > li:nth-child(1)'))
        if house_type:
            house_type = house_type[4:]
        floor_text = self.get_text_safe(soup.select_one('#introduction > div > div > div.base > div.content > ul > li:nth-child(2)'))
        floor = ''
        if floor_text:
            floor = floor_text[4:].split(' ')[0]
        area_text = self.get_text_safe(soup.select_one('#introduction > div > div > div.base > div.content > ul > li:nth-child(3)'))
        if area_text:
            area_text = area_text[4:]
        area_match = re.search(r'(\d+(?:\.\d+)?)', area_text)
        area = area_match.group(1) if area_match else ''
        house_structure = self.get_text_safe(soup.select_one('#introduction > div > div > div.base > div.content > ul > li:nth-child(4)'))
        if house_structure:
            house_structure = house_structure[4:]
        building_type = self.get_text_safe(soup.select_one('#introduction > div > div > div.base > div.content > ul > li:nth-child(6)'))
        if building_type:
            building_type = building_type[4:]
        orientation = self.get_text_safe(soup.select_one('#introduction > div > div > div.base > div.content > ul > li:nth-child(7)'))
        if orientation:
            orientation = orientation[4:]  
        building_structure = self.get_text_safe(soup.select_one('#introduction > div > div > div.base > div.content > ul > li:nth-child(8)'))
        if building_structure:
            building_structure = building_structure[4:]
        decoration_degree = self.get_text_safe(soup.select_one('#introduction > div > div > div.base > div.content > ul > li:nth-child(9)'))
        if decoration_degree:
            decoration_degree = decoration_degree[4:]
        ladder_ratio = self.get_text_safe(soup.select_one('#introduction > div > div > div.base > div.content > ul > li:nth-child(10)'))
        if ladder_ratio:
            ladder_ratio = ladder_ratio[4:]
        has_elevator = self.get_text_safe(soup.select_one('#introduction > div > div > div.base > div.content > ul > li:nth-child(11)'))
        if has_elevator:
            has_elevator = has_elevator[4:]
        build_info = self.get_text_safe(soup.select_one('body > div.overview > div.content > div.houseInfo > div.area > div.subInfo.noHidden'))
        build_time = ''
        if build_info:
            build_time_match = re.search(r'(\d+)', build_info)
            if build_time_match:
                build_time = build_time_match.group(1)
        housing_age = self.get_text_safe(soup.select_one('#introduction > div > div > div.transaction > div.content > ul > li:nth-child(5) > span:nth-child(2)'))
       
        item = {
            'URL': url,
            'Title': title,
            'Subtitle': subtitle,
            'Total': total_price,
            'Average': average_price,
            'District': district,
            'Community': community,
            'Community URL': community_url,
            'House Type': house_type,
            'Floor': floor,
            'Area': area,
            'House Structure': house_structure,
            'Building Type': building_type,
            'Orientation': orientation,
            'Building Structure': building_structure,
            'Decoration Degree': decoration_degree,
            'Ladder Ratio': ladder_ratio,
            'Has Elevator': has_elevator,
            'Build Time': build_time,
            'Housing Age': housing_age
        }
        self.parse_community_address(community_url, item)


    def parse_community_address(self, url, item):
        response = requests.get(url, headers=self.headers)

        if response.status_code != 200:
            item.update({
                'Community Address': '',
                'Total Buildings': '',
                'Total Houses': '',
                'Average Property Cost': ''
            })

            self.write_row_to_csv(item)

        else: 
            soup = BeautifulSoup(response.text, 'html.parser')

            community_address = self.get_text_safe(soup.select_one('body > div.xiaoquDetailHeader > div > div.detailHeader.fl > div'))
            total_bulidings_text = self.get_text_safe(soup.select_one('body > div.xiaoquOverview > div.xiaoquDescribe.fr > div.xiaoquInfo > div:nth-child(6) > span.xiaoquInfoContent'))
            total_buildings_match = re.search(r'(\d+)', total_bulidings_text)
            total_buildings = total_buildings_match.group(1) if total_buildings_match else ''
            total_houses_text = self.get_text_safe(soup.select_one('body > div.xiaoquOverview > div.xiaoquDescribe.fr > div.xiaoquInfo > div:nth-child(7) > span.xiaoquInfoContent'))
            total_houses_match = re.search(r'(\d+)', total_houses_text)
            total_houses = total_houses_match.group(1) if total_houses_match else ''

            property_costs_text = self.get_text_safe(soup.select_one('body > div.xiaoquOverview > div.xiaoquDescribe.fr > div.xiaoquInfo > div:nth-child(3) > span.xiaoquInfoContent'))
            property_cost_match = re.search(r'(\d+(\.\d+)?)(至(\d+(\.\d+)?))?\D+', property_costs_text)
            average_property_cost = None
            if property_cost_match:
                if property_cost_match.group(4):
                    property_cost1 = float(property_cost_match.group(1))
                    property_cost2 = float(property_cost_match.group(4))
                    average_property_cost = (property_cost1 + property_cost2) / 2
                else:
                    average_property_cost = float(property_cost_match.group(1))

            item.update({
                'Community Address': community_address,
                'Total Buildings': total_buildings,
                'Total Houses': total_houses,
                'Average Property Cost': average_property_cost
            })

            self.write_row_to_csv(item)

    def write_row_to_csv(self, item):
        if not self.csv_header_written:
            self.csv_writer.writeheader()
            self.csv_header_written = True
        self.csv_writer.writerow(item)

    def close_csv_file(self):
        self.csv_file.close()

if __name__ == "__main__":
    crawler = Crawldata()
    crawler.start_requests()
    crawler.close_csv_file()
