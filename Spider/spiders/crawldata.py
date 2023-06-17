import scrapy
import csv
import os
import re

class crawldata(scrapy.Spider):
    name = 'crawldata'
    custom_settings = {
        'DOWNLOAD_DELAY': 0.1,
        'FEEDS': {
            'new_data.csv': {
                'format': 'csv',
                'encoding': 'utf8',
                'store_empty': False,
                'fields': None,
                'indent': 4,
                'item_export_kwargs': {
                    'export_empty_fields': True,
                },
            },
        },
    }

    def start_requests(self):
        with open('Spider/data/new_htmls.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                url = row[0]
                yield scrapy.Request(url=url, callback=self.parse)
                
               
    def parse(self, response):
        url = response.url
        title = response.xpath('/html/body/div[3]/div/div/div[1]/h1/text()').get(default='')
        subtitle = response.xpath('/html/body/div[3]/div/div/div[1]/div/text()').get(default='')
        total_price = response.xpath('/html/body/div[5]/div[2]/div[3]/div/span[1]/text()').get(default='')
        average_price = response.xpath('/html/body/div[5]/div[2]/div[3]/div/div[1]/div[1]/span/text()').get(default='')
        district = response.xpath('/html/body/div[5]/div[2]/div[5]/div[2]/span[2]/a[1]/text()').get(default='')
        community = response.xpath('/html/body/div[5]/div[2]/div[5]/div[1]/a[1]/text()').get(default='')
        community_url = "https://sh.lianjia.com" + response.xpath('/html/body/div[5]/div[2]/div[5]/div[1]/a[1]/@href').get(default='')
        house_type = response.css('#introduction > div > div > div.base > div.content > ul > li:nth-child(1)::text').get(default='')
        floor = response.css('#introduction > div > div > div.base > div.content > ul > li:nth-child(2)::text').get(default='')
        area_text = response.css('#introduction > div > div > div.base > div.content > ul > li:nth-child(3)::text').get(default='')
        area_match = re.search(r'(\d+(?:\.\d+)?)', area_text)
        area = area_match.group(1) if area_match else ''
        house_structure = response.css('#introduction > div > div > div.base > div.content > ul > li:nth-child(4)::text').get(default='')
        building_type = response.css('#introduction > div > div > div.base > div.content > ul > li:nth-child(6)::text').get(default='')
        orientation = response.css('#introduction > div > div > div.base > div.content > ul > li:nth-child(7)::text').get(default='')
        building_structure = response.css('#introduction > div > div > div.base > div.content > ul > li:nth-child(8)::text').get(default='')
        decoration_degree = response.css('#introduction > div > div > div.base > div.content > ul > li:nth-child(9)::text').get(default='')
        ladder_ratio = response.css('#introduction > div > div > div.base > div.content > ul > li:nth-child(10)::text').get(default='')
        has_elevator = response.css('#introduction > div > div > div.base > div.content > ul > li:nth-child(11)::text').get(default='')
        build_info = response.css('body > div.overview > div.content > div.houseInfo > div.area > div.subInfo.noHidden::text').get(default='')
        build_time = ''
        if build_info:
            build_time_match = re.search(r'(\d+)', build_info)
            if build_time_match:
                build_time = build_time_match.group(1)
        housing_age = response.css('#introduction > div > div > div.transaction > div.content > ul > li:nth-child(5) > span:nth-child(2)::text').get(default='')

        yield scrapy.Request(url=community_url, callback=self.parse_community_address, cb_kwargs={'item': {
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
            'Housing Age': housing_age,                  
        }})

    def parse_community_address(self, response, item):
        community_address = response.css('body > div.xiaoquDetailHeader > div > div.detailHeader.fl > div::text').get(default='')
        total_bulidings_text = response.css('body > div.xiaoquOverview > div.xiaoquDescribe.fr > div.xiaoquInfo > div:nth-child(6) > span.xiaoquInfoContent::text').get(default='')
        total_buildings_match = re.search(r'(\d+)', total_bulidings_text)
        total_buildings = total_buildings_match.group(1) if total_buildings_match else ''
        total_houses_text = response.css('body > div.xiaoquOverview > div.xiaoquDescribe.fr > div.xiaoquInfo > div:nth-child(7) > span.xiaoquInfoContent::text').get(default='')
        total_houses_match = re.search(r'(\d+)', total_houses_text)
        total_houses = total_houses_match.group(1) if total_houses_match else ''

        property_costs_text = response.css('body > div.xiaoquOverview > div.xiaoquDescribe.fr > div.xiaoquInfo > div:nth-child(3) > span.xiaoquInfoContent::text').get(default='')
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
            'Average Property Cost': average_property_cost,
        })
        yield item
