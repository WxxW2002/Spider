import scrapy
import csv

class geturl(scrapy.Spider):
    name = 'geturl'

    def start_requests(self):
        with open('Spider/data/urls.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                url = row[0].strip()
                total_pages = int(row[1].strip())
                for page in range(1, total_pages):
                    page_url = url.replace("pg1", f"pg{page}")
                    yield scrapy.Request(url=page_url, callback=self.parse)

    def parse(self, response):
        # get brief information
        listings = response.css('#content > div.leftContent > ul > li')

        for listing in listings:
            link = listing.css('div.info.clear div.title a::attr(href)').get()

            # Write link to csv file
            with open('Spider/data/htmls.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([link])
            