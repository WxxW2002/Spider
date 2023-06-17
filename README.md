Spider
======
Spider for Lianjia，to obtain information about second-hand housing in Shanghai.

File Structure
--------------
* `Spider/` - directory for scrapy code and data.
    * `data/` - data obtained by scrapy
        * `data_with_coordinites.csv` - data with coordinates
        * `htmls.csv` - all htmls
        * `original_data.csv` - data without coordinates
        * `subway.csv` - subway information
        * `urls.csv` - start urls
        * `valid_htmls.csv` - valid htmls
    * `spiders/` - scrapy code
        * `crawldata.py` - scrapy code, to obtain data from htmls
        * `geturl.py` - scrapy code, to obtain htmls from start urls
    * `items.py` - scrapy project code, define items
    * `middlewares.py` - scrapy project code, define middlewares
    * `pipelines.py`- scrapy project code, define pipelines
    * `settings.py` - scrapy project code, define settings
* `utils/` - tool function
    * `baidu_get_LLitude.py` - get coordinates from baidu map
    * `gaode_get_LLitude.py` - get coordinates from gaode map
    * `tencent_get_LLitude.py` - get coordinates from tencent map
    * `BeautifulSoup.py` - BeautifulSoup crawler script, to obtain data from valid htmls
    * `del_invalid_urls.py` - delete invalid html urls
    * `delete_used_urls.py` - delete used html urls
* `scrapy.cfg` - scrapy project code, define settings
* `README.md` - README file
