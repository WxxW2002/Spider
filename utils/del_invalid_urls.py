import requests
import csv
from requests.exceptions import RequestException

def check_url(url):
    try:
        response = requests.get(url, timeout=3)  # 设置超时时间为5秒
        if response.status_code == 200:
            return True
        else:
            return False
    except RequestException:  # 如果发生网络错误（如超时、网络未连接等），则返回False
        return False

def clean_urls(input_file, output_file):
    with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        for row in reader:
            url = row[0]
            if check_url(url):
                writer.writerow([url])

clean_urls('../Spider/data/new_htmls.csv', '../Spider/data/valid_htmls.csv')