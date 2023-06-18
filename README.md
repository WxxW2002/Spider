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
* `predict` - data preprocess and model train
    * `data/` - data after preprocess
    * `out/` - output directory
    * `nn_pred.py` - prediction
    * `preprocess.py` - data proprecess
    * `run.py` - model definition and train
    * `run.sh` - run script
    * `title_wordcloud.py` - make wordcloud for titles
    * `word_embedding.py` - word embedding
* `utils/` - tool function
    * `baidu_get_LLitude.py` - get coordinates from baidu map
    * `gaode_get_LLitude.py` - get coordinates from gaode map
    * `tencent_get_LLitude.py` - get coordinates from tencent map
    * `BeautifulSoup.py` - BeautifulSoup crawler script, to obtain data from valid htmls
    * `del_invalid_urls.py` - delete invalid html urls
    * `delete_used_urls.py` - delete used html urls
* `scrapy.cfg` - scrapy project code, define settings
* `README.md` - README file

Data Preprocess
--------------
```bash
# 数据预处理
python ./preprocess.py
# 词嵌入
python ./word_embedding.py
```

Model Prediction
--------------
```bash
# 传统模型预测（后两个参数只在Model为Multi-layer Perceptron时起效果）
python ./run.py --model [Model Name] --hidden_layer_sizes [隐藏层大小] --max_iter [最大迭代次数]
# 运行所有预测模型
chmod +x ./run.sh
./run.sh
# 神经网络预测
python ./nn_pred.py
```
