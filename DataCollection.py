"""
《红楼梦》全文数据采集脚本
----------------------------
功能：从古典名著网站抓取《红楼梦》120 回正文，按章节保存为本地 txt 文件。

数据源：http://www.gudianmingzhu.com/guji/hongloumeng/
输出目录：红楼梦/
文件命名：{回次序号}_{完整标题}.txt（UTF-8 编码）

依赖：requests, beautifulsoup4, lxml, tqdm
用法：python DataCollection.py
"""

import requests
import re
import os
from bs4 import BeautifulSoup
from lxml import etree  # noqa: F401 — 供 BeautifulSoup(..., 'lxml') 解析器使用
from tqdm import tqdm

# =============================================================================
# 1. 初始化：若不存在则创建本地存储目录
# =============================================================================
if not os.path.exists('红楼梦'):
    os.mkdir('红楼梦')

# =============================================================================
# 2. 抓取索引页：解析 120 回的目录标题与正文链接
# =============================================================================
url = 'http://www.gudianmingzhu.com/guji/hongloumeng/'

# 设置 User-Agent 伪装浏览器，避免被目标站点拒绝访问
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36'
}

page_text = requests.get(url=url, headers=headers)
page_text.encoding = page_text.apparent_encoding  # 自动识别响应编码
page_text = page_text.text

soup = BeautifulSoup(page_text, 'lxml')
aTagList = soup.select('#leftdg > div.dgcon.ui > div > a')  # 目录区全部章节 <a> 标签
titleList = [i.text for i in aTagList]   # 各回在目录中显示的标题
urlList = [i["href"] for i in aTagList]  # 各回正文页面的 URL

# =============================================================================
# 3. 逐回抓取正文、清洗文本并写入本地
# =============================================================================
progress_bar = tqdm(total=120, desc='抓取章节')

for index in range(1, 121):
    progress_bar.update(1)

    title = titleList[index - 1]
    url = urlList[index - 1]

    # 请求单章页面（超时 10 秒）
    page_text = requests.get(url=url, headers=headers, timeout=10)
    page_text.encoding = page_text.apparent_encoding
    page_text = page_text.text

    soup = BeautifulSoup(page_text, 'lxml')

    # 定位正文容器，拼接所有段落 <p> 的纯文本
    tmp = soup.select('#leftdg > div:nth-child(1) > div > div')
    content = ''.join([p.text for result in tmp for p in result.find_all('p')])

    # 正文首段通常包含完整回目标题
    titleText = soup.select('#leftdg > div:nth-child(1) > div > div > p:nth-child(1)')[0].text

    # 将连续两个全角空格（U+3000）替换为换行符，保留段落换行
    pattern = re.compile(r'\u3000{2}')
    content = re.sub(pattern, '\n', content)

    # 去除标题中的全角空格，避免文件名含不可见字符
    titleText = titleText.replace('\u3000', '')

    # 合并目录标题与页面内标题，构成完整文件名
    title = titleList[index - 1] + titleText

    chapter_path = '红楼梦/{}_{}.txt'.format(index, title)

    # 若同名文件已存在则先删除，再写入最新抓取内容
    if os.path.exists(chapter_path):
        os.remove(chapter_path)

    with open(chapter_path, 'w', encoding='utf-8') as f:
        f.write(content)

progress_bar.close()
