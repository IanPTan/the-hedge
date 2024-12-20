import requests as rq
from lxml import html
from time import time
import pandas as pd


HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"}


INDEX_URL = "https://finance.yahoo.com/sitemap"


XPATHS = {
        "month": "/html/body/div[1]/div/main/div/div/div/ul/li/ul/li/a",
        "day": "/html/body/div[1]/div/main/div/div/div/div[1]/div/a",
        "article": "/html/body/div[1]/div/main/div/div/div/ul/li/a",
        "next": "/html/body/div[1]/div/main/div/div/div/div[2]/a",
        "tickers": "/html/body/div[2]/main/section/section/section/article/div/div[1]/div[3]/div[1]/div/div/div/div/div/a/div/span",
        "time": "/html/body/div[2]/main/section/section/section/article/div/div[1]/div[2]/div[1]/div/div[2]/time"
        }


def article_scan(url, xpaths=XPATHS, headers=HEADERS):

    page = rq.get(url, headers=HEADERS)

    if page.status_code != 200:
        raise ValueError(f"\"{url}\" failed to load: ERROR {page.status_code}")
    
    page_tree = html.fromstring(page.text)
    tickers = [element.text.strip() for element in page_tree.xpath(xpaths["tickers"])]
    time = page_tree.xpath(xpaths["time"]).text.strip()

    return tickers, time


if __name__ == "__main__":

    pass
