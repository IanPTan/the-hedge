import concurrent.futures as cf
import requests as rq
import pandas as pd
from lxml import html
from time import time, sleep
import datetime as dt
import yfinance as yf
from tqdm import tqdm


# 19632
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"}
INDEX_URL = "https://finance.yahoo.com/sitemap"
XPATHS = {
        "months": "/html/body/div[1]/div/main/div/div/div/ul/li/ul/li/a",
        "days": "/html/body/div[1]/div/main/div/div/div/div[1]/div/a",
        "articles": "/html/body/div[1]/div/main/div/div/div/ul/li/a",
        "next": "/html/body/div[1]/div/main/div/div/div/div[2]/a",
        "tickers": "/html/body/div[2]/main/section/section/section/article/div/div[1]/div[3]/div[1]/div/div/div/div/div/a/div/span",
        "time": "/html/body/div[2]/main/section/section/section/article/div/div[1]/div[2]/div[1]/div/div[2]/time",
        "text": "/html/body/div[2]/main/section/section/section/article/div/div[1]/div[3]/div[2]/p"
        }
EPOCH = dt.date(1970, 1, 1)
days_to_date = lambda days: EPOCH + dt.timedelta(days=days)
date_to_days = lambda date: (date - EPOCH).days
get_day = lambda date: f"https://finance.yahoo.com/sitemap/{date.year}_{date.month}_{date.day}"


def day_urls(date, end_date):

    while date <= end_date:
        yield get_day(date)
        date += dt.timedelta(days=1)


def get_page(url, headers=HEADERS):

    response = rq.get(url, headers=HEADERS)

    if response.status_code != 200:
        raise ValueError(f"\"{url}\" failed to load: ERROR {response.status_code}")

    page = html.fromstring(response.text)

    return page


def scan_day(days, patience=10, wait=10, xpaths=XPATHS, headers=HEADERS):

    date = days_to_date(days)
    day_url = get_day(date)
    #print(f"Scanning {day_url}")

    article_urls = []
    article_titles = []
    tries = 0
    while tries < patience:
        sleep(tries * wait)

        try:
            page = get_page(day_url)
        except ValueError as e:
            #print(f"\tRETRYING {e}")
            tries += 1
            continue
            
        articles = page.xpath(xpaths["articles"])
        for article in articles:
            if "href" in article.attrib:
                article_urls.append(article.attrib["href"])
                article_titles.append(article.text)

        next_elements = page.xpath(xpaths["next"])
        if not next_elements:
            #print("\tRETRYING")
            tries += 1
            continue
        tries = 0

        next_element = next_elements[-1]
        if next_element.text != "Next":
            #print(f"Successfully scraped {len(article_urls)} articles from {day_url}")
            return article_urls, article_titles, day_url, 1
        day_url = next_element.attrib["href"]

    #print("\tGIVING UP")
    return article_urls, article_titles, day_url, 0


def scan_days(all_days, workers=16, patience=10, wait=10, xpaths=XPATHS, headers=HEADERS):

    article_urls = []
    article_titles = []
    urls = []
    statuses = []
    counts = []
    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(scan_day, days, patience, wait, xpaths, headers) for days in all_days]
        for future in tqdm(cf.as_completed(futures), total=len(futures), desc="Scraping...", unit="days"):
            day_urls, day_titles, url, status = future.result()
            article_urls += day_urls
            article_titles += day_titles
            urls.append(url
            statuses.append(status)
            counts.append(len(day_urls))
    return article_urls, article_titles, url, statuses, counts


def scan_article(url, xpaths=XPATHS, headers=HEADERS):

    page = get_page(url, headers)

    tickers = [element.text.strip() for element in page.xpath(xpaths["tickers"])]
    time = page.xpath(xpaths["time"])[0].text.strip()

    return tickers, time


def yfin_scan(ticker, interval="30m", price_cols=["Open", "Close", "High", "Low", "Volume"]):

    ticker = yf.Ticker(ticker)
    prices = ticker.history(interval=interval)[price_cols]
    prices["time"] = prices.index.astype(int) // 10 ** 9
    prices.rename(columns={name: name.lower() for name in price_cols}, inplace=True)
    
    return prices
