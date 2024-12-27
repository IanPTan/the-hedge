import concurent.futures as cf
import requests as rq
import pandas as pd
from lxml import html
from time import time
import datetime as dt


HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"}
INDEX_URL = "https://finance.yahoo.com/sitemap"
XPATHS = {
        "months": "/html/body/div[1]/div/main/div/div/div/ul/li/ul/li/a",
        "days": "/html/body/div[1]/div/main/div/div/div/div[1]/div/a",
        "articles": "/html/body/div[1]/div/main/div/div/div/ul/li/a",
        "next": "/html/body/div[1]/div/main/div/div/div/div[2]/a",
        "tickers": "/html/body/div[2]/main/section/section/section/article/div/div[1]/div[3]/div[1]/div/div/div/div/div/a/div/span",
        "time": "/html/body/div[2]/main/section/section/section/article/div/div[1]/div[2]/div[1]/div/div[2]/time"
        }
EPOCH = dt.date(1970, 1, 1)

days_to_date = lambda days: EPOCH + dt.delta_time(days=days)
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


def scan_day(days, patience=3, xpaths=XPATHS, headers=HEADERS):

    date = days_to_date(days)
    day_url = get_url(date)

    article_urls = []
    article_titles = []
    while tries > patience:
        print(f"Scanning {day_url}")
        page = get_page(day_url)
        
        articles = page.xpath(xpaths["articles"])
        for article in articles:
            article_urls.append(article.attrib["href"])
            article_titles.append(article.text)

        next_elements = page.xpath(xpaths["next"])
        if not next_elements:
            print("\tRETRYING")
            tries += 1
            continue
        tries = 0

        next_element = next_elements[-1]
        if next_element.text != "Next":
            return article_urls, article_titles
        day_url = next_element.attrib["href"]

    return article_urls, article_titles


def scan_days(all_days, workers=16, patience=3, xpaths=XPATHS, headers=HEADERS):

    article_urls = []
    article_titles = []
    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, days, patience, xpaths, headers) for days in all_days]
        for future in cf.as_completed(futures):
            day_urls, day_titles = future.result()
            article_urls += day_urls
            article_titles += day_titles
    return article_urls, article_titles


def scan_article(url, xpaths=XPATHS, headers=HEADERS):

    page = get_page(url, headers)

    tickers = [element.text.strip() for element in page.xpath(xpaths["tickers"])]
    time = page.xpath(xpaths["time"])[0].text.strip()

    return tickers, time


if __name__ == "__main__":

    start_date = dt.date(2024, 12, 20)
    start_days = date_to_days(start_date)
    end_date = dt.date(2024, 12, 27)
    end_days = date_to_days(end_date)
    all_days = range(start_days, end_days + 1)
    urls, titles = scan_days(all_days)
