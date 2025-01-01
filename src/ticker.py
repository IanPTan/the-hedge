import concurrent.futures as cf
import requests as rq
import pandas as pd
from lxml import html
from time import time, sleep
import datetime as dt
import yfinance as yf
from tqdm import tqdm


HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"}
INDEX_URL = "https://finance.yahoo.com/sitemap"
XPATHS = {
        "months": "/html/body/div[1]/div/main/div/div/div/ul/li/ul/li/a",
        "days": "/html/body/div[1]/div/main/div/div/div/div[1]/div/a",
        "articles": "/html/body/div[1]/div/main/div/div/div/ul/li/a",
        "next": "/html/body/div[1]/div/main/div/div/div/div[2]/a",
        "tickers": "/html/body/div[2]/main/section/section/section/article/div/div[1]/div[3]/div[1]/div/div/div/div/div/a/div/span",
        "time": "/html/body/div[2]/main/section/section/section/article/div/div[1]/div[2]/div[1]/div/div[2]/time",
        "paragraphs": "/html/body/div[2]/main/section/section/section/article/div/div[1]/div[3]/div[2]/p"
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


def scan_day(day_url, patience=5, wait=60, xpaths=XPATHS, headers=HEADERS):

    article_urls = []
    article_titles = []
    tries = 0

    while tries < patience:
        sleep(tries * wait)

        try:
            page = get_page(day_url)
        except Exception as e:
            tries += 1
            continue
            
        articles = page.xpath(xpaths["articles"])
        for article in articles:
            if "href" in article.attrib:
                article_urls.append(article.attrib["href"])
                article_titles.append(article.text)

        next_elements = page.xpath(xpaths["next"])
        if not next_elements:
            tries += 1
            continue
        tries = 0

        next_element = next_elements[-1]
        if next_element.text != "Next":
            return article_urls, article_titles, day_url, 1
        day_url = next_element.attrib["href"]

    return article_urls, article_titles, day_url, 0


def scan_days(day_urls, workers=1, patience=10, wait=10, xpaths=XPATHS, headers=HEADERS):

    article_urls = []
    article_titles = []
    urls = []
    statuses = []
    counts = []

    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(scan_day, day_url, patience, wait, xpaths, headers) for day_url in day_urls]

        for future in tqdm(cf.as_completed(futures), total=len(futures), desc="Scraping...", unit="day"):
            day_urls, day_titles, url, status = future.result()
            article_urls += day_urls
            article_titles += day_titles
            urls.append(url)
            statuses.append(status)
            counts.append(len(day_urls))

    return article_urls, article_titles, urls, statuses, counts


def scan_article(url, patience=10, wait=10, xpaths=XPATHS, headers=HEADERS):

    tries = 0
    while tries < patience:

        try:
            sleep(tries * wait)
            page = get_page(url, headers)

            tickers = [element.text.strip() for element in page.xpath(xpaths["tickers"])]
            time_str = page.xpath(xpaths["time"])[0].text.strip()
            time = pd.to_datetime(time_str).tz_localize('America/New_York')
            text = "\n".join("".join(paragraph.itertext()) for paragraph in page.xpath(xpaths["paragraphs"])).replace("\xa0", "")

            return tickers, text, time, url, 1
        except ValueError:
            tries += 1
            print(f"404 waiting {tries * wait}s for {url}")
            continue
        except Exception as e:
            tries += 1
            print(f"\n\n\n\n\n\tACHTUNG\n\t{e}\n\t{url}\n\tWaiting {tries * wait} seconds\n\n\n\n\n")
            continue

    print(f"\n\n\n\n\n\tACHTUNG [FAILED]\n\t{url}\n\n\n\n\n")
    return [], "", pd.to_datetime(0), url, 0


def scan_articles(article_urls, filtered=True, workers=1, patience=3, wait=60, xpaths=XPATHS, headers=HEADERS):

    tickers = []
    text = []
    times = []
    urls = []
    statuses = []

    with cf.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(scan_article, article_url, patience, wait, xpaths, headers) for article_url in article_urls]

        for future in tqdm(cf.as_completed(futures), total=len(futures), desc="Scraping...", unit="article"):
            try:
                article_tickers, article_text, article_time, url, status = future.result()
                if not article_tickers and filtered:
                    continue
                tickers.append(article_tickers)
                text.append(article_text)
                times.append(article_time)
                urls.append(url)
                statuses.append(status)
            except Exception as e:
                print(f"\n\n\n\n\n\tACHTUNG [COMPLETE FAILURE]\n\t{e}\n\n\n\n\n")

    return tickers, text, times, urls, statuses


def scan_yfin(ticker, interval="1h", period="max", patience=4, wait=10, price_cols=["Open", "Close", "High", "Low", "Volume"]):

    ticker = yf.Ticker(ticker)

    tries = 0
    while tries < patience:
        sleep(tries * wait)
        prices = ticker.history(interval=interval, period=period)[price_cols]
        if len(prices):
            prices["time"] = prices.index.astype(int) // 10 ** 9
            prices.rename(columns={name: name.lower() for name in price_cols}, inplace=True)
            
            return prices
        tries += 1
        print(f"FAILED {ticker} waiting {tries * wait}s")

    return None



def scan_change(prices, article_time, hist_win=12, spike_win=1, spike_dis=1.5):
    
    prices_left = prices[prices["time"] < article_time]
    prices_before = prices_left.iloc[-hist_win:]
    prices_right = prices[article_time <= prices["time"]]
    prices_after = prices_right.iloc[:spike_win]

    close_before = prices_before["close"].sort_values()
    iqr = close_before.quantile(0.75) - close_before.quantile(0.25)
    m = close_before.quantile(0.5)

    upper_bound = m + iqr * spike_dis
    lower_bound = m - iqr * spike_dis

    highest = prices_after.loc[prices_after["high"].idxmax()]
    lowest = prices_after.loc[prices_after["low"].idxmin()]

    condition = lowest["low"] < lower_bound, highest["high"] > upper_bound, lowest["time"] > highest["time"]
    spike_case = {
            (0, 0, 0): 4,
            (0, 0, 1): 4,
            (0, 1, 0): 3,
            (0, 1, 1): 3,
            (1, 0, 0): 0,
            (1, 0, 1): 0,
            (1, 1, 0): 1,
            (1, 1, 1): 2
            }[condition]

    return spike_case
