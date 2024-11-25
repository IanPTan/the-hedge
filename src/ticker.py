from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
import requests as rq
from lxml import html
from time import time

driver_path = "/usr/bin/geckodriver"
service = Service(driver_path)
options = Options()
#options.add_argument("--headless")
driver = webdriver.Firefox(service=service, options=options)
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"}

url = "https://www.google.com/search?q=fda+site:https://finance.yahoo.com/news/&sca_esv=a3aa32e04255f5fb&source=lnt&tbs=qdr:m&sa=X&ved=2ahUKEwif45GVl_iJAxUDmokEHdgPDogQpwV6BAgFEAo&biw=1707&bih=944&dpr=1.5"
xpaths = {
        "results": "/html/body/div[3]/div/div[13]/div/div[2]/div[2]/div/div/div/div/div/div/div[1]/div/div/span/a | /html/body/div[3]/div/div[13]/div/div[2]/div[2]/div/div/div/div/div/div[1]/div/div/span/a",
        "next": "/html/body/div[3]/div/div[13]/div/div[4]/div/div[3]/table/tbody/tr/td[12]/a",
        "tickers": "/html/body/div[2]/main/section/section/section/article/div/div[1]/div[3]/div[1]/div/div/div/div/div/a/div/span",
        "robot": "/html/body/div[1]/div/b",
        }


def ticker_scan(url, xpaths=xpaths, headers=headers):

    page = rq.get(url, headers=headers)

    if page.status_code != 200:
        raise ValueError(f"\"{url}\" failed to load: ERROR {page.status_code}")

    return [element.text.strip() for element in html.fromstring(page.text).xpath(xpaths["tickers"])]


def raw_google_scan(driver, url, xpaths=xpaths):

    driver.get(url)
    results = []
    while 1:

        if driver.find_elements(By.XPATH, xpaths["robot"]):
            if pause:
                _ = input("Captcha detected! Completed?: ")
            else:
                return tickers

        results += [element.get_dom_attribute("href") for element in driver.find_elements(By.XPATH, xpaths["results"])]

        next = driver.find_elements(By.XPATH, xpaths["next"])
        if next:
            next[0].click()
            if driver.find_elements(By.XPATH, xpaths["robot"])[0].text == "About this page" and pause:
                _ = input("Captcha detected! Completed?: ")
                continue
        else:
            return results


def google_scan(driver, url, xpaths=xpaths, headers=headers, pause=1):

    driver.get(url)
    tickers = set()
    while 1:

        for element in driver.find_elements(By.XPATH, xpaths["results"]):
            article_url = element.get_dom_attribute("href")
            try:
                tickers = tickers.union(set(ticker_scan(article_url, xpaths, headers)))
                print(f"Successfully scraped: {article_url}")
            except:
                print(f"Failed to scrape: {article_url}")

        next = driver.find_elements(By.XPATH, xpaths["next"])
        if next:
            next[0].click()
        else:
            robot_element = driver.find_elements(By.XPATH, xpaths["robot"])
            if robot_element and robot_element[0].text == "About this page" and pause:
                _ = input("Captcha detected! Completed?: ")
                continue
            return tickers


results = google_scan(driver, url, xpaths, headers, 1)
