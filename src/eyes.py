from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
import pandas as pd
import torch as pt
from tokenizers import Tokenizer
from model import Model
import matplotlib.pyplot as plt
import h5py as hp
from utils import Embedder, pca
from tqdm import tqdm


xpaths = {
        "link": "/html/body/div[3]/div/div[11]/div/div/div[2]/div[2]/div/div/div/div/div/div/div/a",
        "title": "/html/body/div[3]/div/div[11]/div/div/div[2]/div[2]/div/div/div/div/div/div/div/a/div/div[2]/div[2]",
        "next": "/html/body/div[3]/div/div[11]/div/div/div[4]/div/div[3]/table/tbody/tr/td/a/span[2]",
        "robot": "/html/body/div[1]/div/b",
        }


def get_url(search):
    query = search.replace(" ", "+")
    return f"https://www.google.com/search?q={query}&tbs=qdr:d,sbd:1&tbm=nws&source=lnt&sa=X"


def google_scan(driver, searches, pause=True, xpaths=xpaths):


    results = pd.DataFrame(columns=["url", "title"])

    for search in searches:

        url = get_url(search)
        driver.get(url)

        while 1:

            robot = driver.find_elements(By.XPATH, xpaths["robot"])
            if robot and robot[0].text == "About this page":
                if pause:
                    _ = input("Captcha detected! Completed?: ")
                else:
                    break

            link_els = driver.find_elements(By.XPATH, xpaths["link"])
            title_els = driver.find_elements(By.XPATH, xpaths["title"])
            for link_el, title_el in zip(link_els, title_els):
                if title_el.text not in results["title"]:
                    results.loc[len(results)] = [link_el.get_dom_attribute("href"), title_el.text]

            next = driver.find_elements(By.XPATH, xpaths["next"])

            if not next or next[-1].text != "Next":
                break

            next[-1].click()

    return results


if __name__ == "__main__":
    driver_path = "/usr/bin/geckodriver"
    service = Service(driver_path)
    options = Options()
    #options.add_argument("--headless")
    driver = webdriver.Firefox(service=service, options=options)

    searches = [
            "Biotech Catalyst News",
            "FDA News",
            "FDA Approval News",
            "Pipeline Drug News",
            ]
    starts = [
            "Biotech",
            "Pharmaceutical",
            "Healthcare",
            "Therapeutics",
            "Medical",
            ]
    ends = [
            "News",
            "Catalyst News",
            "Price Target News",
            "News Release",
            "Quartlery Report News",
            ]
    for start in starts:
        for end in ends:
            searches.append(f"{start} {end}")
    results = google_scan(driver, searches, 1)
    results.to_csv("scan.backup.csv", index=False)
    #driver.quit()

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
RWKV_FILE = "../model/RWKV-4-Pile-430M-20220808-8066.pth"
TOKENIZER_FILE = "../model/20B_tokenizer.json"
model_file = "model.ckpt"
N_LAYER = 24
N_EMBD = 1024
batch_size = 128

print(f"Loading {TOKENIZER_FILE} and {RWKV_FILE}...")
embed = Embedder(TOKENIZER_FILE, RWKV_FILE, N_LAYER, device=device)

print(f"Loading {model_file}...")
model = Model(features=[1024, 512, 256, 128, 64, 32, 2]).to(device)
weights = pt.load(model_file)
model.load_state_dict(weights)
model.eval()

results_len = len(results)
print(f"Embedding {results_len} articles...")
embs = pt.zeros((results_len, N_EMBD), pt.float32, device=device)
batch_amnt = results_len // batch_size + (results_len % batch_size > 0)
for start in tqdm(range(0, results_len, batch_size), desc="Embedding...", unit="batch"):
    batch = slice(start, start + batch_size)
    text = results["title"][batch]
    embs[batch] = embed(text)

print(f"Predicting...") 
results["labels"] = model(embs)
