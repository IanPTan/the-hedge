from ticker import *


articles = pd.read_csv("articles.csv")

unique_tickers = set()
for i, article in tqdm(articles.iterrows(), total=len(articles), desc="Scanning...", unit="article"):
    unique_tickers = unique_tickers.union(set(eval(article["tickers"])))


fails = []
with pd.HDFStore("prices.h5", 'a') as store:
    for ticker in tqdm(unique_tickers, desc="Scraping...", unit="ticker"):
        if f"/{ticker}" in store:
            #print(f"{ticker} is already stored")
            continue
        prices = scan_yfin(ticker, interval="1h", period="max", patience=2, wait=10, price_cols=["Open", "Close", "High", "Low", "Volume"])
        if prices is None:
            fails.append(ticker)
            print(f"{ticker} not found in yfinance.")
        else:
            store.put(ticker, prices, format="table", data_columns=True)

    print(f"Successfully scraped {len(store)}/{len(unique_tickers)}.")

with open("logs2.txt", "w") as file:
    file.write("\n".join(fails))
