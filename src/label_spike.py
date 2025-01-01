from ticker import *


hist_win = 4
spike_win = 8
spike_dis = 0.15

articles = pd.read_csv("articles.csv")


def scan_change(prices_before, prices_after, spike_dis=0.15):
    
    close_before = prices_before["open"].sort_values()
    m = close_before.quantile(0.5)

    upper_bound = m + m * spike_dis
    lower_bound = m - m * spike_dis

    highest = prices_after.loc[prices_after["high"].idxmax()]
    lowest = prices_after.loc[prices_after["low"].idxmin()]

    condition = lowest["low"] <= lower_bound, highest["high"] >= upper_bound, lowest["time"] >= highest["time"]
    spike_case = {
            (0, 0, 0): 0,
            (0, 0, 1): 0,
            (0, 1, 0): 2,
            (0, 1, 1): 2,
            (1, 0, 0): 1,
            (1, 0, 1): 1,
            (1, 1, 0): 2,
            (1, 1, 1): 1
            }[condition]

    return spike_case



dataset = pd.DataFrame(columns=["url", "ticker", "time", "title", "text", "label"])
pairs = 0
successes = 0
with pd.HDFStore("prices.h5", 'r') as store:
    valid_keys = store.keys()
    for i, article in tqdm(articles.iterrows(), total=len(articles), desc="Processing...", unit="article"):
        labels = []
        for ticker in eval(article["tickers"]):
            pairs += 1
            key = f"/{ticker}"
            if key not in valid_keys:
                print(f"{ticker} missing")
                continue

            prices = store.get(key)

            prices_left = prices[prices["time"] <= article["time"]]
            prices_right = prices[prices["time"] >= article["time"]]

            left_out = len(prices_left) < hist_win
            right_out = len(prices_right) < spike_win
            if left_out or right_out:
                print(f"{ticker} not enough data\n\t{len(prices_left)} to left\n\t{len(prices_right)} to right")
                continue

            prices_before = prices_left.iloc[-hist_win:]
            prices_after = prices_right.iloc[:spike_win]

            left_dis = (article["time"] - prices_before.iloc[-1]["time"]) / 60 ** 2
            right_dis = (prices_after.iloc[0]["time"] - article["time"]) / 60 ** 2
            left_far = left_dis > 72
            right_far = right_dis > 72
            if left_far or right_far:
                print(f"{ticker} data too sparse\n\t{left_dis}h to left\n\t{right_dis}h to right")
                continue

            if prices_before.isnull().any().any() or prices_after.isnull().any().any():
                print(f"{ticker} data is null")
                continue

            label = scan_change(prices_before, prices_after, spike_dis=spike_dis)
            dataset.loc[len(dataset)] = [article["url"], ticker, *article[["time", "title", "text"]], label]
            successes += 1

dataset.to_csv("dataset.csv", index=False)
print(f"Successfully labeled {successes}/{pairs} pairs of articles and tickers")
