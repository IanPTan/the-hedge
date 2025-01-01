from ticker import *

article_title = "FDA approves Ionis’ TRYNGOLZA for FCS treatment"
articles = pd.read_csv("articles.csv")
article = articles[articles["title"] == article_title]

prices = scan_yfin("ions")
hist_win = 6
spike_win = 1
spike_dis = 1.5

article_time = article["time"].item()
label = scan_change(prices, article_time, hist_win=hist_win, spike_win=spike_win, spike_dis=spike_dis)


cases = {0: "negative spike",
         1: "negative spike followed by positive spike",
         2: "positive spike followed by negative spike",
         3: "positive spike",
         4: "negligible change",
         }
print(f"Detected {cases[label]}")
