from ticker import *


all_articles = pd.read_csv("all_articles.csv")
keywords = "FDA|Biotech|Pharma|Therapu|Healthcare|Medic|Clinic|Tech|AI|Communic|Acqui|Buyout|Ware|Systems|Manufac|Qua|Tele|Electronic|Chip|Data|Debt|Credit|Loan|Fund|Performance|Service"
# keywords = "FDA|Biotech|Pharma|Therapu|Healthcare|Medic"
valid_titles = all_articles["title"].str.contains(keywords)
valid_urls = all_articles["url"].str.contains("https://finance.yahoo.com/news/")
#filtered = valid_titles & valid_urls
filtered = valid_urls
all_articles = all_articles[filtered]
#all_articles = all_articles

article_urls = all_articles["url"]

start = time()
article_tickers, article_text, article_times, urls, statuses = scan_articles(article_urls, True, 16, 5, 60)
dur = time() - start

data = pd.DataFrame({"url": urls, "tickers": article_tickers, "text": article_text}, index=article_times)
data["time"] = data.index.astype(int) // 10 ** 9
data = pd.merge(data, all_articles, on='url', how='left')

data_og = data.copy()
i = 0
while i < len(data) - 1:
    if (data.loc[i] == data.loc[i + 1]).all():
        data = data.drop(i)
    i += 1

data.to_csv("articles.csv", index=False)

logs = pd.DataFrame({"url": urls, "status": statuses})
logs.to_csv("logs1.csv", index=False)

print(f"Took {dur:.4f} seconds to scrape {len(urls)} articles, {sum(statuses)}/{len(urls)} days succeeded")
