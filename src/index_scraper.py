from ticker import *


start_date = dt.date(2023, 2, 1)
end_date = dt.date(2024, 12, 26)

start_days = date_to_days(start_date)
end_days = date_to_days(end_date)

day_urls = day_urls(start_date, end_date)

start = time()
article_urls, article_titles, urls, statuses, counts = scan_days(day_urls, 1)
dur = time() - start

success = 0
for status in statuses:
    if status[0] == "S":
        success += 1

data = pd.DataFrame({"url": article_urls, "title": article_titles})
data.to_csv("all_articles.csv", index=False)

logs = pd.DataFrame({"url": urls, "status": statuses, "count": counts})
logs.to_csv("logs.csv", index=False)

print(f"Took {dur:.4f} seconds to scrape {len(urls)} articles, {success}/{end_days - start_days + 1} days succeeded")
