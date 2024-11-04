from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


driver = webdriver.Chrome()  # Replace with your preferred browser

def extract_article_urls(url):
    text_list = []
    driver.get(url)
    restart = True
    while restart==True:
        # takes all of the article links
        links = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.XPATH, "/html/body/div[3]/div/div[13]/div/div[2]/div[2]/div/div/div/div/div/div[1]/div/div/span/a"))  # xpath for the article
        )

        article_urls = []
        #finds all of the links to the articles

        for link in links:
            article_url = link.get_attribute('href')
            article_urls.append(article_url)
        #open text list

        for article_url in article_urls:
            # Open a new tab
            driver.execute_script("window.open('');")
            #opens up the given article

            # switch to the new tab
            window_after = driver.window_handles[1]
            driver.switch_to.window(window_after)

            driver.get(article_url)

            if check_xpath_existence(driver,"/html/body/div[2]/main/section/section/section/article/div/div[1]/div[3]/div[1]/div/div/div/div/div/a/div/span"):
                # the xpath exists
                element = WebDriverWait(driver, 20).until(
                    EC.presence_of_element_located((By.XPATH, "/html/body/div[2]/main/section/section/section/article/div/div[1]/div[3]/div[1]/div/div/div/div/div/a/div/span"))
                )
                text = element.text.strip()
                if text not in text_list:
                    text_list.append(text)

            #closes the tab
            driver.close()
            driver.switch_to.window(driver.window_handles[0])

            #Code to see if there is a next page
        if check_xpath_existence(driver,"/html/body/div[3]/div/div[13]/div/div[4]/div/div[3]/table/tbody/tr/td[12]/a" ):
            # Find the "Next" button
            next_button = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.XPATH, "/html/body/div[3]/div/div[13]/div/div[4]/div/div[3]/table/tbody/tr/td[12]/a")))
            # Click the "Next" button
            next_button.click()
        else:
            return text_list


#checks if it exists
def check_xpath_existence(driver, xpath):
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, xpath))
        )
        return True
    except:
        return False

# Example usage:
url = "https://www.google.ca/search?q=site:finance.yahoo.com+%22FDA%22&sca_esv=69b2d222702df3a7&source=lnms&fbs=AEQNm0CrHVBV9axs7YmgJiq-TjYc7RgyMjmhctvLCnk5YpVfOzTk9UgrCkq1LL6wECoQ_WHzfZcSELpXaEiujjL7Q6oq7xqg_ok_kSsBaDM483VRK4lyvIYAvA84z9GW-6XFz0AOQRHB2zMuinHUAngvDSQ835MgWH4RST5egs07mXF9v4oFWz17sYS05cF7cYs8Af6knIYVcIflpfPTXDGSq6aOvTMJ5Q&sa=X&ved=2ahUKEwj_vvqso7-JAxWThIkEHT2iMCMQ0pQJegQIBhAD&biw=1440&bih=669&dpr=1"
text_list = extract_article_urls(url)

for text in text_list:
    print(text)
#for url in article_urls:
  #print(url)

