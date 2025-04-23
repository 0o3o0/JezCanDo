from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import pandas as pd
import random

# === CONFIG ===
EMAIL = "Insert_your_LinkedIn_email_here"
PASSWORD = "Insert_your_password"
BASE_SEARCH_URL = (
    "https://www.linkedin.com/search/results/people/?geoUrn=%5B%22101620260%22%5D&keywords=finance&origin=FACETED_SEARCH&sid=e6%40"
)
START_PAGE = 4     # מאיזה עמוד להתחיל
END_PAGE = 8      # איפה לעצור
SCROLL_PAUSE = 2
PAGE_LOAD_SLEEP = 3

# === DRIVER SETUP ===
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
# options.add_argument('--headless')  # uncomment to run headless

driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 15)

# === LOGIN ===
print("🔐 Logging into LinkedIn...")
driver.get("https://www.linkedin.com/login")
wait.until(EC.presence_of_element_located((By.ID, "username"))).send_keys(EMAIL)
driver.find_element(By.ID, "password").send_keys(PASSWORD)
driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
try:
    wait.until(lambda d: "feed" in d.current_url)
    print("✅ Login successful")
except TimeoutException:
    print("⚠️ Login may have failed; proceeding...")

# === PHASE 1: SCRAPE PROFILES ===
profiles = []
for page in range(START_PAGE, END_PAGE + 1):
    url = f"{BASE_SEARCH_URL}&page={page}"
    print(f"📄 Loading search page {page}: {url}")
    driver.get(url)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(SCROLL_PAUSE + random.random())
    cards = driver.find_elements(By.CSS_SELECTOR, "div[data-view-name='search-entity-result-universal-template']")
    print(f"🔎 Found {len(cards)} profile cards on page {page}")
    for card in cards:
        try:
            link_elem = card.find_element(By.CSS_SELECTOR, "a[data-test-app-aware-link]")
            link = link_elem.get_attribute("href").split("?")[0]
            name = card.find_element(By.CSS_SELECTOR, "span.t-16 span[aria-hidden='true']").text.strip()
            try:
                title = card.find_element(By.CSS_SELECTOR, "div.entity-result__primary-subtitle").text.strip()
            except NoSuchElementException:
                try:
                    title = card.find_element(By.CSS_SELECTOR, "div.t-black--light").text.strip()
                except NoSuchElementException:
                    title = "N/A"
            print(f"▶️ Scraped: {name} | {title} | {link}")
            profiles.append({"Name": name, "Title": title, "LinkedIn URL": link})
        except Exception as e:
            print(f"❌ Error scraping card: {e}")

# === PHASE 2: SEND CONNECTION REQUESTS ===
results = []
for entry in profiles:
    name = entry['Name']
    title = entry['Title']
    link = entry['LinkedIn URL']
    print(f"\n🔗 Visiting {name}: {link}")
    driver.get(link)
    time.sleep(PAGE_LOAD_SLEEP)

    # Debug: list button texts and aria-labels
    for b in driver.find_elements(By.TAG_NAME, "button"):
        txt = b.text.strip()
        aria = b.get_attribute('aria-label') or ''
        if txt or aria:
            print(f"    ▶️ text={txt!r}, aria_label={aria!r}")

    status = ""
    try:
        # Send new connection request
        invite_buttons = driver.find_elements(
            By.XPATH,
            "//button[contains(@aria-label,'Invite') and contains(@aria-label,'to connect')]"
        )
        selected = next(
            (btn for btn in invite_buttons if name in (btn.get_attribute('aria-label') or '')),
            None
        )
        if selected:
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", selected)
            time.sleep(0.5)
            driver.execute_script("arguments[0].click();", selected)
            print(f"🎯 Clicked Invite for {name}")
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[role='dialog']")))
            send_btn = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[.//span[text()='Send without a note']]"))
            )
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", send_btn)
            time.sleep(0.5)
            driver.execute_script("arguments[0].click();", send_btn)
            print("🎯 Clicked 'Send without a note'")
            status = "Connection request sent"
            time.sleep(1)
        else:
            raise TimeoutException("No matching Invite button for user")
    except TimeoutException as e:
        print(f"⚠️ Invite flow unavailable: {e}")
        if driver.find_elements(By.XPATH, "//button[.//span[text()='Pending']]"):
            status = "Request Sent"
        elif driver.find_elements(By.XPATH, "//button[.//span[text()='Follow']]"):
            status = "Not connected - follow only"
    except NoSuchElementException as e:
        print(f"⚠️ Element missing: {e}")
    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")
    print(f"✅ {name} -> {status}")
    results.append({"Name": name, "Title": title, "LinkedIn URL": link, "Connection": status})

# === SAVE RESULTS ===
print("💾 Saving results to CSV...")
try:
    driver.quit()
except Exception as e:
    print(f"⚠️ Error quitting driver: {e}")

df = pd.DataFrame(results, columns=["Name", "Title", "LinkedIn URL", "Connection"])
df.drop_duplicates(subset=['LinkedIn URL'], inplace=True)
df.to_csv('linkedin_leads1.csv', index=False)
print("✅ Done! Leads saved to linkedin_leads1.csv")
