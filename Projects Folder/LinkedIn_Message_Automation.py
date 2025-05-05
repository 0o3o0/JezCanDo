# send_linkedin_messages.py

import time
import random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# === CONFIG ===
EMAIL       = "INSERT YOUR LINKEDIN EMAIL ADDRESS"
PASSWORD    = "INSERT YOUR PASSWORD"
INPUT_FILE  = r"INPUT HERE ROUTE TO THE FILE"
OUTPUT_FILE = r"INPUT HERE ROUTE TO THE FOLDER WITH THE REQUIRED FILE NAME AND THE ADDITION OF TYPE CSV"

MESSAGES = [
    "Hi {first},  1",
    "Hey {first}, 2",
    "Hi {first}, 3",
    "Hello {first}, 4"
]

# === SETUP DRIVER ===
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(options=options)
wait = WebDriverWait(driver, 15)


def linkedin_login():
    driver.get("https://www.linkedin.com/login")
    wait.until(EC.presence_of_element_located((By.ID, "username"))).send_keys(EMAIL)
    driver.find_element(By.ID, "password").send_keys(PASSWORD)
    driver.find_element(By.CSS_SELECTOR, "button[type='submit']").click()
    wait.until(lambda d: "feed" in d.current_url)
    print("✅ Logged in")


def send_message_only(profile_url, name):
    driver.get(profile_url)
    time.sleep(3)
    first = name.split()[0]

    # 1) Open message dialog
    try:
        buttons = driver.find_elements(By.XPATH, "//button[.//span[text()='Message']]")
        for b in buttons:
            if b.is_displayed() and b.is_enabled():
                b.click()
                break
        else:
            print(f"⚠️ No visible 'Message' button for {name}")
            return "Failed"
    except Exception as e:
        print(f"⚠️ Error opening message for {name}: {e}")
        return "Failed"

    # 2) Type the message
    try:
        input_box = wait.until(EC.presence_of_element_located((
            By.CSS_SELECTOR,
            "div.msg-form__contenteditable[contenteditable='true']"
        )))
        msg = random.choice(MESSAGES).format(first=first)
        input_box.clear()
        input_box.send_keys(msg)
    except Exception as e:
        print(f"⚠️ Error typing message for {name}: {e}")
        return "Failed"

    # 3) Click send and verify
    try:
        send_btn = wait.until(EC.element_to_be_clickable((
            By.CSS_SELECTOR,
            "div.msg-form__right-actions button.msg-form__send-button"
        )))
        send_btn.click()
        # wait for input box to clear, indicating send succeeded
        def box_cleared(driver):
            txt = driver.find_element(By.CSS_SELECTOR, "div.msg-form__contenteditable[contenteditable='true']").text
            return txt.strip() == ''
        wait.until(box_cleared)
        return "Message Sent"
    except Exception as e:
        print(f"⚠️ Error sending message for {name}: {e}")
        return "Failed"


def main():
    linkedin_login()
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["LinkedIn URL"])
    results = []

    for _, row in df.iterrows():
        name = row["Name"]
        url  = row["LinkedIn URL"]
        print(f"→ Processing {name}")
        status = send_message_only(url, name)
        print(f"↪️ Status: {status}")
        results.append(status)
        time.sleep(random.uniform(2, 4))

    df['Result'] = results
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Done! Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
