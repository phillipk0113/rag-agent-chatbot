from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup

def extract_dynamic_content(url):
    with sync_playwright() as p:
        # Launch a headless browser instance
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        # Navigate to the URL
        page.goto(url)
        page.wait_for_load_state('networkidle')  # Wait for the page to fully load

        # Get the page content
        html = page.content()

        # Close the browser
        browser.close()

        # Parse the content with BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

         # Extract desired elements: <h1> through <h5> and <p>
        tags_to_extract = ['h1', 'h2', 'h3', 'h4', 'h5', 'p', 'li', 'div']
        extracted_texts = set()

        for tag in tags_to_extract:
            elements = soup.find_all(tag)
            for element in elements:
                text = element.text.strip()
                if text:
                    extracted_texts.add(element.text)

        """
        # Print extracted content
        for text in extracted_texts:
            print(text)
        """
        return extracted_texts
    
# URL of the website to scrape
url = 'https://www.meridianexecutivelimo.com/'
extract_dynamic_content(url)
