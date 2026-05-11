"""
Scrapes Google Maps for hospitals and healthcare centers by search term + location.
Extracts facility names, coordinates, specialty tags, and care types.
Exports the results to a clean CSV file.

Requirements:
    pip install playwright pandas
    playwright install chromium

Usage:
    python healthcare_scraper.py
"""

import re
import os
import time
import random
import asyncio
import logging
import pandas as pd
from playwright.async_api import async_playwright

# ─────────────────────────────────────────────
# CONFIG — edit these before running
# ─────────────────────────────────────────────

SEARCHES = [
    # ── AVEIRO ──────────────────────────────────────────
    ("Hospital Geral", "Aveiro Portugal", "Hospital"),
    ("Hospital Privado", "Aveiro Portugal", "Hospital"),
    ("Centro de Saúde", "Aveiro Portugal", "Health Center"),
    ("Clínica Médica Especializada", "Aveiro Portugal", "Clinic"),
    ("Maternidade", "Aveiro Portugal", "Maternity"),

    # ── PORTO ───────────────────────────────────────────
    ("Hospital Universitário", "Porto Portugal", "Hospital"),
    ("Hospital Privado", "Porto Portugal", "Hospital"),
    ("Centro de Saúde", "Porto Portugal", "Health Center"),
    ("Clínica Médica", "Porto Portugal", "Clinic"),
    ("Instituto de Oncologia", "Porto Portugal", "Specialized Care"),
    
    # ── LISBOA ──────────────────────────────────────────
    ("Hospital Público", "Lisboa Portugal", "Hospital"),
    ("Clínica Privada", "Lisboa Portugal", "Clinic"),
    ("Centro de Saúde", "Lisboa Portugal", "Health Center"),
]

MAX_RESULTS_PER_SEARCH = 120      # max api cap
OUTPUT_FILE = "../data/hospitals.csv"
# ─────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

def extract_coordinates(url: str) -> str:
    """Extracts Latitude and Longitude from a Google Maps URL."""
    match = re.search(r"@(-?\d+\.\d+),(-?\d+\.\d+)", url)
    if match:
        return f"{match.group(1)}, {match.group(2)}"
    return ""

async def scrape_google_maps(search_term: str, location: str, care_type: str, max_results: int) -> list[dict]:
    """Scrape Google Maps for healthcare centers matching search_term near location."""
    results = []
    query = f"{search_term} {location}"
    url = f"https://www.google.com/maps/search/{query.replace(' ', '+')}"

    log.info(f"Searching Google Maps: '{query}'")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            locale="pt-PT"
        )
        page = await context.new_page()

        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(2000)

            # Handle Cookie Consent
            try:
                await page.click('button:has-text("Aceitar tudo")', timeout=3000)
                await page.wait_for_timeout(1000)
            except:
                try:
                    await page.click('button:has-text("Accept all")', timeout=3000)
                    await page.wait_for_timeout(1000)
                except:
                    pass

            # Scroll to load listings
            scrollable = page.locator('div[role="feed"]')
            for _ in range(5):
                await scrollable.evaluate("el => el.scrollBy(0, 1000)")
                await page.wait_for_timeout(1500)

            listings = await page.query_selector_all('a[href*="/maps/place/"]')
            seen_hrefs = set()
            unique_listings = []
            
            for l in listings:
                href = await l.get_attribute("href")
                if href and href not in seen_hrefs:
                    seen_hrefs.add(href)
                    unique_listings.append(l)

            log.info(f"   Found {len(unique_listings)} listings, processing up to {max_results}")

            for i, listing in enumerate(unique_listings[:max_results]):
                try:
                    await listing.click()
                    await page.wait_for_timeout(2500)

                    data = {
                        "Hospital Name": "",
                        "Coordinates": extract_coordinates(page.url),
                        "Specialty Tags": "",
                        "Care Type": care_type,
                    }

                    # Name
                    try:
                        data["Hospital Name"] = await page.inner_text('h1.DUwDvf', timeout=2000)
                    except:
                        pass

                    # Specialty Tags (Using the Maps Category)
                    try:
                        data["Specialty Tags"] = await page.inner_text('button[jsaction*="category"]', timeout=2000)
                    except:
                        try:
                            cat_el = await page.query_selector('span.DkEaL, .YhemCb span')
                            if cat_el:
                                data["Specialty Tags"] = await cat_el.inner_text()
                        except:
                            pass

                    # Clean up specific map markers acting as tags
                    if data["Specialty Tags"]:
                        data["Specialty Tags"] = data["Specialty Tags"].replace("·", "").strip()

                    if data["Hospital Name"]:
                        results.append(data)
                        log.info(f"   [{i+1}/{max_results}] ✓ {data['Hospital Name']} ({data['Specialty Tags']})")

                    await page.wait_for_timeout(random.uniform(800, 1500))

                except Exception as e:
                    log.debug(f"Error processing listing {i}: {e}")

        except Exception as e:
            log.error(f"Error scraping '{query}': {e}")
        finally:
            await browser.close()

    return results

def save_to_csv(facilities: list[dict], filename: str):
    """Save results to a formatted CSV file."""
    if not facilities:
        log.warning("No data to save.")
        return

    # Ensure the target directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    df = pd.DataFrame(facilities)

    # Reorder columns explicitly to match request
    cols = ["Hospital Name", "Coordinates", "Specialty Tags", "Care Type"]
    df = df[cols]

    # Remove duplicates by name to keep the dataset clean
    df = df.drop_duplicates(subset=["Hospital Name"], keep="first")

    # Save as CSV
    df.to_csv(filename, index=False, encoding='utf-8')

    log.info(f"\n✅ Saved {len(df)} healthcare facilities to '{filename}'")

async def main():
    all_facilities = []

    for search_term, location, care_type in SEARCHES:
        facilities = await scrape_google_maps(search_term, location, care_type, MAX_RESULTS_PER_SEARCH)
        all_facilities.extend(facilities)
        time.sleep(random.uniform(2, 4))  # pause between searches

    # Deduplicate before final save
    seen = set()
    unique = []
    for f in all_facilities:
        if f["Hospital Name"] and f["Hospital Name"] not in seen:
            seen.add(f["Hospital Name"])
            unique.append(f)

    log.info(f"\n Total unique facilities found: {len(unique)}")

    # Save
    save_to_csv(unique, OUTPUT_FILE)
    print(f"\n Finished! Check your clean dataset at '{OUTPUT_FILE}'.")

if __name__ == "__main__":
    asyncio.run(main())
