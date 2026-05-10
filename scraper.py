"""
SHL Product Catalog Scraper
Scrapes ALL Individual Test Solutions from the SHL product catalog.
Outputs catalog.json with full assessment metadata.
"""
import json
import time
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def log(msg):
    print(msg, flush=True)


def fetch_page(url, timeout=30):
    """Fetch a page with retries."""
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            if attempt == 2:
                log(f"  FAILED after 3 attempts: {e}")
                return None
            time.sleep(1)


def find_individual_table(soup):
    """Find the Individual Test Solutions table in the page."""
    tables = soup.find_all("table")
    for table in tables:
        # Check header row - uses <th> tags
        header_row = table.find("tr")
        if not header_row:
            continue
        # Look in both th and td tags for the header
        header_cells = header_row.find_all(["th", "td"])
        for cell in header_cells:
            if "Individual Test Solutions" in cell.get_text(strip=True):
                return table
    return None


def scrape_listing_page(page_num: int) -> list[dict]:
    """Scrape one page of Individual Test Solutions."""
    start = page_num * 12
    url = f"{CATALOG_URL}?start={start}&sz=12&type=2&type=1"

    html = fetch_page(url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    table = find_individual_table(soup)

    if not table:
        log(f"  WARNING: Individual Test Solutions table not found on page {page_num + 1}")
        return []

    items = []
    rows = table.find_all("tr")[1:]  # Skip header row

    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        # Cell 0: Name + Link
        link_tag = cells[0].find("a")
        if not link_tag:
            continue
        name = link_tag.get_text(strip=True)
        href = link_tag.get("href", "")
        full_url = BASE_URL + href if href.startswith("/") else href

        # Cell 1: Remote Testing
        remote_testing = bool(cells[1].find("span", class_=lambda c: c and "-yes" in c))

        # Cell 2: Adaptive/IRT
        adaptive_irt = bool(cells[2].find("span", class_=lambda c: c and "-yes" in c))

        # Cell 3: Test Type codes
        type_text = cells[3].get_text(strip=True)
        test_types = [c for c in type_text if c in TEST_TYPE_MAP]

        items.append({
            "name": name,
            "url": full_url,
            "test_type": test_types,
            "remote_testing": remote_testing,
            "adaptive_irt": adaptive_irt,
        })

    return items


def scrape_product_page(item: dict) -> dict:
    """Scrape a product page for description and metadata."""
    html = fetch_page(item["url"])
    if not html:
        item["description"] = ""
        return item

    soup = BeautifulSoup(html, "html.parser")

    # Extract description
    description = ""
    for h4 in soup.find_all("h4"):
        if h4.get_text(strip=True) == "Description":
            sibling = h4.find_next_sibling()
            if sibling:
                description = sibling.get_text(strip=True)
            break

    # Fallback: OG description
    if not description:
        og = soup.find("meta", property="og:description")
        if og and og.get("content"):
            content = og["content"]
            if ":" in content:
                description = content.split(":", 1)[1].strip()
            else:
                description = content

    item["description"] = description

    # Extract job levels
    for h4 in soup.find_all("h4"):
        if h4.get_text(strip=True) == "Job levels":
            sib = h4.find_next_sibling()
            if sib:
                item["job_levels"] = sib.get_text(strip=True).rstrip(",")
            break

    # Extract languages
    for h4 in soup.find_all("h4"):
        if h4.get_text(strip=True) == "Languages":
            sib = h4.find_next_sibling()
            if sib:
                item["languages"] = sib.get_text(strip=True).rstrip(",")
            break

    # Extract assessment length
    for h4 in soup.find_all("h4"):
        if h4.get_text(strip=True) == "Assessment length":
            sib = h4.find_next_sibling()
            if sib:
                item["duration"] = sib.get_text(strip=True)
            break

    # Supplement remote testing from product page
    for p_tag in soup.find_all("p"):
        if "Remote Testing" in p_tag.get_text(strip=True):
            if p_tag.find("span", class_=lambda c: c and "-yes" in c):
                item["remote_testing"] = True
            break

    return item


def main():
    log("=" * 60)
    log("SHL Product Catalog Scraper")
    log("=" * 60)

    # Phase 1: Scrape all 32 listing pages
    log("\nPhase 1: Scraping listing pages...")
    all_items = {}
    total_pages = 32

    for page in range(total_pages):
        items = scrape_listing_page(page)
        new = 0
        for item in items:
            if item["url"] not in all_items:
                all_items[item["url"]] = item
                new += 1
        log(f"  Page {page+1}/{total_pages}: {len(items)} items, {new} new (total: {len(all_items)})")
        time.sleep(0.3)

    log(f"\nPhase 1 done: {len(all_items)} unique assessments")

    if len(all_items) == 0:
        log("ERROR: No assessments found! Check the scraper logic.")
        return

    # Phase 2: Scrape product pages for descriptions
    items_list = list(all_items.values())
    log(f"\nPhase 2: Scraping {len(items_list)} product pages for details...")

    completed = [0]
    total = len(items_list)

    def scrape_one(item):
        result = scrape_product_page(item)
        completed[0] += 1
        if completed[0] % 25 == 0 or completed[0] == total:
            log(f"  Progress: {completed[0]}/{total}")
        time.sleep(0.15)
        return result

    results = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(scrape_one, item) for item in items_list]
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                log(f"  ERROR: {e}")

    # Sort and fill defaults
    results.sort(key=lambda x: x["name"])
    for item in results:
        item.setdefault("description", "")
        item.setdefault("job_levels", "")
        item.setdefault("languages", "")
        item.setdefault("duration", "")

    # Save
    with open("catalog.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Stats
    log(f"\n{'='*60}")
    log(f"Total assessments: {len(results)}")
    log(f"With descriptions: {sum(1 for i in results if i.get('description'))}")
    log(f"Remote testing:    {sum(1 for i in results if i.get('remote_testing'))}")
    log(f"Adaptive/IRT:      {sum(1 for i in results if i.get('adaptive_irt'))}")

    type_counts = {}
    for item in results:
        for t in item.get("test_type", []):
            type_counts[t] = type_counts.get(t, 0) + 1
    log("Test types:")
    for t, count in sorted(type_counts.items()):
        log(f"  {t} ({TEST_TYPE_MAP.get(t, '?')}): {count}")

    log(f"\nSaved to catalog.json - Done!")


if __name__ == "__main__":
    main()
