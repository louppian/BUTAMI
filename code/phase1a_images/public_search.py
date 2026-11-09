"""AutoCrawler, icrawler, yt-dlp 원본 구현을 호출하는 검색 연결부."""
import json
import sys
from pathlib import Path
from urllib.parse import urlencode

import requests


def edge_driver(no_gui=False, proxy=None, user_data_dir=None):
    from selenium import webdriver
    options = webdriver.EdgeOptions()
    if no_gui:
        options.add_argument("--headless=new")
    if user_data_dir:
        options.add_argument(f"--user-data-dir={user_data_dir.resolve()}")
    if proxy:
        options.add_argument(f"--proxy-server={proxy}")
    return webdriver.Edge(options=options)


def search(site, query, limit=100):
    if site in ("google", "naver"):
        from autocrawler.config import CrawlConfig
        from autocrawler.collectors.google import collect_google_full
        from autocrawler.collectors.naver import collect_naver_full
        import autocrawler.collectors.google as google
        import autocrawler.collectors.naver as naver
        from autocrawler.collectors.base import is_captcha_page
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        def check_captcha(driver, no_gui):
            if is_captcha_page(driver):
                raise RuntimeError(f"{site}: CAPTCHA 확인 필요")
        google.build_driver = naver.build_driver = edge_driver
        google.wait_for_manual_captcha = naver.wait_for_manual_captcha = check_captcha
        def click_result(driver, xpath, timeout=15):
            check_captcha(driver, True)
            element = WebDriverWait(driver, timeout).until(EC.element_to_be_clickable((By.XPATH, xpath)))
            element.click()
            return element
        google.wait_and_click = naver.wait_and_click = click_result
        config = CrawlConfig(no_gui=True, face=True, full_resolution=True, limit=limit,
                             chrome_profile_dir=Path("results/phase1a_images/browser") / site)
        links = (collect_google_full if site == "google" else collect_naver_full)(query, config)
        source = ("https://www.google.com/search?" + urlencode({"q": query, "udm": 2})
                  if site == "google" else "https://search.naver.com/search.naver?" + urlencode({"query": query, "where": "image"}))
        return [{"image": url, "url": source, "title": ""} for url in links]
    if site == "bing":
        from icrawler.builtin import BingImageCrawler
        crawler = BingImageCrawler(storage={"root_dir": "results/phase1a_images/bing"})
        results = {}
        with requests.Session() as session:
            for offset in range(0, limit, 20):
                items = []
                for _ in range(3):
                    response = session.get("https://www.bing.com/images/async", params={
                        "q": query, "first": offset, "qft": "+filterui:face-face+filterui:photo-photo"}, timeout=30)
                    response.raise_for_status()
                    items = list(crawler.parser.parse(response))
                    if items:
                        break
                for item in items:
                    url = item["file_url"]
                    results[url] = {"image": url, "url": response.url, "title": ""}
        return list(results.values())[:limit]
    if site == "youtube":
        from yt_dlp import YoutubeDL
        with YoutubeDL({"quiet": True, "extract_flat": True, "skip_download": True, "socket_timeout": 30}) as ydl:
            entries = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)["entries"]
        return [{"image": f'https://i.ytimg.com/vi/{item["id"]}/maxresdefault.jpg',
                 "fallback_image": f'https://i.ytimg.com/vi/{item["id"]}/hqdefault.jpg',
                 "url": f'https://www.youtube.com/watch?v={item["id"]}', "title": item.get("title", "")}
                for item in entries if item and item.get("id")]
    raise ValueError(site)


if __name__ == "__main__":
    request = json.load(sys.stdin)
    results = search(**request)
    print(json.dumps(results, ensure_ascii=True))
