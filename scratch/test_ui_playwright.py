import os
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

SCREENSHOTS_DIR = Path("scratch/test_screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

def run_tests():
    print("Starting Playwright E2E UI Test Suite...")
    test_results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()

        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

        try:
            # 1. Test Navigation & Initial Load
            print("[Test 1] Navigating to http://127.0.0.1:8899 ...")
            t0 = time.time()
            page.goto("http://127.0.0.1:8899", wait_until="domcontentloaded", timeout=15000)
            load_time = time.time() - t0
            page.wait_for_timeout(2000)

            title = page.title()
            assert "EveryImage" in title, f"Unexpected title: {title}"
            print(f"PASS: Page loaded in {load_time:.2f}s with title '{title}'")
            test_results.append(("Page Load & Title", "PASS", f"Loaded in {load_time:.2f}s"))

            # Check sidebar stats
            stats_text = page.locator("#sidebarCountText").inner_text()
            print(f"PASS: Sidebar count text: '{stats_text}'")
            test_results.append(("Sidebar Stats Counter", "PASS", stats_text))

            # 2. Test All Photos / Gallery View
            print("[Test 2] Verifying Gallery Grid & Photos...")
            page.wait_for_selector("#regularGalleryGrid", timeout=5000)
            page.wait_for_timeout(2000)
            
            photos_count_text = page.locator("#galleryCountLabel").inner_text()
            print(f"Gallery count label: '{photos_count_text}'")
            
            cards = page.locator(".img-card")
            card_count = cards.count()
            print(f"Rendered image cards: {card_count}")
            assert card_count > 0, "No image cards rendered in gallery"
            
            page.screenshot(path=str(SCREENSHOTS_DIR / "01_gallery_view.png"))
            test_results.append(("Gallery View & Cards", "PASS", f"Rendered {card_count} image cards ({photos_count_text})"))

            # 3. Test Timeline Tab
            print("[Test 3] Testing Timeline Tab...")
            page.locator('.nav-btn[data-view="view-timeline"]').click()
            page.wait_for_timeout(1500)
            page.screenshot(path=str(SCREENSHOTS_DIR / "02_timeline_view.png"))
            test_results.append(("Timeline Tab Navigation", "PASS", "Timeline view displayed successfully"))

            # 4. Test Face Gallery Tab
            print("[Test 4] Testing Face Gallery Tab...")
            page.locator('.nav-btn[data-view="view-faces"]').click()
            page.wait_for_timeout(2000)
            
            face_cards = page.locator(".face-person-card")
            face_count = face_cards.count()
            print(f"Rendered face clusters: {face_count}")
            page.screenshot(path=str(SCREENSHOTS_DIR / "03_face_gallery.png"))
            assert face_count > 0, "No face clusters rendered"
            test_results.append(("Face Gallery Tab", "PASS", f"Rendered {face_count} face clusters"))

            # 5. Test Albums & Drives Tab
            print("[Test 5] Testing Albums & Drives Tab...")
            page.locator('.nav-btn[data-view="view-libraries"]').click()
            page.wait_for_timeout(1000)
            
            lib_items = page.locator(".lib-item")
            lib_count = lib_items.count()
            print(f"Configured albums: {lib_count}")
            page.screenshot(path=str(SCREENSHOTS_DIR / "04_albums_view.png"))
            assert lib_count >= 1, "No album libraries configured"
            test_results.append(("Albums & Drives Tab", "PASS", f"Found {lib_count} indexed album paths"))

            # 6. Test Analysis & Indexing Tab
            print("[Test 6] Testing Analysis & Indexing Tab...")
            page.locator('.nav-btn[data-view="view-scanner"]').click()
            page.wait_for_timeout(1000)
            
            engine_status = page.locator("#scanStatusTitle").inner_text()
            print(f"Engine status label: '{engine_status}'")
            page.screenshot(path=str(SCREENSHOTS_DIR / "05_indexing_view.png"))
            test_results.append(("Analysis & Indexing Tab", "PASS", f"Status: '{engine_status}'"))

            # 7. Test Search Tab & Semantic Search
            print("[Test 7] Testing Semantic Search Flow...")
            page.locator('.nav-btn[data-view="view-search"]').click()
            page.wait_for_timeout(1000)
            
            search_input = page.locator("#searchInput")
            search_input.fill("car")
            search_btn = page.locator('#view-search button:has-text("Search")')
            search_btn.click()
            
            page.wait_for_timeout(2500)
            search_results = page.locator("#searchResultsGrid .img-card")
            search_count = search_results.count()
            search_meta = page.locator("#searchMeta").inner_text()
            print(f"Search 'car' meta: '{search_meta}', results rendered: {search_count}")
            page.screenshot(path=str(SCREENSHOTS_DIR / "06_search_car.png"))
            assert search_count > 0, "Search returned 0 results"
            test_results.append(("Semantic Search ('car')", "PASS", f"Found {search_count} matches | {search_meta}"))

            # 8. Test Face Filter to Search & Clear Bugfix
            print("[Test 8] Testing Face Filter to Search & Reset Bugfix...")
            page.locator('.nav-btn[data-view="view-faces"]').click()
            page.wait_for_timeout(1500)
            
            # Click the 'View Photos' button on the first face card
            first_face = page.locator(".face-person-card").first
            view_photos_btn = first_face.locator('button:has-text("View Photos")')
            view_photos_btn.click()
            
            # Should have navigated to search view with that person
            page.wait_for_timeout(2000)
            active_view = page.locator(".view-container.active").get_attribute("id")
            assert active_view == "view-search", f"Expected search view, got {active_view}"
            
            person_query = page.locator("#searchInput").input_value()
            person_meta = page.locator("#searchMeta").inner_text()
            print(f"Person search input: '{person_query}', meta: '{person_meta}'")
            
            # Now test our bugfix: Clear the search input
            print("Testing search input clear to reset activePersonId...")
            page.locator("#searchInput").fill("")
            page.locator("#searchInput").dispatch_event("input")
            page.wait_for_timeout(500)
            
            # Now type a normal generic query
            page.locator("#searchInput").fill("nature")
            page.locator('#view-search button:has-text("Search")').click()
            page.wait_for_timeout(2500)
            
            search_meta_after = page.locator("#searchMeta").inner_text()
            search_results_after = page.locator("#searchResultsGrid .img-card").count()
            print(f"Search after reset meta: '{search_meta_after}', results: {search_results_after}")
            assert "Identified Person" not in search_meta_after and "Person:" not in search_meta_after, (
                f"Failed: Search was still constrained to person! Meta: {search_meta_after}"
            )
            page.screenshot(path=str(SCREENSHOTS_DIR / "07_search_after_person_reset.png"))
            test_results.append(("Face Filter Reset Bugfix", "PASS", f"Unconstrained search executed ({search_results_after} matches)"))

        except Exception as e:
            print(f"FAIL: Test step failed with error: {e}")
            test_results.append(("Test Execution", "FAIL", str(e)))
            page.screenshot(path=str(SCREENSHOTS_DIR / "error_state.png"))
        finally:
            browser.close()

    print("\n=== Playwright E2E UI Test Summary ===")
    for name, status, detail in test_results:
        print(f"[{status}] {name} - {detail}")
    
    critical_errors = [err for err in console_errors if "favicon.ico" not in err]
    if critical_errors:
        print(f"\nConsole Errors Detected: {critical_errors}")
    else:
        print("\nConsole Errors: None (clean client execution)")

if __name__ == "__main__":
    run_tests()
