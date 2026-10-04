"""Browser acceptance journeys 1-12 (prompt section 22.3), numbered jNN. Run with: make e2e"""

from __future__ import annotations

import itertools
import re

import pytest
from playwright.sync_api import expect

from tests.e2e.conftest import shot

pytestmark = pytest.mark.e2e
_n = itertools.count()


def register(page, base, prefix="e2e"):
    name = f"{prefix}{next(_n)}x{abs(hash(base)) % 10000}"
    page.goto(base + "/register")
    page.fill("#username", name)
    page.fill("#password", "journey password 1")
    page.click("#auth-submit")
    page.wait_for_url(re.compile(r".*/welcome$"))
    return name


def login(page, base, user, pw="anilogic-demo"):
    page.goto(base + "/login")
    page.fill("#username", user)
    page.fill("#password", pw)
    page.click("#auth-submit")
    page.wait_for_url(base + "/")


def test_j01_j11_new_user_discovery_then_not_interested(page, live_server):
    base = live_server
    register(page, base)
    page.wait_for_selector("#tag-chips .chip")
    for chip in page.locator("#tag-chips .chip").all()[:3]:
        chip.click()
    selects = page.locator("#title-list select").all()
    for s in selects[:3]:
        s.select_option("9")
    shot(page, "j01-welcome")
    page.click("#welcome-finish")
    page.wait_for_url(base + "/")
    page.wait_for_selector("#home-list .card")
    mode = page.inner_text("#home-mode")
    assert "Warming up" in mode or "Starter picks" in mode          # labelled initial recommendations
    first = page.locator("#home-list .card a.title").first.inner_text()
    shot(page, "j01-home-starter")
    page.locator("#home-list .card").first.locator("text=Not interested").click()
    page.wait_for_timeout(500)
    page.reload()
    page.wait_for_selector("#home-list .card")
    assert first not in page.locator("#home-list .card a.title").all_inner_texts()   # exclusion takes effect
    assert not page.errors


def test_j02_detail_top5_with_sources_and_reasons(page, live_server):
    base = live_server
    page.goto(base + "/discover")
    page.wait_for_selector("#results .card")
    page.fill("#year_from", "2090")                                   # start-year filter: nothing qualifies
    page.click("#search-form button[type=submit]")
    expect(page.locator("#results")).to_contain_text("No titles match")
    assert "year_from=2090" in page.url                               # filters are kept in the URL
    page.click("#clear-filters")
    page.wait_for_selector("#results .card")
    page.locator("#results .card a.title").first.click()
    page.wait_for_selector("#similar-list .card")
    cards = page.locator("#similar-list .card")
    assert 1 <= cards.count() <= 5
    assert cards.first.locator(".reasons li").count() >= 1
    page.click("text=Data sources")
    shot(page, "j02-detail-similar")
    cards.first.locator("a.title").click()
    page.wait_for_selector("#detail-root h1")
    assert not page.errors


def test_j03_j04_two_collections_strict_filters_and_relaxation(page, live_server):
    base = live_server
    register(page, base)
    page.goto(base + "/collections")
    for name in ("Mysteries", "Comedies"):
        page.fill("#c-name", name)
        page.click("#create-form button[type=submit]")
        page.wait_for_selector(f"text={name}")
    ids = [a.get_attribute("href").split("/")[-1] for a in page.locator("#collection-list a").all()]
    for cid, genre in zip(ids, ("Mystery", "Comedy"), strict=True):
        keys = page.evaluate(f"""async () => (await (await fetch('/api/v1/anime?genres={genre}&page_size=4')).json())
                                 .items.map(i => i.item_key)""")
        for k in keys:
            page.evaluate("""async ([cid, k]) => fetch(`/api/v1/collections/${cid}/items/${encodeURIComponent(k)}`,
                               {method: 'PUT', headers: {'X-CSRF-Token': document.cookie.match(/anilogic_csrf=([^;]+)/)[1]}})""",
                          [cid, k])
    results = []
    for cid in ids:
        page.goto(f"{base}/collections/{cid}")
        page.wait_for_selector("#rec-list .card")
        results.append(page.locator("#rec-list .card a.title").all_inner_texts())
    assert results[0] != results[1]                                   # different collection -> different profile
    page.fill("#max_episodes", "30")
    page.click("#filter-form button[type=submit]")
    page.wait_for_selector("#rec-list .card")
    hrefs = [a.get_attribute("href") for a in page.locator("#rec-list .card a.title").all()]
    assert hrefs
    for href in hrefs:                                                # strict success: every result satisfies the rule
        eps = page.evaluate("async (h) => (await (await fetch('/api/v1' + h)).json()).episodes", href)
        assert eps is not None and eps <= 30
    page.fill("#max_episodes", "1")
    page.fill("#min_score", "9.9")
    page.click("#filter-form button[type=submit]")
    page.wait_for_selector("#rec-status .notice")
    assert page.locator("#rec-list .card").count() == 0              # strict result stays empty
    shot(page, "j04-collection-no-match")
    btn = page.locator("#rec-status button").first
    btn.click()
    page.wait_for_selector("#rec-status .notice.warn >> text=relaxed")
    page.wait_for_selector("#rec-status >> text=With your original filters there were 0 matching titles")
    shot(page, "j04-collection-relaxed")
    assert not page.errors


def test_j05_rating_cold_to_warm_and_undo(page, live_server):
    base = live_server
    register(page, base)
    page.goto(base + "/")
    page.wait_for_selector("#home-mode:not(:empty)")
    assert "Starter picks" in page.inner_text("#home-mode")
    keys = page.evaluate("""async () => (await (await fetch('/api/v1/anime?sort=popularity&page_size=9')).json())
                           .items.map(i => i.item_key)""")
    for k in keys[:8]:
        page.evaluate("""async (k) => fetch(`/api/v1/anime/${encodeURIComponent(k)}/rating`, {method: 'PUT',
            headers: {'Content-Type': 'application/json', 'X-CSRF-Token': document.cookie.match(/anilogic_csrf=([^;]+)/)[1]},
            body: JSON.stringify({rating: 9})})""", k)
    page.reload()
    page.wait_for_selector("#home-list .card")
    assert "Personalised" in page.inner_text("#home-mode")
    shot(page, "j05-home-personalised")
    page.goto(base + "/anime/" + keys[0])
    page.wait_for_selector("text=Clear rating")
    page.click("text=Clear rating")
    page.wait_for_timeout(400)
    page.goto(base + "/")
    page.wait_for_selector("#home-mode:not(:empty)")
    assert "Warming up" in page.inner_text("#home-mode")             # back below the warm gate
    assert not page.errors


def test_j06_ga_vs_greedy_on_the_same_pool(page, live_server):
    base = live_server
    login(page, base, "demo_admin")
    page.goto(base + "/dev")
    page.click("#home-run")
    page.wait_for_selector("#home-out details")
    text = page.inner_text("#home-out")
    assert "icf+ga" in text and "icf+greedy" in text
    shot(page, "j06-dev-ga-vs-greedy")
    assert not page.errors


def test_j07_j08_j10_chat_constraints_refinement_local_mode(page, live_server):
    base = live_server
    register(page, base)
    page.goto(base + "/chat")
    page.fill("#message", "a comedy with at most 12 episodes and no horror")
    page.click("#send")
    page.wait_for_selector(".msg.assistant")
    assert page.locator(".chip.hard >> text=<= 12 episodes").count() == 1
    assert page.locator("text=Local mode").count() >= 1              # LLM not configured -> labelled local mode
    first_answer = page.locator(".msg.assistant").first
    assert first_answer.locator(".result-row").count() >= 1          # local results ...
    assert first_answer.locator("details.evidence li").count() >= 1  # ... backed by retrieved evidence
    page.fill("#message", "at most 6 episodes")
    page.click("#send")
    page.locator(".msg.assistant").nth(1).wait_for()
    assert page.locator(".msg.assistant").last.locator(".chip.hard >> text=<= 6 episodes").count() == 1
    shot(page, "j07-chat-refinement")
    title = page.evaluate("""async () => (await (await fetch('/api/v1/anime?types=Movie&page_size=1')).json()).items[0].title""")
    page.fill("#message", f"which episode of {title} has the rooftop confession?")
    page.click("#send")
    page.locator(".msg.assistant").nth(2).wait_for()
    answer = page.locator(".msg.assistant").last.locator(".bubble").inner_text()
    assert not re.search(r"episode \d+", answer, re.I)
    assert not page.errors


def test_j09_fifteen_hour_plan_edit_and_export(page, live_server):
    base = live_server
    register(page, base)
    page.goto(base + "/plan")
    page.fill("#hours", "15")
    page.fill("#max_titles", "3")
    page.select_option("#diversity", "high")
    page.click("#preview")
    page.wait_for_selector("#plan-result table")
    summary = page.inner_text("#plan-result")
    used = int(re.search(r"(\d+) of 900 min planned", summary).group(1))
    assert used <= 900 and "Fits the budget" in summary
    page.click("#save")
    page.wait_for_url(re.compile(r".*/plans/\d+$"))
    page.wait_for_selector("#plan-result table")
    before = page.locator("#plan-result tbody tr").count()
    page.locator("#plan-result tbody tr").first.locator("button.danger").click()
    expect(page.locator("#plan-result tbody tr")).to_have_count(before - 1)
    assert "Fits the budget" in page.inner_text("#plan-result")      # re-validated after the edit
    shot(page, "j09-plan-edited")
    with page.expect_download() as dl:
        page.click("a[href$='format=markdown']")
    text = open(dl.value.path(), encoding="utf-8").read()
    shown = int(re.search(r"(\d+) of 900 min planned", page.inner_text("#plan-result")).group(1))
    exported = float(re.search(r"planned: ([\d.]+) min", text).group(1))
    assert "Budget: 900" in text and abs(exported - shown) <= 1       # export matches the saved revision
    assert text.count("\n| ") - 1 == page.locator("#plan-result tbody tr").count()
    assert not page.errors


def test_j12_second_account_is_isolated(page, live_server, browser_ctx):
    base = live_server
    register(page, base)
    page.goto(base + "/collections")
    page.fill("#c-name", "Secret list")
    page.click("#create-form button[type=submit]")
    page.wait_for_selector("text=Secret list")
    href = page.locator("#collection-list a").first.get_attribute("href")
    other = browser_ctx.new_context().new_page()
    register(other, base, "other")
    other.goto(base + href)
    other.wait_for_selector("#item-list .state.error, #item-list .state")
    assert "Secret list" not in other.content()
    r = other.evaluate(f"async () => (await fetch('/api/v1{href}')).status")
    assert r == 404
