"""Headless smoke test for the interactive 12x8 cell map component.

Renders the generated HTML in Chromium, hovers aggregate cells and switches
players in the per-athlete O→D route panel. Run manually:

    python3 scripts/check_cell_map_interactive.py
"""

from __future__ import annotations

import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import xp_engine as xe  # noqa: E402
import xp_maps_interactive as xmi  # noqa: E402
import xp_study_engine as xpe  # noqa: E402

TOP_N = 250


def _pool():
    season = xe.load_european_league_season_passes()
    completed = season[season["is_won"] & season["has_end"]]
    counts = completed.groupby("player_id").size().sort_values(ascending=False)
    return completed[completed["player_id"].isin(counts.head(TOP_N).index)]


def main() -> int:
    from playwright.sync_api import sync_playwright

    analysis = xpe.build_player_cell_heatmap_bundle(_pool())
    html = xmi.build_cell_map_html(analysis)
    path = pathlib.Path(tempfile.mkdtemp()) / "cell_map.html"
    path.write_text(html, encoding="utf-8")

    cols = int(analysis["cols"])
    rows = int(analysis["rows"])
    failures: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 1100})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.goto(path.as_uri())
        page.wait_for_selector("#qmap-agg-plot", timeout=30_000)
        page.wait_for_selector("#qmap-pl-player-select", timeout=30_000)
        page.wait_for_timeout(1500)

        default_player = (analysis.get("players") or [{}])[0]
        default_name = str(default_player.get("name") or "")
        player_panel = page.locator("#qmap-pl-panel .qp-title")
        if default_name and default_name not in player_panel.inner_text():
            failures.append(
                f"player panel title unexpected: {player_panel.inner_text()!r}; "
                f"expected player {default_name!r}"
            )

        agg_panel = page.locator("#qmap-agg-panel .qp-title")
        if "Todos os atletas" not in agg_panel.inner_text():
            failures.append(f"aggregate panel title unexpected: {agg_panel.inner_text()!r}")

        field_x = float(analysis["field_x"])
        field_y = float(analysis["field_y"])

        def cell_point(plot_id: str, col: int, row: int) -> tuple[float, float]:
            pos = page.evaluate(
                """(args) => {
                    const gd = document.getElementById(args.plotId);
                    const fl = gd._fullLayout;
                    const bb = gd.getBoundingClientRect();
                    return {
                        x: bb.left + fl.xaxis._offset + fl.xaxis.l2p(args.point[0]),
                        y: bb.top + fl.yaxis._offset + fl.yaxis.l2p(args.point[1]),
                    };
                }""",
                {
                    "plotId": plot_id,
                    "point": [(col + 0.5) * field_x / cols, (row + 0.5) * field_y / rows],
                },
            )
            return pos["x"], pos["y"]

        def hover_agg_cell(col: int, row: int) -> tuple[float, float]:
            x, y = cell_point("qmap-agg-plot", col, row)
            page.mouse.move(x, y, steps=4)
            page.mouse.move(x + 1, y + 1)
            page.wait_for_timeout(400)
            return x, y

        for col, row in ((5, 3), (9, 1), (2, 6)):
            hover_agg_cell(col, row)
            title = agg_panel.inner_text()
            expected = f"C{col + 1}/L{row + 1}"
            if expected not in title:
                failures.append(f"agg hover ({col},{row}) -> panel {title!r}, expected {expected}")

        x, y = hover_agg_cell(5, 3)
        page.mouse.click(x, y)
        page.wait_for_timeout(400)
        if not page.locator("#qmap-agg-panel .qp-pin").count():
            failures.append("click did not pin the aggregate cell")
        pinned_title = agg_panel.inner_text()
        hover_agg_cell(9, 6)
        if pinned_title not in agg_panel.inner_text():
            failures.append("pinned aggregate cell changed on hover")

        page.click("#qmap-agg-reset")
        page.wait_for_timeout(400)
        if "Todos os atletas" not in agg_panel.inner_text():
            failures.append("reset did not restore the aggregate view")

        page.click('#qmap-agg-wrap .qmap-btn[data-metric="volume"]')
        page.wait_for_timeout(400)
        page.click('#qmap-agg-wrap .qmap-btn[data-scale="relative"]')
        page.wait_for_timeout(400)

        panel_text = page.locator("#qmap-pl-panel").inner_text().lower()
        if "mais comuns" not in panel_text:
            failures.append("player panel missing common O→D routes section")
        if "maior xp" not in panel_text:
            failures.append("player panel missing high-xP O→D routes section")

        players = analysis.get("players") or []
        if len(players) > 1:
            second = players[1]
            page.select_option("#qmap-pl-player-select", str(second["id"]))
            page.wait_for_timeout(600)
            if str(second.get("name") or "") not in player_panel.inner_text():
                failures.append("player select did not switch the route panel")

        if errors:
            failures.append("JS errors: " + " | ".join(errors[:5]))
        browser.close()

    if failures:
        print("FAIL")
        for line in failures:
            print(" -", line)
        return 1
    print("OK — aggregate hover, pin, reset, toolbar and player routes behave")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
