"""System tests: every page of the Streamlit demo renders without errors."""
from pathlib import Path

import pytest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
PAGES = ["Live risk check", "Geofence simulation", "Digital ID (blockchain)", "Results dashboard",
         "System validation & tests", "About"]


@pytest.mark.system
@pytest.mark.parametrize("page", PAGES)
def test_app_page_renders(artifacts, page):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.system
def test_app_anchors_alert_on_blockchain(artifacts):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value("Geofence simulation").run()
    sc = next(x for x in at.selectbox if x.label == "Scenario")
    sc.select(next(o for o in sc.options if "hazard_night" in str(o) or "Hazard Night" in str(o))).run()
    next(x for x in at.checkbox if x.label.startswith("Anchor")).check().run()
    assert not at.exception
    assert any("anchored as tx" in str(m.value) for m in at.markdown)
