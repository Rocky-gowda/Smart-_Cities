"""Run the full automated test suite and write a test report.

Run:  python run_tests.py
Outputs: results/test_report.html (open in a browser), results/test_summary.json,
         results/test_results.csv
"""
import json
import sys

import pytest

from src import config as C

CATS = ["unit", "integration", "system", "e2e", "performance"]


class Collector:
    def __init__(self):
        self.rows = []

    def pytest_runtest_logreport(self, report):
        if report.when == "call" or (report.when == "setup" and report.outcome != "passed"):
            item_markers = getattr(report, "_cats", None)
            self.rows.append(dict(test=report.nodeid, outcome=report.outcome,
                                  duration_s=round(report.duration, 4), category=item_markers))

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        out = yield
        rep = out.get_result()
        rep._cats = next((m for m in CATS if item.get_closest_marker(m)), "other")


def main():
    col = Collector()
    code = pytest.main(["tests", "-v", f"--html={C.RESULT_DIR / 'test_report.html'}", "--self-contained-html",
                        "-p", "no:cacheprovider"], plugins=[col])
    rows = col.rows
    by = []
    for c in CATS:
        r = [x for x in rows if x["category"] == c]
        by.append(dict(category=c, total=len(r), passed=sum(x["outcome"] == "passed" for x in r),
                       failed=sum(x["outcome"] == "failed" for x in r),
                       skipped=sum(x["outcome"] == "skipped" for x in r)))
    summary = dict(total=len(rows), passed=sum(x["outcome"] == "passed" for x in rows),
                   failed=sum(x["outcome"] == "failed" for x in rows), by_category=by)
    json.dump(summary, open(C.RESULT_DIR / "test_summary.json", "w"), indent=2)
    import csv
    with open(C.RESULT_DIR / "test_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "test", "outcome", "duration_s"])
        w.writeheader()
        for x in rows:
            w.writerow({k: x[k] for k in w.fieldnames})
    print(json.dumps(summary, indent=2))
    sys.exit(code)


if __name__ == "__main__":
    main()
