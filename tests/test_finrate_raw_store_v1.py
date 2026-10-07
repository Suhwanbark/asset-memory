"""Regression checks for source fidelity, addressing and temporal identity."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "method"))
from finrate_raw_store_v1 import project_html
from build_finrate_raw_smoke10_v1 import validate_projection


class ProjectionTests(unittest.TestCase):
    def test_source_paths_survive_removal_of_same_tag_siblings(self):
        source = b'''<html><head><title>not body</title></head><body>
        <div style="display:none">hidden</div><div>Visible <b>2023</b> revenue</div>
        <script>bad()</script></body></html>'''
        result = project_html(source)
        self.assertNotIn("hidden", result["text"])
        self.assertNotIn("bad()", result["text"])
        self.assertIn("Visible 2023 revenue", result["text"])
        self.assertTrue(any("/div[2]/" in s["dom_path"] for s in result["spans"]))
        self.assertEqual(validate_projection(source, result), len(result["spans"]))

    def test_table_headers_symbols_and_span_metadata_survive(self):
        source = b'''<html><body><p>USD in millions</p><table>
        <tr><th rowspan="2">Revenue</th><th colspan="2">2023 / 2022</th></tr>
        <tr><td><span>(</span><span>123</span><span>)</span></td><td>456</td></tr>
        </table><p>Footnote A: discontinued operations.</p></body></html>'''
        result = project_html(source)
        self.assertIn("Revenue\t2023 / 2022\t", result["text"])
        self.assertIn("(123)\t456\t", result["text"])
        self.assertIn("Footnote A", result["text"])
        cells = result["structure"]["tables_and_cells"]
        self.assertTrue(any(c["rowspan"] == "2" for c in cells))
        self.assertTrue(any(c["colspan"] == "2" for c in cells))
        validate_projection(source, result)

    def test_item_navigation_points_to_body_not_toc(self):
        source = b'''<html><body><a href="#risk">Item 1A. Risk Factors</a>
        <a href="#business">Item 1. Business</a><div id="risk"><p>Actual risks.</p></div>
        <div id="business"><p>Actual business.</p></div></body></html>'''
        result = project_html(source)
        items = [s for s in result["sections"] if s["kind"] == "item"]
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["source_anchor"], "risk")
        self.assertEqual(result["text"][items[0]["start"]:items[0]["end"]], "Actual risks.\n")
        self.assertEqual(items[-1]["end"], len(result["text"]))


if __name__ == "__main__":
    unittest.main()
