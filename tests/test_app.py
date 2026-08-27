"""End-to-end smoke tests for the Streamlit credit-risk dashboard."""

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class CreditRiskAppTests(unittest.TestCase):
    def run_app(self):
        return AppTest.from_file(str(APP_PATH)).run(timeout=30)

    def test_dashboard_loads(self):
        app = self.run_app()

        self.assertEqual([], list(app.exception))
        self.assertEqual("RUN RISK ASSESSMENT", app.button[0].label)

    def test_assessment_accepts_no_account_values(self):
        app = self.run_app()
        app.selectbox[3].set_value("No Account")
        app.selectbox[4].set_value("No Account")
        app.button[0].click().run(timeout=30)

        self.assertEqual([], list(app.exception))
        self.assertEqual([], list(app.error))
        result_markup = "\n".join(markdown.value for markdown in app.markdown)
        self.assertTrue("LOW RISK" in result_markup or "HIGH RISK" in result_markup)


if __name__ == "__main__":
    unittest.main()
