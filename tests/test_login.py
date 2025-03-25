import pytest
from flask_testing import TestCase
from app import app  # Import your Flask app

class TestTemplates(TestCase):
    def create_app(self):
        app.config['TESTING'] = True
        return app

    # ========================= #
    # Test Rendering of Pages   #
    # ========================= #
    
    def test_home_page_renders(self):
        response = self.client.get("/")
        self.assert200(response)
        self.assert_template_used("login.html")
        self.assertIn(b"Login", response.data)  # Ensure page contains 'Login'

    def test_index_page_renders(self):
        response = self.client.get("/index")
        self.assert200(response)
        self.assert_template_used("index.html")
        self.assertIn(b"Dashboard", response.data)  # Ensure 'Dashboard' is present

    def test_analyze_page_renders(self):
        response = self.client.post("/analyze", data={"prompt": "AI and NLP"})
        self.assert200(response)
        self.assert_template_used("result.html")
        self.assertIn(b"Results", response.data)  # Ensure 'Results' appears on the page

    # ========================= #
    # Test Form Submission      #
    # ========================= #

    def test_analyze_form_submission(self):
        response = self.client.post("/analyze", data={"prompt": "Machine Learning"})
        self.assert200(response)
        self.assert_template_used("result.html")
        self.assertIn(b"Results", response.data)

    # ========================= #
    # Test Error Handling       #
    # ========================= #

    def test_404_page(self):
        response = self.client.get("/nonexistentpage")
        self.assert404(response)
        self.assertIn(b"404 Not Found", response.data)

if __name__ == "__main__":
    pytest.main()
import pytest
from flask_testing import TestCase
from app import app  # Import your Flask app

class TestTemplates(TestCase):
    def create_app(self):
        app.config['TESTING'] = True
        return app

    # ========================= #
    # Test Rendering of Pages   #
    # ========================= #
    
    def test_home_page_renders(self):
        response = self.client.get("/")
        self.assert200(response)
        self.assert_template_used("login.html")
        self.assertIn(b"Login", response.data)  # Ensure page contains 'Login'

    def test_index_page_renders(self):
        response = self.client.get("/index")
        self.assert200(response)
        self.assert_template_used("index.html")
        self.assertIn(b"Dashboard", response.data)  # Ensure 'Dashboard' is present

    def test_analyze_page_renders(self):
        response = self.client.post("/analyze", data={"prompt": "AI and NLP"})
        self.assert200(response)
        self.assert_template_used("result.html")
        self.assertIn(b"Results", response.data)  # Ensure 'Results' appears on the page

    # ========================= #
    # Test Form Submission      #
    # ========================= #

    def test_analyze_form_submission(self):
        response = self.client.post("/analyze", data={"prompt": "Machine Learning"})
        self.assert200(response)
        self.assert_template_used("result.html")
        self.assertIn(b"Results", response.data)

    # ========================= #
    # Test Error Handling       #
    # ========================= #

    def test_404_page(self):
        response = self.client.get("/nonexistentpage")
        self.assert404(response)
        self.assertIn(b"404 Not Found", response.data)

if __name__ == "__main__":
    pytest.main()
