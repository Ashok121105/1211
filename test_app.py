import unittest
from app import app
import json

class FlaskAppTests(unittest.TestCase):

    def setUp(self):
        # Set up a test client
        self.app = app.test_client()
        self.app.testing = True

    def test_home_page(self):
        # Test if home page loads correctly
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'AI-Powered Credit Card Approval Prediction', response.data)

    def test_about_page(self):
        # Test if about page loads correctly
        response = self.app.get('/about')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Problem Statement', response.data)

    def test_predict_page(self):
        # Test if predict page loads correctly
        response = self.app.get('/predict_page')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Credit Card Eligibility Check', response.data)

    def test_api_predict_missing_data(self):
        # Test the API endpoint without sending JSON payload
        response = self.app.post('/api/predict')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b'No JSON payload', response.data)

if __name__ == '__main__':
    unittest.main()
