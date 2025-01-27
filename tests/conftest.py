import pytest
import requests

@pytest.fixture(scope="module")
def base_url():
    return "http://localhost:3000"

@pytest.fixture(scope="module")
def auth_token(base_url):
    """Fixture to obtain an authentication token"""
    credentials = {
        "username": "max",
        "password": "lamenace"
    }
    response = requests.post(f"{base_url}/login", json=credentials)
    assert response.status_code == 200
    return response.json().get("token")

@pytest.fixture(scope="module")
def headers(auth_token):
    """Fixture for headers with authentication"""
    return {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }