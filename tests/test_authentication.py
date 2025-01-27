import requests
import jwt
from datetime import datetime, timedelta, timezone

# JWT configuration – must match your service's configuration
JWT_SECRET_KEY = "your_jwt_secret_key_here"
JWT_ALGORITHM = "HS256"

# Utility functions to create expired and valid tokens
def create_expired_token(user_id: str):
    expiration = datetime.now(timezone.utc) - timedelta(hours=1)  # expired time
    payload = {"sub": user_id, "exp": expiration}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def create_valid_token(user_id: str):
    expiration = datetime.now(timezone.utc) + timedelta(hours=1)  # valid for 1 hour
    payload = {"sub": user_id, "exp": expiration}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

# Data input for prediction endpoint
input_data = {
    "gre_score": 320,
    "toefl_score": 110,
    "university_rating": 4,
    "sop": 4.0,
    "lor": 4.0,
    "cgpa": 9.0,
    "research": 1
}

def test_authentication_missing_token(base_url):
    """Verify that authentication fails when JWT token is missing."""
    response = requests.post(f"{base_url}/v1/models/gbr/predict", json=input_data)
    assert response.status_code == 401
    data = response.json()
    assert data["detail"] == "Missing authentication token"

def test_authentication_invalid_token(base_url):
    """Verify that authentication fails with an invalid JWT token."""
    headers = {"Authorization": "Bearer invalidtoken"}
    response = requests.post(f"{base_url}/v1/models/gbr/predict", headers=headers, json=input_data)
    assert response.status_code == 401

def test_authentication_expired_token(base_url):
    """Verify that authentication fails with an expired JWT token."""
    expired_token = create_expired_token("max")
    headers = {"Authorization": f"Bearer {expired_token}"}
    response = requests.post(f"{base_url}/v1/models/gbr/predict", headers=headers, json=input_data)
    assert response.status_code == 401
    data = response.json()
    assert "expired" in data["detail"].lower()

def test_authentication_valid_token(base_url):
    """Verify that authentication succeeds with a valid JWT token."""
    valid_token = create_valid_token("max")
    headers = {"Authorization": f"Bearer {valid_token}"}
    response = requests.post(f"{base_url}/v1/models/gbr/predict", headers=headers, json=input_data)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
