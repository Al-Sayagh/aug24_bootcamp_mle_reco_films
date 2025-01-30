import requests
import jwt

# JWT configuration – must match your service's configuration
JWT_SECRET_KEY = "your_jwt_secret_key_here"
JWT_ALGORITHM = "HS256"

def test_login_success(auth_token):
    """Test that a valid JWT token is returned for correct credentials."""
    # The auth_token fixture has already attempted to log in with correct credentials.
    # We just verify that a token was returned and that it's valid.
    assert auth_token is not None
    
    # Decode the token to verify its payload
    decoded = jwt.decode(auth_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
    assert decoded["sub"] == "max"

def test_login_failure(base_url):
    """Test login API with incorrect credentials should return 401 error."""
    credentials = {"username": "max", "password": "wrongpassword"}
    response = requests.post(f"{base_url}/login", json=credentials)
    
    # Expecting a 401 Unauthorized request
    assert response.status_code == 401
    
    data = response.json()
    # Check that the response detail matches the expected message
    assert "detail" in data
    assert data["detail"] == "Invalid credentials"
