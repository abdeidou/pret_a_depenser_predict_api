import pytest
import json
from main import customers_data, customers_data_ohe, lgbm
from main import app

# Fixtures for configuration

# Fixture providing a Flask test client
@pytest.fixture()
def client():
    with app.test_client() as client:
        return client

# Fixture providing an example customer_id for tests
@pytest.fixture()
def customer_id():
    return 100038

# Fixture providing expected JSON data for a customer_id
@pytest.fixture()
def expected_customer_data(customer_id):
    return customers_data[customers_data['SK_ID_CURR'] == str(customer_id)].to_json()

# Fixture providing the expected prediction result for a customer_id
@pytest.fixture()
def expected_customer_predict(customer_id):
    customer_row = customers_data[customers_data['SK_ID_CURR'] == str(customer_id)]
    customer_row_ohe = customers_data_ohe.iloc[customer_row.index].drop(columns=['SK_ID_CURR'], axis=1)
    return lgbm.predict_proba(customer_row_ohe).tolist()

# Test functions

# Test function for customer_data
def test_customer_data_api(client, customer_id, expected_customer_data):
    # Make a request to the API
    with client.get(f"/customer_data", query_string={"customer_id": customer_id}, allow_redirects=False) as response:
        # Follow the redirect if present
        if response.status_code == 308:
            location = response.headers["Location"]
            response = client.get(location)
        # Verify the response status
        assert response.status_code == 200
        # Verify the response
        response_data = json.loads(response.data)
        assert response_data['customer_data'] == expected_customer_data

# Test function for predict
def test_predict_api(client, customer_id, expected_customer_predict):
    # Make a request to the API
    with client.get(f"/predict", query_string={"customer_id": customer_id}, allow_redirects=False) as response:
        # Follow the redirect if present
        if response.status_code == 308:
            location = response.headers["Location"]
            response = client.get(location)
        # Verify the response status
        assert response.status_code == 200
        # Verify the response
        response_data = json.loads(response.data)
        assert response_data['customer_predict'] == expected_customer_predict