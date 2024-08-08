## Description

Santander Product Recommendations using KNN model.

Generate product recommendations for a given user based on their historical product ownership.

This model uses a customer's ID to predict which banking products would be of interest to the customer. The recommendations are derived from a collaborative filtering approach using K-Nearest Neighbors (KNN). The model utilizes preprocessed data and precomputed KNN indices loaded from disk to generate recommendations.

## Example Payload

Here is an example payload you can use to test model inference.

```json
{
    "inputs": [
        {
            "name": "input-0",
            "shape": [1],
            "datatype": "INT32",
            "parameters": null,
            "data": [1361367]
        }
    ]
}

```