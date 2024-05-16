## Description

Predict house prices in California.

The target variable is the median house value for California districts, expressed in hundreds of thousands of dollars ($100,000)

This model was trained on the [California Housing dataset](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html). Prices are predicted based on these features: median income, housing median age, total rooms, total bedrooms, population, households, latitude, and longitude.

## Example Payload

Here is an example payload you can use to test model inference.

```json
{
    "inputs": [
        {
            "name": "input-0",
            "shape": [1],
            "datatype": "BYTES",
            "parameters": null,
            "data": [
                [1.6812, 25.0, 4.1922, 1.0223, 1392.0, 3.8774, 36.06, -119.01]
            ]
        }
    ]
}
```