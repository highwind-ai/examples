## Description

Classify different iris flowers based on their measurements.

This model was trained on the classic Iris classification dataset. 

Iris class is predicted based on these feaures:

- Sepal length in cm
- Sepal width in cm
- Petal length in cm
- Petal width in cm

Output Iris class:

1. Iris-Setosa
2. Iris-Versicolour
3. Iris-Virginica

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
                [6.1, 2.8, 4.7, 1.2]
            ]
        }
    ]
}
```
