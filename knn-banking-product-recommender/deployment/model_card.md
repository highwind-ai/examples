## Description

Santander Product Recommendations using KNN model.

Generate product recommendations for a given user based on their historical product ownership.

This model uses a customer's ID to predict which banking products would be of interest to the customer. The recommendations are derived from a collaborative filtering approach using K-Nearest Neighbors (KNN). The model utilizes preprocessed data and precomputed KNN indices loaded from disk to generate recommendations.



## Example Payload

Here is an example payload you can use to test model inference.
 - `data` is made up of the following:
    - cust_id: The customer id for whom we want to generate recommendations.
    - num_recommendations: Value specifying the number of recommendations to display.

```json
{
    "inputs": [
        {
            "name": "input-0",
            "shape": [
                2
            ],
            "datatype": "INT64",
            "data": [
                1219564,
                3
            ]
        }
    ]
}

```

## References
The KNN and PCA used in this model are based on `scikit-learn’s` KNN and Dimensionality Reduction with Neighborhood Components Analysis:

- [Neighborhood Components Analysis (NCA)](https://scikit-learn.org/stable/auto_examples/neighbors/plot_nca_dim_reduction.html)

- [K-Nearest Neighbors Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)


The Santander dataset used in this model can be accessed on Kaggle:

- [Santander Product Recommendation Dataset](https://www.kaggle.com/competitions/santander-product-recommendation/data)

- **Disclaimer**: The above data set does not contain any real Santander Spain's customers, therefore, it is not representative of Spain's customer base.

Santander Bank offers a lending hand to their customers through personalized product recommendations:

- [Santander Bank](https://www.santanderbank.com/home)