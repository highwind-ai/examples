## Description

An example of a machine translation model that translates Dyula to French using a pre-trained `t5-small` model form Hugging Face.

This following example was trained on a Dyula-French translation datset that was kindly created by [data354](https://data354.com/en/).

> This is not a production-ready model since it hasn't been trained for many epochs. It is for illustration purposes only.

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
                "i tɔgɔ bi cogodɔ"
            ]
        }
    ]
}
```
