## Description

This is an example of serving a small LLM with Ollama. The LLM in this example is [`phi3:3.8b`](https://ollama.com/library/phi3:3.8b)

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
            "data": ["What is the capital of France?"]
        }
    ]
}
```
