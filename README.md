<div align="center">

<h1>Google Paligemma | Worker</h1>

🚀 | RunPod implementation of Google Paligemma for serverless deployment.
</div>

## 📖 | Getting Started

### Params:
Input example:
```json
{
  "input": {
    "prompt": "What is the brand and model of the car in the image?",
    "image_url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
  }
}
```

### To make updates:

1. Clone this repository.
2. Obtain access to the Paligemma model on Huggingface
3. Build a container using token in the command: `HF_TOKEN=<YOUR_TOKEN> docker build --secret id=HF_TOKEN .`
4. Push it to your DockerHub account
5. Create a new serverless endpoint on RunPod with the image you just pushed and specify `HF_TOKEN` env variable

## 🔗 | Links
- 🐳 [Docker Container](https://hub.docker.com/r/hypnocapybara/runpod-paligemma)
- [RunPod workers repo](https://github.com/runpod-workers)
