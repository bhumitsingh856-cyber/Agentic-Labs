from openai import OpenAI
import os
# The correct cloud gateway for Nvidia Hosted NIMs
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1", 
    api_key=os.getenv("NVIDIA_API_KEY")
)

# Call the image generator
response = client.images.generate(
    model="qwen/qwen-image",
    prompt="A futuristic city with flying cars at sunset, cinematic lighting, highly detailed",
)

# Extract and isolate the string URL
image_url = response.data[0].url
print(f"Success! Image generated at: {image_url}")