from openai import OpenAI

# The correct cloud gateway for Nvidia Hosted NIMs
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1", 
    api_key="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"
)

# Call the image generator
response = client.images.generate(
    model="qwen/qwen-image",
    prompt="A futuristic city with flying cars at sunset, cinematic lighting, highly detailed",
)

# Extract and isolate the string URL
image_url = response.data[0].url
print(f"Success! Image generated at: {image_url}")