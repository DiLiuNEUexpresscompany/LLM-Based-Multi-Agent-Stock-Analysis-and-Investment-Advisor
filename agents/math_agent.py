# If necessary, install the openai Python library by running 
# pip install openai
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
	base_url="https://mqiez28uc19t0z3q.us-east-1.aws.endpoints.huggingface.cloud/v1/", 
	api_key=os.getenv("HUGGUNGFACE_ENDPOINT_API_KEY")
)

chat_completion = client.chat.completions.create(
	model="tgi",
	messages=[
    {
        "role": "user",
        "content": "小明折了9只纸飞机，比小军少折3只，小军折了几只纸飞机？"
    }
],
	top_p=None,
	temperature=None,
	max_tokens=150,
	stream=True,
	seed=None,
	frequency_penalty=None,
	presence_penalty=None
)

for message in chat_completion:
	print(message.choices[0].delta.content, end="")