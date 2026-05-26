import anthropic
from dotenv import load_dotenv
import time
from datetime import datetime

load_dotenv()
client = anthropic.Anthropic()

def claude(model: str, prompt: str, budget_tokens: int, img_urls: dict) -> dict:

    start_time = time.perf_counter()

    image_blocks = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": data_uri.split(",", 1)[1],
            }
        }
        for data_uri in img_urls.values()
    ]

    response = client.messages.create(
        model=model,
        max_tokens=budget_tokens + 2000,
        thinking={"type": "enabled", "budget_tokens": budget_tokens},
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, *image_blocks]
            }
        ]
    )
    
    wall_time = time.perf_counter() - start_time
    
    thinking_text = ""
    response_text = ""
    
    for block in response.content:
        if block.type == "thinking":
            thinking_text = block.thinking
        elif block.type == "text":
            response_text = block.text
    
    return {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "budget_tokens": budget_tokens,
        "rt_tokens": response.usage.output_tokens,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "budget_utilisation": response.usage.output_tokens / budget_tokens,
        "response_text": response_text,
        "stop_reason": response.stop_reason,
        "thinking_text": thinking_text,
        "wall_time_s": wall_time,
    }