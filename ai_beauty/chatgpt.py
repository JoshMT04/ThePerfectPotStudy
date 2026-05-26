
import os
import time
from datetime import datetime
from openai import OpenAI

client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

def chatgpt(model: str, prompt: str, budget_tokens: int, img_urls: dict) -> dict:
    
    start_time = time.perf_counter()

    image_blocks = [
        {
            "type": "input_image",
            "image_url": data_uri,
        }
        for data_uri in img_urls.values()
    ]

    response = client.responses.create(
        model=model,
        reasoning={"effort": "high"},
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}, *image_blocks]
            }
        ],
        max_output_tokens=budget_tokens,
    )

    wall_time = time.perf_counter() - start_time

    response_text = response.output_text

    thinking_text = ""
    for item in response.output:
        if item.type == "reasoning":
            for summary in (item.summary or []):
                thinking_text += summary.text

    return {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "budget_tokens": budget_tokens,
        "rt_tokens": response.usage.output_tokens_details.reasoning_tokens,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "budget_utilisation": response.usage.output_tokens / budget_tokens,
        "response_text": response_text,
        "stop_reason": response.status,
        "thinking_text": thinking_text,
        "wall_time_s": wall_time,
    }
