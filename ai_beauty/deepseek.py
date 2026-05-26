
import os
import time
from datetime import datetime
from openai import OpenAI

client = OpenAI(api_key=os.environ.get('DEEPSEEK_API_KEY'), base_url="https://api.deepseek.com")


def deepseek(model: str, prompt: str, budget_tokens: int, img_urls: dict) -> dict:

    start_time = time.perf_counter()

    image_blocks = [
        {
            "type": "image_url",
            "image_url": {"url": data_uri},
        }
        for data_uri in img_urls.values()
    ]

    response = client.chat.completions.create(
        model=model,
        extra_body={"thinking": {"type": "enabled", "budget_tokens": budget_tokens}},
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}, *image_blocks]
            }
        ],
        stream=False
    )

    wall_time = time.perf_counter() - start_time

    message = response.choices[0].message
    thinking_text = message.reasoning_content or ""
    response_text = message.content or ""

    return {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "budget_tokens": budget_tokens,
        "rt_tokens": response.usage.completion_tokens,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "budget_utilisation": response.usage.completion_tokens / budget_tokens,
        "response_text": response_text,
        "stop_reason": response.choices[0].finish_reason,
        "thinking_text": thinking_text,
        "wall_time_s": wall_time,
    }
