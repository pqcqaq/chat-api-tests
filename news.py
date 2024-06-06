import os

import requests
import json
import time
import copy
import tiktoken

from graph import plot_prompt_analysis

# 替换为你的API URL和API KEY
API_URL = 'http://localhost:8866/v1/chat/completions'
API_KEY = 'your-api-key'
HEADERS = {'Authorization': f'Bearer {API_KEY}'}

REQUEST_JSON = {
    "model": "chatglm3-6b",
    "messages": [
        {
            "role": "user",
            "content": "{Prompt}"
        }
    ],
    "stream": False,
    "max_tokens": 2048,
    "temperature": 10,
    "top_p": 0.8
}

# 初始化tiktoken编码器
encoder = tiktoken.get_encoding("cl100k_base")

def replacer(prompt):
    json_copy = copy.deepcopy(REQUEST_JSON)
    json_copy['messages'][0]['content'] = json_copy['messages'][0]['content'].replace("{Prompt}", prompt)
    return json_copy

def count_response_tokens(response_json):
    return len(encoder.encode(response_json.get('choices', [{}])[0].get('message', {}).get('content', '')))

def call_api(prompt, thread_name, i, j):
    start_time = time.time()
    try:
        response = requests.post(API_URL, headers=HEADERS, json=replacer(prompt))
        response.raise_for_status()
        end_time = time.time()
        response_time = (end_time - start_time) * 1000  # 转换为毫秒
        response_json = response.json()
        out_token = count_response_tokens(response_json)
        # 将结果输出到文件
        with open(f'news-results/{thread_name}_prompt{i + 1}_round{j + 1}.json', 'w', encoding='utf-8') as f:
            json.dump(response_json, f, indent=4, ensure_ascii=False)
        return response_time, out_token, True
    except requests.exceptions.RequestException as e:
        print(f"API call failed for prompt '{prompt}': {e}")
        return 0, 0, False

def count_tokens(text):
    return len(encoder.encode(text))

def test_single_thread(prompts, test_rounds):
    results = {}
    total_taken_time = 0
    total_success = 0
    total_failures = 0

    print("Starting single-thread test...")
    for i, prompt in enumerate(prompts):
        prompt_results = {
            "in_token": count_tokens(prompt),
            "out_token": 0,
            "take_time": 0,
            "success_count": 0,
            "failure_count": 0,
            "max_response_time_diff": 0
        }
        total_time = 0
        total_out_token = 0
        max_time = float('-inf')
        min_time = float('inf')

        for j in range(test_rounds):
            response_time, out_token, success = call_api(prompt, "single-thread", i, j)
            total_time += response_time
            total_out_token += out_token
            if success:
                prompt_results["success_count"] += 1
                total_success += 1
            else:
                prompt_results["failure_count"] += 1
                total_failures += 1

            if response_time > max_time:
                max_time = response_time
            if response_time < min_time:
                min_time = response_time

            print(f"Test {i + 1}, Round {j + 1}: {response_time:.2f}ms, {out_token} tokens, Success: {success}")

        prompt_results["out_token"] = total_out_token // test_rounds if prompt_results["success_count"] > 0 else 0
        prompt_results["take_time"] = total_time // test_rounds if prompt_results["success_count"] > 0 else 0
        prompt_results["max_response_time_diff"] = max_time - min_time if prompt_results["success_count"] > 0 else 0

        results[f"prompt{i + 1}"] = prompt_results
        total_taken_time += total_time

    results["total_tests"] = len(prompts)
    results["test_round"] = test_rounds
    results["total_taken_time"] = total_taken_time
    results["total_success"] = total_success
    results["total_failures"] = total_failures

    with open('news-result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print("Single-thread test completed!")

if __name__ == "__main__":
    # 从 news_prompt 文件夹中读取内容
    prompts = []
    for filename in os.listdir('news_prompt'):
        if filename.endswith('.txt'):
            with open(os.path.join('news_prompt', filename), 'r', encoding='utf-8') as f:
                prompts.append(f.read().strip())

    # 测试单线程
    test_rounds = 3
    test_single_thread(prompts, test_rounds)

    plot_prompt_analysis('news-result.json')