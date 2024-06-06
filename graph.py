import json
import matplotlib.pyplot as plt
import pandas as pd


def plot_prompt_analysis(json_filepath):
    """
    读取JSON文件内容并生成一个双竖轴折线图，反映每个Prompt的输入token、输出token以及响应时间的变化。

    Parameters:
    json_filepath (str): JSON文件的路径
    """
    # 读取JSON文件
    with open(json_filepath, 'r') as file:
        data = json.load(file)

    # 提取相关数据
    prompts = ['prompt1', 'prompt2', 'prompt3', 'prompt4', 'prompt5', 'prompt6', 'prompt7', 'prompt8', 'prompt9']
    in_tokens = [data[prompt]['in_token'] for prompt in prompts]
    out_tokens = [data[prompt]['out_token'] for prompt in prompts]
    take_times = [data[prompt]['take_time'] for prompt in prompts]
    # max_diff_times = [data[prompt]['max_response_time_diff'] for prompt in prompts]

    # 创建DataFrame
    df = pd.DataFrame({
        'Prompt': prompts,
        'Input Tokens': in_tokens,
        'Output Tokens': out_tokens,
        # 'Max Diff Time': max_diff_times,
        'Response Time (ms)': take_times
    })

    # 创建双竖轴图
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制输入和输出token的折线图
    ax1.set_xlabel('Prompt')
    ax1.set_ylabel('Tokens', color='tab:blue')
    ax1.plot(df['Prompt'], df['Input Tokens'], marker='o', label='Input Tokens', color='tab:blue')
    ax1.plot(df['Prompt'], df['Output Tokens'], marker='o', label='Output Tokens', color='tab:cyan')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # 创建第二个y轴共享x轴
    ax2 = ax1.twinx()
    ax2.set_ylabel('Response Time (ms)', color='tab:red')
    ax2.plot(df['Prompt'], df['Response Time (ms)'], marker='o', label='Response Time (ms)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # 添加图例
    fig.tight_layout()
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

    plt.title('Prompt Analysis: Input Tokens, Output Tokens, and Response Time')
    plt.grid(True)
    plt.show()


# 调用函数示例
# plot_prompt_analysis('result.json')

if __name__ == "__main__":
    # 从prompts.txt读取内容
    plot_prompt_analysis('result.json')
