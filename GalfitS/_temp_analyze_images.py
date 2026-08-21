#!/usr/bin/env python3
"""临时脚本：使用 Kimi K2.6 分析两张 GalfitS 结果图的不同之处"""
import base64
import os
import sys

from openai import OpenAI

api_key = os.environ.get("KIMI_API_KEY", "")
if not api_key:
    print("错误: KIMI_API_KEY 未设置")
    sys.exit(1)

client = OpenAI(api_key=api_key, base_url="https://api.moonshot.cn/v1")

IMAGE1 = r"MIRI_NIRCam_result\two_sersic_C1C2\noSED\C1C2_noSEDimage_fit.png"
IMAGE2 = r"MIRI_NIRCam_result\MIRI_clump_departure\component_model_images.png"

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def analyze(image_path, prompt):
    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
    mime_type = mime_map.get(ext, "image/png")

    b64 = encode_image(image_path)
    print(f"正在分析: {image_path} ({mime_type})...", file=sys.stderr, flush=True)

    response = client.chat.completions.create(
        model="kimi-k2.6",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    return response.choices[0].message.content

# Step 1: 分别描述两张图
desc1 = analyze(IMAGE1, "请详细描述这张天文图像的内容，包括：有什么面板/子图，每个面板显示什么内容（如观测数据、模型、残差等），图中可见什么结构（如星系、团块等），以及任何标注或文字信息。这是星系拟合的结果图。")
desc2 = analyze(IMAGE2, "请详细描述这张天文图像的内容，包括：有什么面板/子图，每个面板显示什么内容（如不同波段的图像、模型分量等），图中可见什么结构，以及任何标注或文字信息。这是星系成分模型图。")

# Step 2: 对比分析
comparison = analyze(
    IMAGE1,
    f"""请对比你刚才看到的这张图（C1C2 noSED fit）和之前看过的另一张图（component_model_images），分析它们的不同之处。

C1C2 noSED fit 图的描述：{desc1}

Component model images 图的描述：{desc2}

请从以下角度分析两张图的不同：
1. 科学目的不同（一张是拟合结果，一张是成分分解）
2. 显示内容不同（波段、面板布局、colorbar等）
3. 数据结构/特征不同
4. 图像中星系结构的呈现差异

请给出结构化的对比分析。"""
)

print("=" * 80)
print("【图1: C1C2_noSEDimage_fit.png — 描述】")
print("=" * 80)
print(desc1)
print()
print("=" * 80)
print("【图2: component_model_images.png — 描述】")
print("=" * 80)
print(desc2)
print()
print("=" * 80)
print("【两张图的差异分析】")
print("=" * 80)
print(comparison)
