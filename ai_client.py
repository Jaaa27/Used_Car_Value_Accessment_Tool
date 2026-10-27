import os

from dotenv import load_dotenv
from openai import OpenAI

from data_loader import build_prompt_payload
from models import FlipAssessment

load_dotenv()
client = OpenAI(api_key=os.getenv("API_KEY"))

SYSTEM_PROMPT = """你是二手车翻新套利评估员。输入是一辆二手车的 JSON，字段含义：
- price：挂牌价（美元）；mileage：里程（英里）
- comp_median：同款车（同品牌、同车型、年份相差不超过 1）的中位挂牌价，作为预期转售价基准
- comp_n：同款样本数；comp_mileage_median：同款中位里程
- delta_pct：本车挂牌价相对 comp_median 的百分比，负数表示更便宜

规则：
1. 只依据输入字段评估，不要编造车况、事故史、转售价或市场行情。
2. hidden_defects 只写该车型、年款、里程段已知的常见问题；没有依据就返回空列表。
3. 若本车里程明显高于 comp_mileage_median，折价可能只是里程造成的，要在 reasoning 里说明。
4. 无法判断时，把 confidence 设为 low。
5. 全部用中文回答。"""


def evaluate_car(car_data: dict) -> FlipAssessment | None:
    """接收车辆字典，调用 AI 评估，返回 FlipAssessment（模型拒答时为 None）"""
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        temperature=0,   # 同一辆车重跑，结果尽量一致
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"请评估这台二手车：{build_prompt_payload(car_data)}"},
        ],
        response_format=FlipAssessment,
    )
    return response.choices[0].message.parsed