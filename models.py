from typing import Literal

from pydantic import BaseModel, Field


class FlipAssessment(BaseModel):
    # 字段顺序即模型生成顺序：先写依据，再下结论，结论会更一致
    hidden_defects: list[str] = Field(
        description=(
            "该车型、年款、里程段已知的常见机械或电子隐患，每条一句话，最多 5 条。"
            "没有可靠依据时返回空列表；不要列通用检查清单，不要凭空推测事故史或车况。"
        )
    )
    estimated_repair_cost: int = Field(
        description=(
            "整备与维修总成本估计，单位美元，整数。包含基础整备（保养、深度清洁、"
            "轮胎刹车等易耗件），因此通常不为 0；不含购车价、过户费和运输费。"
        )
    )
    reasoning: str = Field(
        description=(
            "中文理由，不超过 3 句话。必须引用输入里的具体数字"
            "（如 price、comp_median、delta_pct、mileage），"
            "不得编造转售价、车况或市场行情。"
        )
    )
    is_profitable: bool = Field(
        description=(
            "是否建议作为翻新项目买入。以 comp_median 作为预期转售价，"
            "comp_median - price - estimated_repair_cost 为正且留有安全边际，"
            "同时隐患风险可控，才为 true。"
        )
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "对本次评估的把握程度。low：同款样本少或缺少该车型的可靠信息；"
            "medium：信息大体足够；high：该车型常见问题明确且 comp_n 充足。"
            "comp_n 小于 5 时不得选 high。"
        )
    )