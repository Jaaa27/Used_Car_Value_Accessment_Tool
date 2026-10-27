from enum import Enum, auto
import pandas as pd
from ai_client import evaluate_car
from data_loader import load_cars
from datetime import datetime

# 1. 状态锁：用 Enum 锁死所有合法挡位，绝不会因为拼错字母而爆缸
class EngineState(Enum):
    INIT = auto()
    READ = auto()
    EVAL = auto()
    VALIDATE = auto()
    DONE = auto()
    ERROR = auto()

# 2. ECU 容器：将 FSM 封装为类，建立系统边界
class FlipAssessmentEngine:
    def __init__(self):
        self.state = EngineState.INIT
        self.final_results = []
        self.cars_to_process = 5  # 临时计数器，模拟总共有3台车
        
    def run(self):
        print("Assessment In Progress ...")
        try:
            # 主控循环只负责路由分配；DONE / ERROR 都是终止态
            while self.state not in (EngineState.DONE, EngineState.ERROR):
                if self.state == EngineState.INIT:
                    self._handle_init()
                elif self.state == EngineState.READ:
                    self._handle_read()
                elif self.state == EngineState.EVAL:
                    self._handle_eval()
                elif self.state == EngineState.VALIDATE:
                    self._handle_validate()
        finally:
            # 无论正常结束、ERROR 还是意外异常，都导出已完成的结果
            self._export()
        if self.state == EngineState.DONE:
            print("Assessment done!")
        else:
            print("Assessment aborted (see errors above)")
    # 3. 独立模块：具体业务逻辑被拆分到单独的函数中
    def _handle_init(self):
        print("[INIT] Loading Car Data...")
        self.car_queue = load_cars("cars.csv", limit=5)
        self.state = EngineState.READ

    def _handle_read(self):
        print("[READ] 正在拉取下一台车源数据...")
        if len(self.car_queue) > 0:
            self.current_car = self.car_queue.pop(0)
            make = self.current_car.get('make', '未知车型')
            price = self.current_car.get('price', '未知价格')
            print(f"[READ] 正在拉取: {make} - {price}")
            self.state = EngineState.EVAL
        else:
            print("[READ] 车源库已清空。")
            self.state = EngineState.DONE

    def _handle_eval(self):
        print("[EVAL] 调用大模型进行车辆价值评估...")
        try:
            ai_result = evaluate_car(self.current_car)
            print(f" => 建议买入: {ai_result.is_profitable} | 预估维修: ${ai_result.estimated_repair_cost} | 理由: {ai_result.reasoning}")  
            self.current_car['AI_Assessment'] = ai_result.model_dump()
            self.state = EngineState.VALIDATE
        except Exception as e:
            print(f"[EVAL] ECU 响应异常，API 请求失败: {e}")
            self.state = EngineState.ERROR
        
    def _handle_validate(self):
        print("[VALIDATE] 校验数据合法性...")
        a = self.current_car["AI_Assessment"]
        if a["estimated_repair_cost"] < 0 or not a["reasoning"].strip():
            print(f"[VALIDATE] 丢弃异常评估: {self.current_car.get('listingId')}")
        else:
            self.final_results.append(self.current_car)
        self.state = EngineState.READ
    def _export(self):
        if not self.final_results:
            print("[EXPORT] 无结果，不生成报表（避免覆盖旧文件）")
            return
        rows = []
        for car in self.final_results:
            a = car["AI_Assessment"]
            rows.append({
                **{k: v for k, v in car.items() if k != "AI_Assessment"},
                "buy": a["is_profitable"],
                "repair_cost": a["estimated_repair_cost"],
                "defects": "; ".join(a["hidden_defects"]),
                "reasoning": a["reasoning"],
            })
        path = f"report_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
        pd.DataFrame(rows).to_excel(path, index=False)
        print(f"[EXPORT] {len(rows)} 条 -> {path}")