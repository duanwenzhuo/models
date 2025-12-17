import operator
from typing import Annotated, TypedDict, Dict, Any, List, Optional

from langgraph.graph import StateGraph, END

from agents import (
    PlannerAgent,
    PreprocessAgent,
    IntegrationAgent,
    InspectionAgent,
    EvaluationAgent,
    ReporterAgent,
    TuningAgent,
)


# ==========================================
# 1. 定义状态 (State)
# ==========================================
class AgentState(TypedDict):
    """
    定义整个工作流中共享的数据结构。
    LangGraph 会在节点之间传递这个 State。
    """

    # 输入信息
    user_intent: str
    data_path: str

    # 规划结果
    plan: Dict[str, Any]

    # 数据对象 (AnnData) - 注意：内存在节点间传递引用
    data_hvg: Optional[Any]
    data_raw: Optional[Any]
    data_hvg_full: Optional[Any]
    data_raw_full: Optional[Any]

    # 最终结果
    results: Dict[str, Any]

    # 数据检查
    inspection: Optional[Dict[str, Any]]
    evaluation: Optional[Any]
    report_path: Optional[str]
    benchmark_fraction: Optional[float]
    run_all_methods: bool
    method_candidates: Optional[List[str]]
    param_presets: Optional[Dict[str, Dict[str, Dict[str, Any]]]]
    chosen_params: Optional[Dict[str, List[str]]]
    param_results: Optional[Dict[str, Any]]
    compute_budget: Optional[Dict[str, Any]]
    param_search_results: Optional[List[Dict[str, Any]]]
    best_params: Optional[Dict[str, Any]]
    best_rank: Optional[Dict[str, Any]]
    method_errors: Optional[Dict[str, Any]]
    final_selection: Optional[Dict[str, Any]]
    top1_only: bool
    search_params: bool

    # 执行日志/消息历史 (可选)
    logs: Annotated[List[str], operator.add]
    error: Optional[str]


# ==========================================
# 2. 定义节点 (Nodes)
# ==========================================
def inspection_node(state: AgentState) -> AgentState:
    """运行 Inspection Agent"""
    try:
        agent = InspectionAgent()
        print("[Workflow] Entering Inspection Node")
        new_state = agent.run(state)
        new_state["logs"] = state.get("logs", []) + ["Inspection finished."]
        return new_state
    except Exception as e:
        return {**state, "error": str(e), "logs": state.get("logs", []) + [f"Inspection failed: {e}"]}


def planner_node(state: AgentState) -> AgentState:
    """运行 Planner Agent"""
    try:
        agent = PlannerAgent()
        print("[Workflow] Entering Planner Node")
        new_state = agent.run(state)
        new_state["logs"] = state.get("logs", []) + ["Planner finished successfully."]
        return new_state
    except Exception as e:
        return {**state, "error": str(e), "logs": state.get("logs", []) + [f"Planner failed: {e}"]}


def tuning_node(state: AgentState) -> AgentState:
    """运行 Tuning Agent"""
    if state.get("error"):
        return state
    try:
        agent = TuningAgent()
        print("[Workflow] Entering Tuning Node")
        new_state = agent.run(state)
        new_state["logs"] = state.get("logs", []) + ["Tuning finished."]
        return new_state
    except Exception as e:
        return {**state, "error": str(e), "logs": state.get("logs", []) + [f"Tuning failed: {e}"]}


def preprocess_node(state: AgentState) -> AgentState:
    """运行 Preprocess Agent"""
    if state.get("error"):
        return state

    try:
        agent = PreprocessAgent()
        print("[Workflow] Entering Preprocess Node")
        new_state = agent.run(state)
        new_state["logs"] = state.get("logs", []) + ["Preprocessing finished."]
        return new_state
    except Exception as e:
        return {**state, "error": str(e), "logs": state.get("logs", []) + [f"Preprocessing failed: {e}"]}


def integration_node(state: AgentState) -> AgentState:
    """运行 Integration Agent"""
    if state.get("error"):
        return state

    try:
        agent = IntegrationAgent()
        print("[Workflow] Entering Integration Node")
        new_state = agent.run(state)
        new_state["logs"] = state.get("logs", []) + ["Integration finished."]
        return new_state
    except Exception as e:
        return {**state, "error": str(e), "logs": state.get("logs", []) + [f"Integration failed: {e}"]}


def evaluation_node(state: AgentState):
    agent = EvaluationAgent()
    print("[Workflow] Entering Evaluation Node")
    new_state = agent.run(state)
    new_state["logs"] = state.get("logs", []) + ["Evaluation finished."]
    return new_state


def reporter_node(state: AgentState):
    agent = ReporterAgent()
    print("[Workflow] Entering Reporter Node")
    new_state = agent.run(state)
    new_state["logs"] = state.get("logs", []) + ["Reporter finished."]
    return new_state


# ==========================================
# 3. 路由函数 (Routing)
# ==========================================
def should_preprocess(state: AgentState) -> str:
    """根据 Planner 结果判断是否需要运行 Preprocess Agent。"""
    plan = state.get("plan") or {}
    error = state.get("error")

    if error:
        print("[Router] Planner failed, proceeding to END.")
        return END

    if not plan:
        print("[Router] No plan found, defaulting to Preprocessor.")
        return "preprocessor"

    skip_all = all(not subplan.get("preprocess", True) for subplan in plan.values())
    data_ready = bool(state.get("data_hvg"))

    if skip_all and data_ready:
        print("[Router] All modalities set preprocess=False and data exists. Skipping Preprocessor.")
        return "integrator"

    print("[Router] Proceeding to Preprocessor.")
    return "preprocessor"


def check_results_quality(state: AgentState) -> str:
    """初步治理层检查，若结果存在则继续，否则结束。"""
    error = state.get("error")
    if error:
        print(f"[Router] Error detected: {error}. Proceeding to Reporter.")
        return "reporter"

    results = state.get("results", {})
    if not results:
        print("[Router] No integration results found. Proceeding to Reporter.")
        return "reporter"

    print("[Router] Integration successful. Proceeding to Evaluator.")
    return "evaluator"


# ==========================================
# 4. 构建图 (Graph Construction)
# ==========================================
def create_workflow_app():
    """创建并编译 LangGraph 工作流"""

    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("inspection", inspection_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("tuning", tuning_node)
    workflow.add_node("preprocessor", preprocess_node)
    workflow.add_node("integrator", integration_node)
    workflow.add_node("evaluator", evaluation_node)
    workflow.add_node("reporter", reporter_node)

    # 定义执行顺序
    workflow.set_entry_point("inspection")
    workflow.add_edge("inspection", "planner")
    workflow.add_edge("planner", "tuning")

    workflow.add_conditional_edges(
        "tuning",
        should_preprocess,
        {
            "preprocessor": "preprocessor",
            "integrator": "integrator",
            END: END,
        },
    )

    workflow.add_edge("preprocessor", "integrator")

    workflow.add_conditional_edges(
        "integrator",
        check_results_quality,
        {
            "evaluator": "evaluator",
            "reporter": "reporter",
        },
    )

    workflow.add_edge("evaluator", "reporter")
    workflow.add_edge("reporter", END)

    app = workflow.compile()
    return app
