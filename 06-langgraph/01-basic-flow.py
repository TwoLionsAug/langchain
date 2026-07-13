"""LangGraph + DeepSeek 基础流程示例。

这个例子会读取项目根目录 .env 中的 DeepSeek 配置并调用大模型。
它演示了 LangGraph 最核心的几个概念：

1. State（状态）：节点之间共享的数据
2. Node（节点）：处理状态的普通 Python 函数
3. Edge（边）：规定节点的执行顺序
4. Conditional Edge（条件边）：根据状态选择下一条路径
5. compile / invoke：编译并运行工作流
6. LLM Node：在 LangGraph 节点中调用 DeepSeek
"""

import os
from pathlib import Path
from typing import Literal, TypedDict

import httpx
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph


# 当前文件在 06-langgraph 中，parent.parent 就是项目根目录。
# 显式指定 .env 路径后，即使从其他目录运行本文件也能正确加载配置。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)


def create_model() -> ChatDeepSeek:
    """根据 .env 配置创建 DeepSeek 模型对象。"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("未找到 DEEPSEEK_API_KEY，请先在项目根目录的 .env 中配置。")

    return ChatDeepSeek(
        # 如果 .env 没有指定模型，就默认使用 deepseek-chat。
        model=os.getenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat"),
        api_key=api_key,
        base_url=os.getenv("DEEPSEEK_BASE_URL") or None,
        temperature=0.3,
        # 不继承电脑上的全局代理设置，避免代理缺少 SOCKS 依赖时初始化失败。
        # 如果你的网络必须通过代理访问 DeepSeek，可以删除下面两行。
        http_client=httpx.Client(trust_env=False),
        http_async_client=httpx.AsyncClient(trust_env=False),
    )


# 创建一次模型即可，后面的不同节点可以重复使用它。
llm = create_model()


# -----------------------------------------------------------------------------
# 1. 定义状态 State
# -----------------------------------------------------------------------------
# State 可以理解为整个流程共用的“数据包”。
# 每个节点都会接收当前 State，并返回自己想更新的字段。
class LearningState(TypedDict):
    user_name: str
    learning_topic: str
    score: int
    result: str


# -----------------------------------------------------------------------------
# 2. 定义节点 Node
# -----------------------------------------------------------------------------
# 节点本质上就是普通 Python 函数：接收 state，返回需要更新的数据。
def prepare_node(state: LearningState) -> dict:
    """准备节点：打印用户信息，并保留初始分数。"""
    print(
        f"[prepare] 你好，{state['user_name']}，"
        f"你正在学习 {state['learning_topic']}，当前分数是 {state['score']}"
    )

    # 这里只返回 score，LangGraph 会把它合并到原来的 State 中。
    return {"score": state["score"]}


def pass_node(state: LearningState) -> dict:
    """通过节点：让 DeepSeek 生成鼓励和下一步学习建议。"""
    print("[pass] 正在请求 DeepSeek 生成进阶建议……")

    response = llm.invoke(
        [
            SystemMessage(content="你是一位耐心的编程老师，回答要简短、具体、使用中文。"),
            HumanMessage(
                content=(
                    f"学生{state['user_name']}正在学习{state['learning_topic']}，"
                    f"测试得分{state['score']}分，已经通过。"
                    "请先鼓励他，再给出2条下一步学习建议。"
                )
            ),
        ]
    )
    return {"result": str(response.content)}


def retry_node(state: LearningState) -> dict:
    """重试节点：让 DeepSeek 生成复习建议。"""
    print("[retry] 正在请求 DeepSeek 生成复习建议……")

    response = llm.invoke(
        [
            SystemMessage(content="你是一位耐心的编程老师，回答要简短、具体、使用中文。"),
            HumanMessage(
                content=(
                    f"学生{state['user_name']}正在学习{state['learning_topic']}，"
                    f"测试得分{state['score']}分，暂未通过。"
                    "请安慰他，并给出2条最应该优先复习的建议。"
                )
            ),
        ]
    )
    return {"result": str(response.content)}


# -----------------------------------------------------------------------------
# 3. 定义条件路由函数
# -----------------------------------------------------------------------------
# 路由函数只负责“做决定”，返回值会决定流程进入哪个节点。
def route_by_score(state: LearningState) -> Literal["pass", "retry"]:
    """根据分数选择下一条路径。"""
    if state["score"] >= 60:
        return "pass"
    return "retry"


# -----------------------------------------------------------------------------
# 4. 创建并连接工作流
# -----------------------------------------------------------------------------
def create_graph():
    """创建、连接并编译 LangGraph 工作流。"""
    # 告诉 LangGraph：这个工作流中的状态结构是 LearningState。
    builder = StateGraph(LearningState)

    # 注册节点。左边是节点名称，右边是执行该节点的函数。
    builder.add_node("prepare", prepare_node)
    builder.add_node("pass", pass_node)
    builder.add_node("retry", retry_node)

    # START 是 LangGraph 内置的起点。
    builder.add_edge(START, "prepare")

    # prepare 执行完后调用 route_by_score，并按返回结果选择节点。
    builder.add_conditional_edges(
        "prepare",
        route_by_score,
        {
            "pass": "pass",
            "retry": "retry",
        },
    )

    # 两个分支执行完毕后都进入 END，表示工作流结束。
    builder.add_edge("pass", END)
    builder.add_edge("retry", END)

    # StateGraph 是流程设计图；compile 后才得到真正可运行的 graph。
    return builder.compile()


# -----------------------------------------------------------------------------
# 5. 运行工作流
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    graph = create_graph()

    # invoke 会从 START 开始执行，并返回流程结束时的完整 State。
    final_state = graph.invoke(
        {
            "user_name": "小明",
            "learning_topic": "LangGraph",
            "score": 75,  # 改成 59 再运行一次，可以看到 retry 分支。
            "result": "",
        }
    )

    print("\n最终状态：")
    print(final_state)

    # 本例的流程图：
    # START -> prepare -> route_by_score
    #                         |-- score >= 60 --> pass  -> END
    #                         |-- score <  60 --> retry -> END
