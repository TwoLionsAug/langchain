from pathlib import Path
import os
from typing import Literal, TypedDict
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

env = load_dotenv(override=True)

# 创建大模型
def create_deepseek() -> ChatDeepSeek:
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")
    if not DEEPSEEK_API_KEY:
        raise ValueError("未找到 DEEPSEEK_API_KEY，请先在项目根目录的 .env 中配置。")
    return ChatDeepSeek(
      api_key= DEEPSEEK_API_KEY,
      base_url= DEEPSEEK_BASE_URL,
      model= "deepseek-v4-pro"
    )
# 创建一次模型即可，后面的不同节点可以重复使用它。
llm = create_deepseek()
class StudentState(TypedDict, total=False):
    name : str
    sex: str
    age:int
    chinese_score :int
    english_score : int
    math_score : int

def prepare_node(state: StudentState) -> StudentState:
    return {
        "chinese_score": state["chinese_score"],
        "english_score":  state["english_score"],
        "math_score":  state["math_score"],
    }

def prompt_template(state : StudentState)-> ChatPromptTemplate:
    status = "已经通过"
    if state['chinese_score'] >59 and state['english_score'] >59  and state['math_score'] >59:
        status = "已经通过,请夸奖他，再让他继续努力"
    else:
        status = "暂未通过,请先鼓励他，再给出2条下一步学习建议。"

    return ChatPromptTemplate.from_messages(
        [
            ("你是一位耐心的编程老师，回答要简短、具体、使用中文。"),
            (
                    "学生{name}，性别{sex},年龄{age}岁，"
                    "语文得分{chinese_score}分"
                    "英语得分{english_score}分"
                    "数学得分{math_score}分"
                    f"{status}"
            ),
        ]
    )


def pass_node(state:StudentState) -> dict:
    template_message = prompt_template(state)
    feed_invoke = template_message | llm
    response = feed_invoke.invoke(  {
            "name": state["name"],
            "sex": state["sex"],
            "age": state["age"],
            "chinese_score": state["chinese_score"],
            "english_score": state["english_score"],
            "math_score": state["math_score"],
    })
    print(response.content)
    return {"result" : str(response.content)}

def no_pass_node(state:StudentState) -> dict:
    template_message = prompt_template(state)
    feed_invoke = template_message | llm
    response = feed_invoke.invoke( {
            "name": state["name"],
            "sex": state["sex"],
            "age": state["age"],
            "chinese_score": state["chinese_score"],
            "english_score": state["english_score"],
            "math_score": state["math_score"],
    })
    print(response.content)
    return {"result" : str(response.content)}

def route_by_score(state: StudentState) -> Literal["pass", "retry"]:
    """根据分数选择下一条路径。"""
    if state["chinese_score"] >= 60 and state["english_score"] >= 60 and state["math_score"] >= 60:
        return "pass"
    return "no_pass"

def create_graph():
    builder = StateGraph(StudentState)

    # 注册节点。左边是节点名称，右边是执行该节点的函数。
    builder.add_node("prepare", prepare_node)
    builder.add_node("pass", pass_node)
    builder.add_node("no_pass", no_pass_node)

    # START 是 LangGraph 内置的起点。
    builder.add_edge(START, "prepare")

    # prepare 执行完后调用 route_by_score，并按返回结果选择节点。
    builder.add_conditional_edges(
        "prepare",
        route_by_score,
        {
            "pass": "pass",
            "no_pass": "no_pass",
        },
    )

    # 两个分支执行完毕后都进入 END，表示工作流结束。
    builder.add_edge("pass", END)
    builder.add_edge("no_pass", END)

    # StateGraph 是流程设计图；compile 后才得到真正可运行的 graph。
    return builder.compile()

if __name__ == "__main__":
    graph = create_graph()

    # invoke 会从 START 开始执行，并返回流程结束时的完整 State。
    final_state = graph.invoke(
        {
            "name": "小明",
            "sex" :"男",
            "age" :12,
            "chinese_score": 50,
            "english_score": 40,
            "math_score": 52,
        }
    )

    print("\n最终状态：")
    print(final_state)

    # 本例的流程图：
    # START -> prepare -> route_by_score
    #                         |-- score >= 60 --> pass  -> END
    #                         |-- score <  60 --> retry -> END
