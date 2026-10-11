from langgraph.constants import START, END
from langgraph.graph.state import CompiledStateGraph, StateGraph

from knowledge.processor.query_process.base import setup_logging
from knowledge.processor.query_process.nodes.answer_output import AnswerOutputNode
from knowledge.processor.query_process.nodes.hyde_search import HyDeSearchNode
from knowledge.processor.query_process.nodes.item_name_confirm import ItemNameConfirmNode
from knowledge.processor.query_process.nodes.rerank import RerankNode
from knowledge.processor.query_process.nodes.rrf import RrfNode
from knowledge.processor.query_process.nodes.vector_search import VectorSearchNode
from knowledge.processor.query_process.nodes.web_search_mcp import WebSearchMcpNode
from knowledge.processor.query_process.state import QueryGraphState


def route_after_item_confirm(state: QueryGraphState) -> bool:
    """商品名称确认后的路由逻辑。

    根据是否已有答案决定是否跳过搜索直接输出。
    Args:
        state: 查询图状态。
    Returns:
        True 表示已有答案需要跳过搜索，False 表示继续搜索流程。
    """
    if state.get("answer"):
        return True
    return False


def create_query_graph() -> CompiledStateGraph:
    """创建查询流程图。

    Returns:
        编译后的 StateGraph 实例。

    流程结构::

        item_name_confirm
              │
              ├── (有答案) ──────────────────────────> answer_output
              │                                            │
              └── (无答案)                                  │
                   │                                       │
                   v                                       │
              multi_search                                 │
                   │                                       │
             ┌─────┼──────────┐                            │
             │     │          │                            │
             v     v          v                            │
        embedding  hyde    web_mcp                         │
             │     │          │                            │
             └─────┼──────────┘                            │
                   │                                       │
                   v                                       │
                 join                                      │
                   │                                       │
                   v                                       │
                  rrf                                      │
                   │                                       │
                   v                                       │
                rerank                                     │
                   │                                       │
                   v                                       │
             answer_output <───────────────────────────────┘
                   │
                   v
                  END

    Returns:
        编译后的 StateGraph 实例。
    """
    # 1. 定义LangGraph工作流
    workflow = StateGraph(QueryGraphState)

    # 2定义节点
    nodes: dict = {
        "item_name_confirm":ItemNameConfirmNode(),      # 确认商品名
        "multi_search":lambda x: x,                     # 虚拟节点(分发)
        "vector_search":VectorSearchNode(),             # 向量库检索
        "search_embedding_hyde": HyDeSearchNode(),      # 假设文档嵌入 假想答案，还没查库
        "web_search_mcp": WebSearchMcpNode(),           # 外部 web搜索
        "join": lambda x: {},                           # 虚拟节点（汇合）
        "rrf": RrfNode(),                               # 融合排序节点
        "rerank": RerankNode(),                         # 重排序节点
        "answer_output": AnswerOutputNode()             # 答案输出
    }

    # 2.1添加入口节点
    workflow.set_entry_point("item_name_confirm")

    # 2.2添加所有节点到图中
    for name, node in nodes.items():
        workflow.add_node(name, node)

    # 3.定义边
    # 3.1添加条件边：商品名称确认后根据是否有答案路由
    workflow.add_conditional_edges(
        "item_name_confirm",
        route_after_item_confirm,
        {
            False: "multi_search",
            True: "answer_output"
        }
    )

    # 3.2多路搜索分发（并发执行）
    workflow.add_edge("multi_search", "vector_search")
    workflow.add_edge("multi_search", "search_embedding_hyde")
    workflow.add_edge("multi_search", "web_search_mcp")

    # 3.3多路搜索汇合
    workflow.add_edge("vector_search", "join")
    workflow.add_edge("search_embedding_hyde", "join")
    workflow.add_edge("web_search_mcp", "join")

    # 3.4顺序边
    workflow.add_edge("join", "rrf")
    workflow.add_edge("rrf", "rerank")
    workflow.add_edge("rerank", "answer_output")
    workflow.add_edge("answer_output", END)


    # 4.编译图(返回可运行的状态)
    return workflow.compile()


# 创建全局图实例
query_app = create_query_graph()


if __name__ == "__main__":
    from knowledge.processor.query_process.state import create_default_state
    import json

    init_state: QueryGraphState | dict = create_default_state(
        task_id="task_001",
        original_query="RS-12 数字万用表怎么测试电阻？以及华为擎云L420 用户手册 中包含操作环境嘛？"
    )

    setup_logging()

    final_state1 = query_app.invoke(init_state)
    print(json.dumps(final_state1, indent=4, ensure_ascii=False))
    query_app.get_graph().print_ascii()




