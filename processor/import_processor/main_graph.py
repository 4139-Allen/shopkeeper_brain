"""
导入流程主图
使用LangGraph构建文档导入流程
"""
import json

from langgraph.constants import START, END


from knowledge.processor.import_processor.base import setup_logging
from knowledge.processor.import_processor.nodes.entry import EntryNood
from knowledge.processor.import_processor.state import ImportGraphState, create_default_state
from langgraph.graph.state import CompiledStateGraph, StateGraph


# 定义入口节点的 条件路由
def entry_router(state: ImportGraphState):
    """
    入口节点后的条件路由
    根据文件类型判断是进入pdf转md节点，还是直接进入md处理分支
    :param state: 当前图状态
    :return 下一个节点的名称
    """



def create_import_graph() -> CompiledStateGraph:
    """
        创建导入流程图
        :return: 编译后的StateGraph实例
        流程结构：
            entry_node
                  │
                  ├── (PDF) ──> pdf_to_md_node ──┐
                  │                              │
                  └── (MD) ─────────────────────>├──> md_img_node
                                                  │
                                                  v
                                          document_split_node
                                                  │
                                                  v
                                          item_name_rec_node
                                                  │
                                                  v
                                            bge_embedding_node
                                                  │
                                                  v
                                            import_milvus_node
                                                  │
                                                  v
                                                 END
    """

    # 1.定义状态图
    graph_pipeline = StateGraph(ImportGraphState)

    # 2.定义节点
    nodes = {
        "entry_node": EntryNood()
    }

    # 2.1添加入口节点
    graph_pipeline.set_entry_point("entry_node")

    # 2.2添加所有节点到图中
    for key, value in nodes.items():
        graph_pipeline.add_node(key, value)

    # 3.定义边
    # 3.1条件边

    # 3.2 顺序边
    graph_pipeline.add_edge(START, "entry_node")
    graph_pipeline.add_edge("entry_node", END)

    # 4.编译图
    return graph_pipeline.compile()


# 实例化导入流程图
kb_import_graph_app = create_import_graph()

def run_import_graph(import_file_path: str, file_dir: str) -> ImportGraphState | None :
    """
        便捷函数：运行导入流程
        :param import_file_path:
        :param file_dir:
        :return: 最终状态字典
        """
    state = {
        "import_file_path": import_file_path,
        "file_dir": file_dir
    }
    # ** state 是 Python 的解包操作，将字典展开为关键字参数
    init_state = create_default_state(**state)
    # 默认返回None
    final_state = None

    for event in kb_import_graph_app.stream(init_state):
        for node_name, node_state in event.items():
            final_state = init_state

    return final_state

if __name__ == "__main__":

    setup_logging()

    import_file_path1 = r"D:\Python\Project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\基于STM32智能门禁系统_简洁报告.pdf"
    file_dir1 = r"D:\Python\Project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\result_doc"

    final_state1 = run_import_graph(import_file_path1,file_dir1)
    print(json.dumps(final_state1,indent=4,ensure_ascii=False))
    kb_import_graph_app.get_graph().print_ascii()









