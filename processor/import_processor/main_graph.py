"""
导入流程主图
使用LangGraph构建文档导入流程
"""
import json

from langgraph.constants import START, END
from langgraph.graph.state import CompiledStateGraph, StateGraph

from knowledge.processor.import_processor.base import setup_logging
from knowledge.processor.import_processor.nodes.bge_embedding import BgeEmbeddingChunksNode
from knowledge.processor.import_processor.nodes.document_split import DocumentSplitNode
from knowledge.processor.import_processor.nodes.entry import EntryNood
from knowledge.processor.import_processor.nodes.md_img import MarkDownImageNode
from knowledge.processor.import_processor.nodes.import_milvus import ImportMilvusNode
from knowledge.processor.import_processor.nodes.item_name_recognition import ItemNameRecognitionNode
from knowledge.processor.import_processor.nodes.pdf_to_md import PdfToMdNode
from knowledge.processor.import_processor.state import ImportGraphState, create_default_state



# 定义入口节点的 条件路由
def entry_router(state: ImportGraphState) -> str:
    """
    入口节点后的条件路由
    根据文件类型判断是进入pdf转md节点，还是直接进入md处理分支
    :param state: 当前图状态
    :return 下一个节点的名称
    """
    if state.get("is_pdf_read_enabled"):
        return "pdf_to_md_node"
    if state.get("is_md_read_enabled"):
        return "md_img_node"
    return END



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
    nodes: dict = {
        "entry_node": EntryNood(),                                  # 入口，识别pdf或md文档给下一节点处理
        "pdf_to_md_node": PdfToMdNode(),                            # pdf 转 md 文档节点
        "md_img_node": MarkDownImageNode(),                         # md 文档处理
        "document_split_node": DocumentSplitNode(),                 # 文档切片
        "item_name_recognition_node": ItemNameRecognitionNode(),    # 商品名识别
        "bge_embedding_chunks_node": BgeEmbeddingChunksNode(),      # 向量化切片
        "import_milvus_node": ImportMilvusNode()                    # 把切片存入milvus
    }

    # 2.1添加入口节点
    graph_pipeline.set_entry_point("entry_node")

    # 2.2添加所有节点到图中
    for key, value in nodes.items():
        graph_pipeline.add_node(key, value)

    # 3.定义边
    # 3.1条件边
    graph_pipeline.add_conditional_edges(
        "entry_node", entry_router,
        {
            "pdf_to_md_node": "pdf_to_md_node",
            "md_img_node": "md_img_node",
            END: END
        }
    )

    # 3.2 顺序边
    graph_pipeline.add_edge("pdf_to_md_node", "md_img_node")
    graph_pipeline.add_edge("md_img_node", "document_split_node")
    graph_pipeline.add_edge("document_split_node", "item_name_recognition_node")
    graph_pipeline.add_edge("item_name_recognition_node", "bge_embedding_chunks_node")
    graph_pipeline.add_edge("bge_embedding_chunks_node", "import_milvus_node")
    graph_pipeline.add_edge("import_milvus_node", END)

    # 4.编译图(返回可运行的状态)
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
    state:ImportGraphState = {
        "import_file_path": import_file_path,
        "file_dir": file_dir
    }
    # ** state 是 Python 的解包操作，将字典展开为关键字参数
    init_state = create_default_state(**state)

    final_state = kb_import_graph_app.invoke(init_state)
    return final_state

    # final_state = None
    # for event in kb_import_graph_app.stream(init_state):
    #     for node_name, node_state in event.items():
    #         print(f"\n===== 节点[{node_name}]执行完成,状态字段: {list(node_state.keys())} =====")
    #         # 想看节点输出全量内容时取消注释(注意 chunks 含 1024 维向量,打印会很长)
    #         # print(json.dumps(node_state, indent=4, ensure_ascii=False, default=str))
    #         final_state = node_state
    #
    # return final_state

#  ==  测试  ==
if __name__ == "__main__":

    def _brief_state(value):
        """
        递归精简状态字典:长字段摘要输出,避免控制台刷屏
        - dense_vector → <1024维稠密向量,前3值:...>
        - sparse_vector → <n个非零token>
        - 超长字符串(>100字符) → 截断并标注总长度
        - 其余字段原样输出
        """
        if isinstance(value, dict):
            brief = {}
            for key, v in value.items():
                if key == "dense_vector" and isinstance(v, list):
                    brief[key] = f"<{len(v)}维稠密向量,前3值:{[round(x, 4) for x in v[:3]]}>"
                elif key == "sparse_vector" and isinstance(v, dict):
                    brief[key] = f"<{len(v)}个非零token的稀疏向量>"
                elif isinstance(v, (dict, list)):
                    brief[key] = _brief_state(v)  # 递归处理 chunks 等嵌套结构
                elif isinstance(v, str) and len(v) > 100:
                    brief[key] = f"{v[:100]}...(总长度:{len(v)})"
                else:
                    brief[key] = v
            return brief
        if isinstance(value, list):
            return [_brief_state(item) if isinstance(item, (dict, list)) else item for item in value]
        return value

    setup_logging()

    import_file_path1 = r"D:\Python\Project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\华为擎云 L540 用户指南-(KLVV,UOS&KOS_01,zh-cn).pdf"
    file_dir1 = r"D:\Python\Project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir"

    final_state1 = run_import_graph(import_file_path1,file_dir1)

    #打印最终状态(向量等长字段已摘要)及图
    print(json.dumps(_brief_state(final_state1),indent=4,ensure_ascii=False))
    kb_import_graph_app.get_graph().print_ascii()









