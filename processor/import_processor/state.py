"""
导入流程的 状态 定义
定义完整的状态结构和辅助函数
"""
import copy
from typing import TypedDict


class ImportGraphState(TypedDict, total=False): #total=False 代表 "这个字典类型的所有字段都是可选的"，允许状态在各节点间以不完整的形式传递。
    """
    导入流程图 状态
    包含整个 导入流程中传递的所有数据
    """

    # 1.任务标识
    task_id: str    #任务ID ，用于任务追踪（web交互时 实时看到节点的处理日志）

    # 2.控制标志
    is_md_read_enabled: bool    #是否启用 MD 读取
    is_pdf_read_enabled: bool   #是否启用 PDF 读取（其中包哦含了pdf转md）

    # 3.路径信息
    import_file_path: str       #导入文件路径

    file_dir: str               #导入（出）文件 目录，pdf转换为md存放的目录

    pdf_path: str               #pdf 文件路径

    md_path: str                #转换后的 .md文件路径

    # 4.文件信息
    file_title: str             #文件标题

    item_name: str              #识别出的产品/商品名称

    # 5.处理中间数据
    md_content: str             #md 文档内容

    chunks: list                #文档切片 列表

#===================================================
# 设置默认状态
GRAPH_DEFAULT_STATE: ImportGraphState = {
    "task_id": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "import_file_path": "",
    "file_dir": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "item_name": "",
    "md_content": "",
    "chunks": []
}

#===================================================
# 工具
def create_default_state(**overrides) -> dict:
    """
    创建默认状态，支持覆盖
    Args:
        **overrides: 要覆盖的字段
    Return:
        新状态实例
    Examples:
        state = create_default_state(task_id="task_001", local_file_path="doc.pdf")
    """
    state = copy.deepcopy(GRAPH_DEFAULT_STATE)  #深拷贝，生成了一个新对象
    state.update(overrides)
    return state

def get_default_state():
    """
    获取默认状态副本
    Return:
        状态副本（避免全局污染）
    """
    return copy.deepcopy(GRAPH_DEFAULT_STATE)
