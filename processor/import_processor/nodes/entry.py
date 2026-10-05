"""
入口节点的实现
"""
from pathlib import Path
import json
from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.exceptions import ValidationError
from knowledge.processor.import_processor.state import ImportGraphState


class EntryNood(BaseNode):
    """
    入口节点的实现逻辑：
    根据输入的文件扩展名设置对应的处理标志 交给下一节点处理
    决定后续流程走PDF转换分支，还是直接处理MD分支
    """
    name = "entry"

    def process(self, state:ImportGraphState) -> ImportGraphState:

        # 1.校验输入路径（把关）
        # ①获取文件的路径以及文件所在目录
        self.log_step("step1", "[获取文件路径]")
        file_dir = state.get("file_dir")
        import_file_path = state.get("import_file_path")

        # ②检查文件路径是否为空
        self.log_step("step2", "[检查文件路径]")
        if not file_dir or not import_file_path:
            raise ValidationError("文件目录或文件不存在", self.name)

        # 2.检查文件类型
        #使用标准的Path对象操作文件逻辑
        path = Path(import_file_path)
        #检查后缀
        suffix = path.suffix.lower()

        # 3.设置路由标志 (核心)，条件边前提
        #负责判断「是哪种文件」并把结果写进状态，自己不转换文件。
        if suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = import_file_path
        elif suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path
        else:
            self.logger.debug(f"文件类型{suffix}不支持")
            raise ValidationError(f"不支持的文件类型:{suffix}",self.name)

        # 4.提取文件标题
        file_title = path.stem  #文件.md -> 文件
        state["file_title"] = file_title

        #返回状态
        return state

if __name__ == "__main__":
    setup_logging()
    entry_state:ImportGraphState = {
        "import_file_path":r"D:\Python\Project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\基于STM32智能门禁系统_简洁报告.pdf",
        "file_dir":r"D:\Python\Project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\result_doc"
    }
    #实例化入口节点 类对象
    entry = EntryNood()
    process_state = entry.process(entry_state)
    print(json.dumps(process_state, indent=4, ensure_ascii=False))



