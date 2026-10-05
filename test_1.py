
from pathlib import Path

import_path = r"D:\Python\Project\shopkeeper_brain\knowledge\processor\import_processor\temp_dir\基于STM32智能门禁系统_简洁报告.pdf"

import_path_obj = Path(import_path)

print(import_path_obj)
print(import_path_obj.parent, type(import_path_obj.parent))
print(Path(import_path_obj.parent), type(import_path_obj.parent))


