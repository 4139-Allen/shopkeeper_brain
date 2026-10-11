import os


# 当前所在文件目录
# D:\Python\Project\shopkeeper_brain\knowledge\core
CURRENT_FILE_DIR = os.path.dirname(__file__)
print(CURRENT_FILE_DIR)

# 项目根目录
# D:\Python\Project\shopkeeper_brain\knowledge
# `os.path.dirname(__file__)` → 当前文件所在文件夹
# `".."` → 向上返回上一级目录
KNOWLEDGE_ROOT = os.path.abspath(os.path.join(CURRENT_FILE_DIR, ".."))
print(KNOWLEDGE_ROOT)

# 本地文件存储基础目录
# D:\Python\Project\shopkeeper_brain\knowledge\temp_data
LOCAL_BASE_DIR = os.path.join(KNOWLEDGE_ROOT, "temp_data")
print(LOCAL_BASE_DIR)

# 前端页面静态资源目录
# D:\Python\Project\shopkeeper_brain\knowledge\front
FRONT_PAGE_STATIC_DIR = os.path.join(KNOWLEDGE_ROOT, "front")
print(FRONT_PAGE_STATIC_DIR)

def get_local_base_dir():
    """"获取本地文件存储目录"""
    return LOCAL_BASE_DIR

def get_front_page_static_dir():
    """获取前端静态资源目录"""
    return FRONT_PAGE_STATIC_DIR


