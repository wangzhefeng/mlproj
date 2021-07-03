

"""
1. 数据日志查看
"""


def data_logging(data, data_name):
    """
    查看数据结构、内容信息
    """
    print("=" * 20)
    print(f"{data_name}.head()")
    print("=" * 20)
    print(data.head())
    print("=" * 20)
    print(f"{data_name}.tail()")
    print("=" * 20)
    print(data.tail())
    print("=" * 20)
    print(f"{data_name}.info()")
    print("=" * 20)
    print(data.info())