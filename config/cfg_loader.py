# -*- coding: utf-8 -*-

# ***************************************************
# * File        : cfg_loader.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2024-09-18
# * Version     : 1.0.091823
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# * TODO        : 1.
# ***************************************************

__all__ = []

# python libraries
import os
import sys
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from typing import Dict
import yaml

# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


def load_yaml(file_name):
    with open(file_name, 'r', encoding = "utf-8") as infile:
        return yaml.load(
            infile, 
            Loader = yaml.FullLoader
        )


def get_params(yaml_path: str) -> Dict:
    """
    读取项目配置参数

    Returns:
        Dict: 项目配置参数
    """
    # 配置文件读取
    cfg_dir = os.path.dirname(__file__)
    # 项目配置 yaml 文件
    cfg_params = load_yaml(os.path.join(cfg_dir, yaml_path))

    return cfg_params




# 测试代码 main 函数
def main(): 
    sys_cfg_path = "./sys_cfg.yaml"
    cfg_params = get_params(sys_cfg_path)
    print(cfg_params)

if __name__ == "__main__":
    main()
