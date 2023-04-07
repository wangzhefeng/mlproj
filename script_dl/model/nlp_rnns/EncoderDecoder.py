# -*- coding: utf-8 -*-


# ***************************************************
# * File        : encoder_decoder.py
# * Author      : Zhefeng Wang
# * Email       : wangzhefengr@163.com
# * Date        : 2023-03-27
# * Version     : 0.1.032717
# * Description : description
# * Link        : link
# * Requirement : 相关模块版本需求(例如: numpy >= 2.1.0)
# ***************************************************


# python libraries
import os
import sys

import torch
from torch import nn


# global variable
LOGGING_LABEL = __file__.split('/')[-1][:-3]


class Encoder(nn.Module):
    """
    编码器-解码器架构的基本编码器接口
    """
    def __init__(self, **kwargs) -> None:
        super(Encoder, self).__init__(**kwargs)
    
    def forward(self, X, **args):
        raise NotImplementedError


class Decoder(nn.Module):
    """
    编码器-解码器架构的基本解码器接口
    """

    def __init__(self, **kwargs) -> None:
        super(Decoder, self).__init__(**kwargs)
        
    def init__state(self, enc_outputs, *args):
        raise NotImplementedError
    
    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    """
    编码器-解码器架构的基类
    """
    def __init__(self, encoder, decoder, **kwargs) -> None:
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)





# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
