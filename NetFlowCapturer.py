'''
这个文件是用来捕获网络流的
并且从网络流中提取需要的特征
'''
from scapy.all import *
import time
from scapy.layers.inet import IP

# 用于存储连接的开始时间
connections = {}

def packet_handler(pkt):
    # 检查数据包是否有IP层
    if IP in pkt:
        proto = pkt[IP].proto  # 协议类型
        src_ip = pkt[IP].src  # 源IP
        dst_ip = pkt[IP].dst  # 目的IP

        # 生成一个唯一的连接标识符
        connection_id = (src_ip, dst_ip, proto)

        # 记录连接的开始时间
        if connection_id not in connections:
            connections[connection_id] = time.time()

        # 计算连接的持续时间
        dur = time.time() - connections[connection_id]

        # 打印信息
        print(f"Connection_id:{connection_id},Proto: {proto}, Src IP: {src_ip}, Dst IP: {dst_ip}, Dur: {dur:.5f} seconds")


# 开始抓包
sniff(prn=packet_handler, filter="ip", store=0)  # 只捕获IP数据包