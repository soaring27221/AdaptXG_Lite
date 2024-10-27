'''feature_columns = ['dur', 'proto', 'service', 'state', 'spkts',
                       'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl',
                       'dttl', 'sload', 'dload', 'sloss', 'dloss',
                       'sinpkt', 'dinpkt', 'sjit', 'djit', 'swin',
                       'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack',
                       'ackdat', 'smean', 'dmean', 'trans_depth', 'response_body_len',
                       'ct_srv_src', 'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
                       'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm',
                       'ct_srv_dst', 'is_sm_ips_ports']  # 特征列名'''

feature_columns = ['dur','proto','service','spkts',
                       'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl',
                       'dttl','dload', 'dloss','dinpkt','djit',
                       'dtcpb', 'dwin', 'tcprtt',
                       'ackdat','dmean', 'trans_depth', 'response_body_len',
                   ]
label_column = 'label' # 标签列名
input_size=len(feature_columns)
DatasetForTraining='../dataset/UNSW_NB15_training-set.csv'
DatasetForTesting='../dataset/UNSW_NB15_testing-set.csv'