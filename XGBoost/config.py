feature_columns = ['dur', 'service','spkts',
                       'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl',
                       'dttl','dload', 'dloss','dinpkt','djit',
                       'dtcpb', 'dwin', 'tcprtt',
                       'ackdat','dmean', 'trans_depth', 'response_body_len',
                   ]
label_column = 'label' # 标签列名
input_size=len(feature_columns)
DatasetForTraining='../dataset/UNSW_NB15_training-set.csv'
DatasetForTesting='../dataset/UNSW_NB15_testing-set.csv'