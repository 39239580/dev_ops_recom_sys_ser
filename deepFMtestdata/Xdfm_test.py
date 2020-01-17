from XDeepFM_DeepFM_test.DFM_XDFM_data_prd import fn_test2
from code_tools.FM_util.XDeepFM import XDeepFM
import tensorflow as tf


def _run_base_model_xdfm(data_in, save_dict, xdfm_params_):
    tf.compat.v1.reset_default_graph()  # 重置计算图
    xdeepfm = XDeepFM(**xdfm_params_)
    xdeepfm.train(data=data_in, save_params=save_dict, train_test_split=True, early_stop=True,
                  autoevaluate=True)
    # xdeepfm.predict(data_in, False)


save_dicts = {"model_name": "model.ckpt", "model_save_path": "./xdeepfmmodel/"}
data, feat_len, field_len = fn_test2()
print(data.num_examples)
# xi,xv,label = data.next_batch(3)
# print(xi)
# print(type(xi))
# print(xv)
# print(type(xv))
# print(label)
# print(type(label))

xdfm_param = {"feature_size": feat_len,
              "field_size": field_len,
              "embedding_size": 8,
              "dropout_cin": [1.0, 0.5],  # cin 第一层的链接部分
              "deep_layers": [32, 64, 32],  # DNN部分的 节点数
              "dropout_deep": [0.5, 0.5, 0.5, 0.7],  # DNN部分的 dropout层  要比deep_layer 长度多1
              "cin_param": [12, 12],  # cin 部件中每个部件 的行数
              "deep_layers_activattion": "relu",
              "batch_size": 64,
              "learning_rate": 0.001,
              "optimizer_type": "adam",
              "batch_norms": True,  # 采用BN 层进行操作
              "bact_norm_decay": 0.995,
              "random_seed": 22,
              "use_cin": True,
              "use_deep": True,
              "loss_type": "logloss",
              "direct": False,  # cin 输出是否只用一半
              "greater_is_better": True,
              "ifregularizer": True,  # 开启正则化
              "movingaverage": True,  # 滑动平均关闭
              "exponentialdecay": True,  # 指数衰减关闭
              "traning_step": 10000,  # 总共的训练代数
              "reduce_d": False,
              "cin_activation": "sigmoid",
              "moving_average_decay": 0.99,  # 滑动平均系数
              "learning_rate_decay": 0.99,  # 指数衰减系数
              "regularaztion_rate": 0.0001,
              "every_epoch_train": False,
              "num_example": data.num_examples,
              "checkpoint_training": True,  # 自动加入继训练功能
              "model_save_path": save_dicts["model_save_path"],
              "evaluate_type": "all",
              "verbose": True}

_run_base_model_xdfm(data_in=data, save_dict=save_dicts, xdfm_params_=xdfm_param)
