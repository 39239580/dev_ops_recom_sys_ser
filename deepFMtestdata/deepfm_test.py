import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from code_tools.FM_util.DeepFM import DeepFM

work_env = "work"
if work_env == "work":
    path = "F:/kanshancup/def/"
else:
    path = "J:/data_set_0926/program/"
train_file = path + "deepFMtestdata/data/train.csv"   # 数据集总计59列
test_file = path + "deepFMtestdata/data/test.csv"
sub_dir = "./output"
num_splits = 3
random_seed = 2017

# 数据集 df中的列名
# 种类列名
categorical_cols = [
    # 'ps_ind_02_cat', 'ps_ind_04_cat', 'ps_ind_05_cat',
    # 'ps_car_01_cat', 'ps_car_02_cat', 'ps_car_03_cat',
    # 'ps_car_04_cat', 'ps_car_05_cat', 'ps_car_06_cat',
    # 'ps_car_07_cat', 'ps_car_08_cat', 'ps_car_09_cat',
    # 'ps_car_10_cat', 'ps_car_11_cat',
]
# 连续性数值列
numeric_cols = [
    # # binary
    # "ps_ind_06_bin", "ps_ind_07_bin", "ps_ind_08_bin",
    # "ps_ind_09_bin", "ps_ind_10_bin", "ps_ind_11_bin",
    # "ps_ind_12_bin", "ps_ind_13_bin", "ps_ind_16_bin",
    # "ps_ind_17_bin", "ps_ind_18_bin",
    # "ps_calc_15_bin", "ps_calc_16_bin", "ps_calc_17_bin",
    # "ps_calc_18_bin", "ps_calc_19_bin", "ps_calc_20_bin",
    # numeric
    "ps_reg_01", "ps_reg_02", "ps_reg_03",
    "ps_car_12", "ps_car_13", "ps_car_14","ps_car_15"
    # 特征工程
    "missing_feat", "ps_car_13_x_ps_reg_03"
]

# 需要忽略的列
# 20列
ignore_cols = [
    "id", "targe",
    "ps_calc_01",  "ps_calc_02",  "ps_calc_03",  "ps_calc_04",
    "ps_calc_05",  "ps_calc_06",  "ps_calc_07",  "ps_calc_08",
    "ps_calc_09",  "ps_calc_10",  "ps_calc_11",  "ps_calc_12",
    "ps_calc_13",  "ps_calc_14",
    "ps_calc_15_bin",  "ps_calc_16_bin",  "ps_calc_17_bin",
    "ps_calc_18_bin",  "ps_calc_19_bin",  "ps_calc_20_bin"]


def gini(actual, pred):  # 实际的与预测的
    assert (len(actual) == len(pred))
    alls = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)  # np.c_ 按列将数组进行拼接
    alls = alls[np.lexsort((alls[:, 2], -1*alls[:, 1]))]   # 按照指定行排序
    totallosses = alls[:, 0].sum()  # 求和
    ginisum = alls[:, 0].cumsum().sum() / totallosses
    ginisum -= (len(actual)+1)/2.
    return ginisum/len(actual)


def gini_normal(actual, pred):
    return gini(actual, pred) / gini(actual, actual)


# 将输出的结果保存成csv 文件
def _make_submission(ids, y_pred, filename="submission.csv"):
    pd.DataFrame({"id": ids, "traget": y_pred.flatten()}).to_csv(
        os.path.join(sub_dir, filename), index=False, float_format="%.5f"
    )


# 进行绘图
def _plot_fig(train_result, valid_results, model_name):
    colors = ["red", "blue", "green"]
    xs = np.arange(1, train_result.shape[1]+1)
    plt.figure()
    lengends = []
    for i in range(train_result.shape[0]):  # 遍历每个离散的点
        plt.plot(xs, train_result[i], color=colors[i], linestyle="solid", marker="0")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="0")
        lengends.append("train-%d" % (i+1))
        lengends.append("valid-%d" % (i+1))
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Gini")
    plt.title("%s" % model_name)
    plt.legend(lengends)
    plt.savefig("./fig/%s.png" % model_name)
    plt.close()


class FeatureDict(object):
    def __init__(self, trainfile=None, testfile=None, dftrain_df=None,
                 dftest_df=None, numeric_cols_=None, ignore_cols_=None):
        assert not ((trainfile is None) and (dftrain_df is None))
        assert not ((trainfile is not None) and (dftrain_df is not None))
        assert not ((testfile is None) and (dftest_df is None))
        assert not ((testfile is not None) and (dftest_df is not None))
        """
        提供直接加载df 对象与读取csv文件两种方式
        """
        self.tranfile = trainfile
        self.testfile = testfile
        self.dftrain = dftrain_df
        self.dftest = dftest_df
        self.numeric_cols = numeric_cols_   # 数值特征
        self.ignore_cols = ignore_cols_    # 忽略的列
        self.feat_dict = {}  # 返回要使用的字典
        self.feat_dim = self.gen_feat_dict()  # 生成特征字典， 即总共的编号长度

    def gen_feat_dict(self):   # 生成特征字典
        if self.dftrain is None:
            dftrain_data = pd.read_csv(self.tranfile)
        else:
            dftrain_data = self.dftrain
        if self.dftest is None:
            dftest_data = pd.read_csv(self.testfile)
        else:
            dftest_data = self.dftest

        df = pd.concat([dftrain_data, dftest_data])  # 两种数据拼接起来

        tc = 0
        # 过滤掉  忽略的列
        for col in df.columns:   # 遍历每列数据
            if col in self.ignore_cols:   # 去掉忽略的列
                continue
            if col in self.numeric_cols:   # 保留需要的连续数值型的列
                self.feat_dict[col] = tc   # 安插 idx  序号
                tc += 1
            else:    # 其余部分
                us = df[col].unique()   # 每一列取不要重复的值
                self.feat_dict[col] = dict(zip(us, range(tc, len(us)+tc)))  # 组成字典,   为特征编号
                tc += len(us)
        return tc  # 特征号


class DataParser(object):  # 数据分离
    def __init__(self, feat_dict):
        self.feat_dict = feat_dict

    def parse(self, infile=None, df=None, has_label=False):
        assert not ((infile is None) and (df is None))
        assert not ((infile is not None) and (df is not None))
        if infile is None:
            dfi = df.copy()
        else:
            dfi = pd.read_csv(infile)

        if has_label:
            y = dfi["target"].values.tolist()
            ids = None
            dfi.drop(["id", "target"], axis=1, inplace=True)
        else:
            ids = dfi["id"].values.tolist()
            y = None
            dfi.drop(["id"], axis=1, inplace=True)

        dfv = dfi.copy()
        # print(dfi)
        # print([col for col in dfi.columns])
        for col in dfi.columns:  # 遍历每一列
            if col in self.feat_dict.ignore_cols:  # 去掉忽略的列
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            if col in self.feat_dict.numeric_cols:   # 连续型数值
                dfi[col] = self.feat_dict.feat_dict[col]
            # 其他类型数据
            else:
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])  # 查找对应的序列的值
                dfv[col] = 1  # 输入为1

        xi = dfi.values.tolist()  # 组成列表
        xv = dfv.values.tolist()
        if has_label:
            return xi, xv, y
        else:
            return xi, xv, ids


def _run_base_model_dfm(dftrain_in, dftest_in, fold, dfm_params_):

    fd = FeatureDict(dftrain_df=dftrain_in, dftest_df=dftest_in, numeric_cols_=numeric_cols, ignore_cols_=ignore_cols)
    print(fd.feat_dim)
    print(fd.feat_dict)
    data_parser = DataParser(feat_dict=fd)
    xi_train_base, xv_train_base, y_train_base = data_parser.parse(df=dftrain_in, has_label=True)
    xi_test_base, xv_test_base, ids_test_base = data_parser.parse(df=dftest_in)

    dfm_params_["feature_size"] = fd.feat_dim  # 序号总长度
    dfm_params_["field_size"] = len(xi_train_base[0])  # 获取 field个数的取值情况
    # print("长度为%d" % (len(xi_train_base[0])))  # 59-20=39列

    y_train_meta = np.zeros((dftrain.shape[0], 1), dtype=float)   # m*1 0矩阵   m为训练数据的长度
    y_test_meta = np.zeros((dftest.shape[0], 1), dtype=float)   # n*1 0矩阵   n为测试集的数据长度
    get_fn = lambda x, l: [x[j] for j in l]

    gini_results_cv = np.zeros(len(fold), dtype=float)  # 长度为　数据集长度　的0阶矩阵
    gini_results_epoch_train = np.zeros((len(fold), dfm_params_["epoch"]), dtype=float)
    gini_results_epoch_valid = np.zeros((len(fold), dfm_params_["epoch"]), dtype=float)
    dfm = DeepFM(**dfm_params_)  # 初始化模型
    for i, (train_idx, valid_idx) in enumerate(fold):  # 遍历每部分的数据长度
        # 批量获取数据
        xi_train_, xv_train_, y_train_ = get_fn(xi_train_base, train_idx), \
                                         get_fn(xv_train_base, train_idx), \
                                         get_fn(y_train_base, train_idx)
        xi_valid_, xv_valid_, y_valid_ = get_fn(xi_train_base, valid_idx), get_fn(xv_train_base, valid_idx), \
                                         get_fn(y_train_base, valid_idx)

        dfm.fit(xi_train_, xv_train_, y_train_, xi_valid_, xv_valid_, y_valid_)  # 训练

        y_train_meta[valid_idx, 0] = dfm.predict(xi_valid_, xv_valid_)  # 预测
        y_test_meta[:, 0] = dfm.predict(xi_test_base, xv_test_base)  # 预测

        gini_results_cv[i] = gini_normal(y_valid_, y_train_meta[valid_idx])
        gini_results_epoch_train[i] = dfm.train_result
        gini_results_epoch_valid[i] = dfm.valid_result

    y_test_meta /= float(len(fold))

    if dfm_params_["use_fm"] and dfm_params_["use_deep"]:
        clf_str = "DeepFM"
    elif dfm_params_["use_fm"]:
        clf_str = "FM"
    elif dfm_params_["use_deep"]:
        clf_str = "DNN"
    else:
        raise ValueError("no select learning model")

    print("%s: %.5f (%.5f)" % (clf_str, gini_results_cv.mean(), gini_results_cv.std()))
    filename = "%s_Mean%.5f_std%.5f.cv" % (clf_str, gini_results_cv.mean(), gini_results_cv.std())
    _make_submission(ids_test, y_test_meta, filename)
    _plot_fig(gini_results_epoch_train, gini_results_epoch_valid, clf_str)
    return y_train_meta, y_test_meta


def _load_data():
    dftrain_df = pd.read_csv(train_file)
    dftest_df = pd.read_csv(test_file)

    def preprocess(df):   # 定义函数
        cols_s = [c for c in df.columns if c not in ["id", "traget"]]  # 列名    获取除了这两个名字的列名
        df["missing_feat"] = np.sum((df[cols_s] == -1).values, axis=1)   # 统计 每行为-1的数量有多少个
        # print("rs")
        # print(df["missing_feat"])
        df["ps_car_13_x_ps_reg_03"] = df["ps_car_13"]*df["ps_reg_03"]  # 特征融合
        return df

    dftrain_df = preprocess(dftrain_df)
    dftest_df = preprocess(dftest_df)
    # 除掉  id ,target 与 忽略的列
    cols = [c for c in dftrain_df.columns if c not in ["id", "target"]]
    cols = [c for c in cols if (not c in ignore_cols)]

    x_trains = dftrain_df[cols].values   # 这是  列
    y_trains = dftrain_df["target"].values   # 这是 标签
    x_tests = dftest_df[cols].values  # 测试集
    ids_tests = dftest_df["id"].values  # 测试集的id
    cat_features_indicess = [i for i, c in enumerate(cols) if c in categorical_cols]   # 种类列的序号值

    return dftrain_df, dftest_df, x_trains, y_trains, x_tests, ids_tests, cat_features_indicess


# 加载数据， dftrain 数据， dftest数据， 全部的数据df, 包括新增的两列统计信息  missing_feat，ps_car_13_x_ps_reg_03
# x_train  x_test  过滤掉的忽略的列与  id, target 列的df 数据格式。 y_train , ids_test 为标签。
dftrain, dftest, x_train, y_train, x_test, ids_test, cat_features_indices = _load_data()
# cat_features_indices   种类列的序号

# 将数据 按了比例拆分与 混合，将训练数据拆成两部分， 训练与验证部分拆成7:3的比例
folds = list(StratifiedKFold(n_splits=num_splits, shuffle=True,
                             random_state=random_seed).split(x_train, y_train))   # 分拆数据的情况

# print(folds)
# raise
# deepFM Model
dfm_params = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 8,
    "dropout_fm": [1.0, 1.0],
    "deep_layers_activation": tf.nn.relu,
    "epoch": 30,
    "batch_size": 64,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "batch_norms": 1,
    "batch_norm_decay": 0.995,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": gini_normal,
    "random_seed": random_seed
}

# DeepFm 模型
# y_train_dfm, y_test_dfm = _run_base_model_dfm(dftrain, dftest, folds, dfm_params)

# FM 模型
fm_params = dfm_params.copy()
fm_params["use_deep"] = False
# y_train_fm, y_test_fm = _run_base_model_dfm(dftrain, dftest, folds, fm_params)

# DNN 模型
dnn_params = dfm_params.copy()
dnn_params["use_fm"] = False
# y_train_dnn, y_test_dnn = _run_base_model_dfm(dftrain, dftest, folds, dnn_params)
