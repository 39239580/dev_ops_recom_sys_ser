import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
# import numpy as np
from deepFMtestdata.deepfm_test import _load_data
from code_tools.FM_util.XdeepFM_batch_next import BacthDataset, scale_data

# from scipy.sparse import


class FeatrureDict(object):  # 生成转换字典
    def __init__(self, trainfile=None, testfile=None, validfile=None,
                 traindf=None, testdf=None, validdf=None, index_col=None, isvalid=False,
                 low=True,default=False,
                 ignore_cols=[], numeric_cols=[], single_cols=[], mult_cols=[], minmax=False,
                 single_feat_frequency=dict(), mult_feat_frequency=dict()):

        assert not ((trainfile is None) and (traindf is None))
        # assert not ((trainfile is not None) and (traindf is not None))
        assert not ((testfile is None) and (testdf is None))
        # assert not ((testfile is not None) and (testdf is not None))
        assert not ((isvalid is True) and (validfile is None) and(validdf is None))
        """
        提供加载 df对象与读取csv 文件两种方式
        """
        self.trainfile = trainfile
        self.testfile = testfile
        self.validfile = validfile
        self.traindf = traindf
        self.testdf = testdf
        self.validdf = validdf
        self.ignore_cols = ignore_cols  # 需要忽略的列
        self.numeric_cols = numeric_cols  # 数值型特征
        self.single_cols = single_cols  #
        self.mult_cols = mult_cols
        self.scalar = MinMaxScaler()  # 实例化  数值转成0到1 的小数
        self.feat_dict ={}
        self.backup_dict ={}  # 用于存放不符合要求的统一增加序号
        self.low = low
        self.index_col = index_col
        self.isvalid = isvalid
        self.default = default
        self.single_feat_frequency = single_feat_frequency
        self.mult_feat_frequency = mult_feat_frequency
        self.df, self.traindf, self.testdf, self.validdf = self.feat_dict_prd()
        # print(self.df)
        # print()
        if minmax:  # 归一化操作
            self.minmax_prd()
            self.check_itself()
        self.feat_dim = self.gen_feat_dict()

    def feat_dict_prd(self):  # 特征字典预处理
        print("load_df....")
        if self.traindf is None:
            dftrain = pd.read_csv(self.trainfile, index_col=self.index_col)
        else:
            dftrain = self.traindf

        if self.testdf is None:
            dftest = pd.read_csv(self.testfile, index_col=self.index_col)
        else:
            dftest = self.testdf

        if self.validdf is None and self.validfile is None:  # 两者都为空
            dfvalid = None
        else:
            if self.validdf is None:
                dfvalid = pd.read_csv(self.validfile, index_col=self.index_col)
            else:
                dfvalid = self.validdf

        if self.isvalid:
            df = pd.concat([dftrain, dftest, dftest])
        else:
            df = pd.concat([dftrain, dftest])
        return df, dftrain, dftest, dfvalid

    def minmax_prd(self):
        if self.validdf:
            for cols in self.numeric_cols:
                self.scalar.fit(self.df[cols].values.reshape(-1, 1))  # 获取每列的值
                self.traindf[cols] = self.scalar.transform(self.traindf[cols].values.reshape(-1, 1))  # 按列进行归一化操作
                self.testdf[cols] = self.scalar.transform(self.testdf[cols].values.reshape(-1, 1))
                self.validdf[cols] = self.scalar.transform(self.validdf[cols].values.reshape(-1, 1))
        else:
            for cols in self.numeric_cols:
                self.scalar.fit(self.df[cols].values.reshape(-1, 1))  # 获取每列的值
                self.traindf[cols] = self.scalar.transform(self.traindf[cols].values.reshape(-1, 1))  # 按列进行归一化操作
                self.testdf[cols] = self.scalar.transform(self.testdf[cols].values.reshape(-1, 1))

    # 检查列数必须相等
    def check_itself(self):
        if self.traindf.shape[1] == self.testdf.shape[1] == self.validdf.shape[1]:
            return True
        else:
            raise ValueError("error, all dataset must have same shape")

    def gen_feat_dict(self):
        print("pepare dict...")
        tc = 0
        for col in self.df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:  # 需要保留的连续数值型列
                self.feat_dict[col] = tc
                tc += 1
                if self.default:
                    self.backup_dict[col] = tc
                    tc += 1

            elif col in self.single_cols:  # 单值离散型
                us = self.df[col].unique()
                if self.single_feat_frequency.get(col, 0) > 0:
                    single_count_dict = dict(self.df[col].value_counts())
                    nan_list, us = self.auto_select_idx(single_count_dict, self.single_feat_frequency[col])
                    if self.default:
                        self.backup_dict[col] = tc
                        tc += 1
                self.feat_dict[col] = dict(zip(us, range(tc, len(us)+tc)))
                tc += len(us)
            elif col in self.mult_cols:  # 多值离散型数据
                mult_temp = list()
                for i in self.df[col].map(lambda x: x.split("|")).tolist():
                    mult_temp += i
                mult_len = len(set(mult_temp))
                if self.mult_feat_frequency.get(col, 0) > 0:
                    mult_count_dict = Counter(mult_temp)
                    nan_mul_list, mult_temp = self.auto_select_idx(mult_count_dict, self.mult_feat_frequency[col])
                    mult_len = len(nan_mul_list)
                    if self.default:
                        self.backup_dict[col] = tc
                        tc += 1
                self.feat_dict[col] = dict(zip(set(mult_temp), range(tc, mult_len+tc)))
                tc += mult_len
            else:  #
                us = self.df[col].unique()   # 每一列取不要重复的值
                self.feat_dict[col] = dict(zip(us, range(tc, len(us)+tc)))  # 组成字典,   为特征编号
                tc += len(us)
        return tc

    def auto_select_idx(self, feat_feat, threshold):
        if self.low:
            nan_list = [k for k, v in feat_feat.items() if int(v) < threshold]
            feat_list = [k for k, v in feat_feat.items() if int(v) >= threshold]
        else:
            nan_list = [k for k, v in feat_feat.items() if int(v) > threshold]
            feat_list = [k for k, v in feat_feat.items() if int(v) <= threshold]
        return nan_list, feat_list


class DataParser(object):
    def __init__(self, orginal_format="csv", targert_format="deepfm",
                 feat_dict=None, backup_dict=None, drop_field=None):
        self.feat_dict = feat_dict
        self.backup_dict = backup_dict
        self.orginal_format = orginal_format
        self.targert_format = targert_format
        self.drop_field = drop_field

    def parse(self, input_file=None, input_df=None,index_col=None):
        assert not ((input_file is None) and (input_df is None))
        assert not ((input_file is not None) and (input_df is not None))
        if input_file is None:
            dfi = input_df.copy()
        else:
            dfi = pd.read_csv(input_file, index_col=index_col)

        if self.drop_field[-1] in dfi.columns.values.tolist():
            has_label = True
        else:
            has_label = False

        if has_label:
            y = dfi[self.drop_field[-1]].values.tolist()
            ids = None
            dfi.drop(self.drop_field, axis=1, inplace=True)
        else:
            ids = dfi["id"].values.tolist()
            y = None
            dfi.drop(self.drop_field[0], axis=1, inplace=True)

        dfv = dfi.copy()

        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue

            if col in self.feat_dict.numeric_cols:
                dfi[col] = self.feat_dict.feat_dict[col]

            else:
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])  # 查找对应的序列的值
                dfv[col] = 1  # 输入为1

        xi = dfi.values.tolist()  # 组成列表
        xv = dfv.values.tolist()
        if has_label:
            return xi, xv, y
        else:
            return xi, xv, ids


def fn_test():
    # --------生成索引序号字典的代码----------------- #
    trainpath = 'F:/kanshancup/def/XDeepFM_DeepFM_test/data/dataset1.csv'
    validpath = 'F:/kanshancup/def/XDeepFM_DeepFM_test/data/dataset2.csv'
    testpath = 'F:/kanshancup/def/XDeepFM_DeepFM_test/data/dataset3.csv'
    numeric_colss = ['all_launch_count', 'last_launch', 'all_video_count', 'last_video', 'all_video_day',
                     'all_action_count', 'last_action',
                     'all_action_day', 'register_day']  # 数值特征9个
    single_colss = ['register_type', 'device_type']  # 单值特征2个  设备类型，注册类型
    mult_colss = ["multi"]  # ["multi"]  # 多值型single_features = ['register_type', 'device_type']
    #  单值特征2个  设备类型，注册类型
    ignore_colss =["user_id", "label"]
    single_feat_frequencys = {"device_type": 10}
    mult_feat_frequencys = {"multi": 0}

    print("ok")
    dicts = FeatrureDict(trainfile=trainpath, testfile=testpath, validfile=validpath, index_col=0,
                         ignore_cols=ignore_colss,
                         numeric_cols=numeric_colss, single_cols=single_colss, mult_cols=mult_colss,
                         single_feat_frequency=single_feat_frequencys, mult_feat_frequency=mult_feat_frequencys)
    print(dicts.feat_dim)
    print(dicts.feat_dict)
    print(len(dicts.feat_dict["device_type"]))
    print(len(dicts.feat_dict))
    print(dicts.feat_dict["register_type"])
    print(dicts.backup_dict)


def fn_test2():
    numeric_colss =[
        "ps_reg_01", "ps_reg_02", "ps_reg_03",
        "ps_car_12", "ps_car_13", "ps_car_14","ps_car_15"
        # 特征工程
        "missing_feat", "ps_car_13_x_ps_reg_03"]
    single_colss = ['register_type', 'device_type']  # 单值特征2个  设备类型，注册类型
    #  单值特征2个  设备类型，注册类型
    ignore_colss = ["id", "targe",
                    "ps_calc_01",  "ps_calc_02",  "ps_calc_03",  "ps_calc_04",
                    "ps_calc_05",  "ps_calc_06",  "ps_calc_07",  "ps_calc_08",
                    "ps_calc_09",  "ps_calc_10",  "ps_calc_11",  "ps_calc_12",
                    "ps_calc_13",  "ps_calc_14",
                    "ps_calc_15_bin",  "ps_calc_16_bin",  "ps_calc_17_bin",
                    "ps_calc_18_bin",  "ps_calc_19_bin",  "ps_calc_20_bin"]
    dftrain, dftest, x_train, y_train, x_test, ids_test, cat_features_indices = _load_data()
    print("ok")
    dicts = FeatrureDict(traindf=dftrain, testdf=dftest, ignore_cols=ignore_colss,
                         numeric_cols=numeric_colss, single_cols=single_colss)
    print(dicts.feat_dim)
    print(dicts.feat_dict)
    print(dicts.feat_dict["target"])
    data_parser = DataParser(feat_dict=dicts, drop_field=["id", "target"])
    xi, xv, y = data_parser.parse(input_df=dicts.traindf)
    field_dim = len(xi[0])
    xi, xv, y = scale_data(xi=xi, xv=xv, label=y, ratio=0.1)
    data = BacthDataset(xi=xi, xv=xv, labels=y)
    return data, dicts.feat_dim, field_dim


if __name__ == "__main__":
    data, feat_len, field_dim = fn_test2()
    print(data.next_batch(20))
    print(data.next_batch(20))
    print(feat_len, field_dim)
