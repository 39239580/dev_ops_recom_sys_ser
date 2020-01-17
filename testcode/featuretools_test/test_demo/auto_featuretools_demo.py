import pandas as pd
from feature_engingneer.featuretools_tools.autofeature import AutoFeaturetools
import os
import featuretools as ft
import time


def load_data(verbose=False):
    data_dir = "./data"

    client = pd.read_csv(os.path.join(data_dir, "clients.csv"), parse_dates=["joined"])
    loans = pd.read_csv(os.path.join(data_dir, "loans.csv"), parse_dates=["loan_start", "loan_end"])
    payments = pd.read_csv(os.path.join(data_dir, "payments.csv"), parse_dates=["payment_date"])
    if verbose:
        print(client.head())  # 客户表，　用户基本信息，　每行代表一个客户信息
        time.sleep(1)
        print(loans.head())  # 向用户提供的贷款，每行为一项贷款，每个用户可能有多行贷款　
        time.sleep(1)
        print(payments.head())  # 还款记录， 每行为一次还款记录，每笔贷款可能有多行还款记录

    """ 
    client表的列名  client_id     joined  income  credit_score
    loans表的列名 client_id loan_type  loan_amount  repaid  loan_id  loan_start   loan_end rate
    payments表的列名 loan_id  payment_amount payment_date  missed
    """
    return client, loans, payments
# 目的是预测用户未来是否偿还一项贷款
# 做法，将关于用户的信息全整合到一张表中


def create_entity(client_df, loand_df,
                  payments_df,verbose=False):
    # 创建空实体
    es = AutoFeaturetools(entityname="clients")
    # 向空实体中添加数据,相当于每张表添加为一个实体
    # "添加实体client, loans"
    es.add_entity(entity_id="clients",
                  df=client_df,
                  index="client_id",  # 唯一元素的键
                  time_index="joined")

    es.add_entity(entity_id="loans",
                  df=loand_df,
                  variable_types={"repaid": ft.variable_types.Categorical},  # 指定数据类型，不指定则进行自动识别
                  index="loan_id", time_index="loan_start")
    #
    es.add_entity(entity_id="payments",
                  df=payments_df,
                  variable_types={"missed": ft.variable_types.Categorical},
                  index="payment_id",
                  make_index=True,  # 创建索引列
                  time_index="payment_date"
                  )
    if verbose:
        es.view_entity()  # 打印实体

    return es


def add_relationship(es, verbose):
    # 通过 共同键"clients"进行实体clients与loans关联
    r_client_previous = es.create_relationship(entity_id_a="clients", indexa="client_id",
                                               entity_id_b="loans", indexb="client_id",)
    es.add_relationship_entity(r_client_previous)

    #  # 通过 共同键"loan_id"进行实体loans与payments关联
    r_payments = es.create_relationship(entity_id_a="loans", indexa="loan_id",
                                        entity_id_b="payments", indexb="loan_id")
    es.add_relationship_entity(r_payments)

    if verbose:  # 打印实体集
        es.view_entity()

    return es


# 深度融合,使用默认的生成新的特征
def deep_merge(es, verbose=False):
    features, feature_names = es.get_feature(target_entity="clients")

    if verbose:
        print(features.head())
        print(features.shape)
    return features, feature_names


# 指定某些参数进行聚合
def deep_merge_transform(es, verbose=False):
    features, feature_names = es.get_feature(target_entity="clients",
                                             agg_primitives=["mean", "max", "percent_true", "last"],
                                             trans_primitives=["year", "month"])
    if verbose:
        print(features.head())
        print(features.shape)
        print(feature_names)
        print(type(feature_names))
    return features, feature_names


def encode(es, features, feature_names, verbose=False):
    fm_encode, f_encode = es.feature_encoder(feature_matrix=features, features=feature_names)

    if verbose:
        print(fm_encode.head(2))
        print(f_encode)
    return fm_encode, f_encode


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    client_df, loand_df, payments_df = load_data(True)
    es = create_entity(client_df, loand_df, payments_df, True)
    add_relationship(es, True)
    deep_merge(es, True)
    time.sleep(3)
    feature, feature_names = deep_merge_transform(es, True)  # 指定进行的操作的操作
    encode(es, feature, feature_names, True)
