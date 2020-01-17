import featuretools as ft
import pandas as pd
import os
from tqdm import tqdm


# 创建用户样本
def make_user_sample(orders, order_products, departments, products, user_ids, out_dir):
    orders_sample = orders[orders["user_id"].isin(user_ids)]  # 在订单df 中筛选指定uid的数据， 列斯excel 中的筛选功能
    orders_keep = orders_sample["order_id"].values  # 获取数组组成的订单id [id1 id2 id3]

    order_products_sample = order_products[order_products["order_id"].isin(orders_keep)]
    # 再到 order_products 表中筛选对应的额订单数据

    try:  # 创建输出路径
        os.mkdir(out_dir)

    except:
        pass
    # 将相关数据存储起来，不需要index
    order_products_sample.to_csv(os.path.join(out_dir, "order_products_prior.csv"), index=None)
    orders_sample.to_csv(os.path.join(out_dir, "orders.csv"), index=None)
    departments.to_csv(os.path.join(out_dir, "departments.csv"), index=None)
    products.to_csv(os.path.join(out_dir, "products.csv"), index=None)


def load_data():
    data_dir = "data"
    prior_df = pd.read_csv(os.path.join(data_dir, "order_products__prior.csv"))
    train_df = pd.read_csv(os.path.join(data_dir, "order_products__train.csv"))
    order_products = pd.concat([prior_df, train_df])   # 订单产品 df
    orders = pd.read_csv(os.path.join(data_dir, "orders.csv"))   # 订单df
    departments = pd.read_csv(os.path.join(data_dir, "departments.csv"))  #
    products = pd.read_csv(os.path.join(data_dir, "products.csv"))  # 产品df

    users_unique = orders["user_id"].unique()   # 得出列的唯一值， 获取有订单的用户id,
    chunksize = 10000  # 10000条数据
    part_num = 0
    partition_dir = "partition_dir"

    try:
        os.mkdir(path=partition_dir)  # 创建路径
    except:
        pass

    print("总共的用户数据有:%d" % len(users_unique))
    for i in tqdm(range(0, len(users_unique), chunksize)):  # 0到用户数中， 步长为10000 采样为10000

        print("当前用户:%d" % i)
        users_keep = users_unique[i:i+chunksize]  # 采样的用户数,每次10000个用户数据
        make_user_sample(orders, order_products, departments, products, users_keep,
                         os.path.join(partition_dir, "part_%d" % part_num))
        part_num += 1


load_data()
