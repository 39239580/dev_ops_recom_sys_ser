from feature_engingneer.featuretools_tools.autofeature import AutoFeaturetools
import featuretools as ft
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


"""
相关的项目代码 https://github.com/Featuretools/predict-next-purchase/blob/master/Tutorial.ipynb
"""


def load_entityset(data_dir):
    order_products = pd.read_csv(os.path.join(data_dir, "order_products_prior.csv"))
    orders = pd.read_csv(os.path.join(data_dir, "orders.csv"))
    departments = pd.read_csv(os.path.join(data_dir, "departments.csv"))
    products = pd.read_csv(os.path.join(data_dir, "products.csv"))
    # print(order_products.head(2))
    # print(orders.head(2))
    # print(departments.head(2))
    # print(products.head(2))
    # print('--------------------')
    order_products = order_products.merge(products)   # dproduct_id 作为共键操作进行聚合
    # print(order_products.head(2))
    order_products = order_products.merge(departments)  # department_id  作为共键操作进行聚合
    # print(order_products.head(2))

    def add_time(df):
        df.reset_index(drop=True)  # 重置索引
        df["order_time"] = np.nan  # 添加一列
        days_since = df.columns.tolist().index("days_since_prior_order")  # 获取索引号
        hour_of_day = df.columns.tolist().index("order_hour_of_day")
        order_time = df.columns.tolist().index("order_time")

        df.iloc[0, order_time] = pd.Timestamp("Jan 1, 2015") + pd.Timedelta(df.iloc[0, hour_of_day], "h")
        # 创建的订单时间
        for i in range(1, df.shape[0]):  # 遍历每行数据，进行时间组装
            df.iloc[i, order_time] = df.iloc[i-1, order_time] +\
                pd.Timedelta(df.iloc[i, days_since], "d") + \
                pd.Timedelta(df.iloc[i, hour_of_day], "h")

        to_drop = ["order_number", "order_dow", "order_hour_of_day", "days_since_prior_order", "eval_set"]
        df.drop(to_drop, axis=1, inplace=True)
        return df

    orders = orders.groupby("user_id").apply(add_time)
    order_products = order_products.merge(orders[["order_id", "order_time"]])
    order_products["order_product_id"] = order_products["order_id"].astype(str) + "_" +\
        order_products["add_to_cart_order"].astype(str)
    order_products.drop(["product_id", "department_id", "add_to_cart_order"], axis=1, inplace=True)
    es = AutoFeaturetools(entityname="instacart")  # 实体集对象

    es.add_entity(entity_id="order_products",
                  df=order_products,
                  index="order_product_id",
                  variable_types={"aisle_id": ft.variable_types.Categorical,
                                  "reordered": ft.variable_types.Boolean},
                  time_index="order_time")

    es.add_entity(entity_id="orders",
                  df=orders,
                  index="order_id",
                  time_index="order_time")

    es.create_relationship(tablea="orders", indexa="id", tableb="users", indexb="user_id")
    es.split(base_entity_id="orders", new_entity_id="users", index="user_id")
    es.add_last_time_indexes()

    es.set_interesting_values(entity_id="order_products", columns="department",
                              values=["produce", "dairy eggs", "snacks", "beverages",
                                      "frozen", "pantry", "bakery", "canned goods",
                                      "deli", "dry goods pasta"])
    es.set_interesting_values(entity_id="order_products", columns="product_name",
                              values=['Banana', 'Bag of Organic Bananas', 'Organic Baby Spinach',
                                      'Organic Strawberries', 'Organic Hass Avocado',
                                      'Organic Avocado', 'Large Lemon', 'Limes',
                                      'Strawberries', 'Organic Whole Milk'])
    return es


def make_labels(esx, traning_window, cutoff_time, product_name, prediction_window):  # 制造标签
    prediction_window_end = cutoff_time+prediction_window  # 预测未来某个时间窗口内的数据
    t_start = cutoff_time - traning_window   # 截止时间- 训练窗口 = 开始时间？
    orders = esx.get_entity_df("orders")
    ops = esx.get_entity_df("order_products")

    traning_data = ops[(ops["order_time"] <= cutoff_time) & (ops["order_time"] > t_start)]
    prediction_data = ops[(ops["order_time"] > cutoff_time) & (ops["order_time"] < prediction_window_end)]

    users_in_training = traning_data.merge(orders)["user_id"].unique()
    valid_pred_data = prediction_data.merge(orders)
    valid_pred_data = valid_pred_data[valid_pred_data["user_id"].isin(users_in_training)]

    def bought_product(df):  # 买产品
        return (df["product_name"] == product_name).any()

    labels = valid_pred_data.groupby("user_id").apply(bought_product).reset_index()
    labels["cutoff_time"] = cutoff_time

    labels.columns = ["user_id", "label", "time"]
    labels = labels[["user_id", "time", "label"]]
    return labels


def dask_make_labels(esf, **kwargs):
    label_times = make_labels(esf, **kwargs)
    return label_times


def merge_features_labels(fm, labels):  # 聚合特征标签
    return fm.reset().merge(labels)


def cal_feature_martixs(esr, label_times, features):  # 计算特征矩阵
    label_times, ess = label_times
    fm = esr.cal_feature_martix(features=features, entities=ess, cutoff_time=label_times,
                                cutoff_time_in_index=True, verbose=False)
    x = merge_features_labels(fm, label_times)
    return x


def feature_importtance(model, feature, n=10):  # 特征重要性
    importance = model.feature_impotance_
    zipped = sorted(zip(feature, importance), key=lambda x: -x[1])
    for i, f in enumerate(zipped[:n]):
        print("%d:Feature:%s,%.3f" % (i+1, f[0].get_name(), f[1]))

    return [f[0] for f in zipped[:n]]


pd.set_option('display.max_columns', None)

# 加载数据
es = load_entityset("partition_dir/part_1/")
print(es)

# 可视化实体集
es.plot()

# make_labels
label_times = make_labels(es=es, product_name="Banana",
                          cutoff_time=pd.Timestamp("March 15, 2015"),
                          prediction_window=ft.Timedelta("4 weeks"),
                          traning_window=ft.Timedelta("60 days"))
print(label_times.head(5))

# 标签分布
label_times["label"].value_counts()

# 自动特征工程
feature_matrix, features = es.get_feature(target_entity="users",
                                          cutoff_time=label_times,
                                          training_window=ft.Timedelta("60 days"),
                                          entities=es,
                                          verbose=True)
fm_encode, f_encode = es.feature_encoder(feature_matrix=feature_matrix,
                                         features=features)
print("Number of featyres %s" % len(fm_encode))
print(fm_encode.head(10))

# 机器学习
X = merge_features_labels(fm_encode, label_times)
X.drop(["user_id", "time"], axis=1, inplace=True)
X = X.fillna(0)
y = X.pop("label")

# 随机森林的使用
clf=RandomForestClassifier(n_estimators=400, n_jobs=-1) # 树的个数
scores = cross_val_score(estimator=clf, X=X, y=y, cv=3, # cv 3折交叉验证
                         scoring="roc_auc",verbose=True)
print("AUC %.2f +/- %.2f"%(scores.mean(), scores.std()))

clf.fit(X,y)   #训练
top_feature = feature_importtance(clf,f_encode, n=20)
ft.save_features(top_feature,"top_features")

