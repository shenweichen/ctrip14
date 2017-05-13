
# coding: utf-8

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import BaggingRegressor
from ctripfunc import transformProductInfo, get_x


def main():
    # 读取数据
    data_path = './'
    product_quantity = pd.read_csv(data_path + 'product_quantity.txt')
    product_info = pd.read_csv(data_path + 'product_info.txt',)

    # 特征处理

    product_info = transformProductInfo(product_info, product_quantity)

    # 将所有商品每个月的销量转化为一个37*4001的数组供之后处理负数预测值的时候使用
    product_quantity['product_date'] = product_quantity['product_date'].str[:7]
    quantity = product_quantity.groupby(['product_id', 'product_date']).sum()
    quantity.reset_index(inplace=True)

    quantity_arr = np.full((37, 4001), -1, dtype=np.int32)
    for idx in quantity.index:
        pid = quantity.loc[idx, 'product_id']
        date = quantity.loc[idx, 'product_date']
        quantity_arr[(int(date[:4]) - 2014) * 12 + int(date[5:7]) -
                     1][pid] = quantity.loc[idx, 'ciiquantity']

    train_y = quantity['ciiquantity'].tolist()

    quantity.rename(columns={'product_date': 'product_month'}, inplace=True)
    train_x = get_x(quantity, product_info)

    # 模型训练

    print('training start')
    num_clfs = 4
    clfs = []
    for i in range(num_clfs):
        clf = BaggingRegressor(lgb.sklearn.LGBMRegressor(max_depth=10, n_estimators=2000, num_leaves=80 + 10 * i,),
                               n_estimators=6, random_state=i, n_jobs=1, max_samples=0.8 + 0.001 * i,)
        clf.fit(train_x, train_y,)
        clfs.append(clf)
    print('training  done')

    # 执行预测
    print('predicting start')
    out = pd.read_csv('../data/prediction_lilei_20170320.txt')
    test_x = get_x(out[['product_id', 'product_month']], product_info)
    ans = [clf.predict(test_x) for clf in clfs]
    # 23个月均无数据的product_id
    invalid_pid = [i for i in range(1, 4001) if quantity_arr[:23, i].max() < 0]
    # 将负数的预测改为前23个月有效的最小值,若无则为0
    history_min = [0] * 4001
    for i in range(1, 4001):
        quantity_i = quantity_arr[:23, i]
        if quantity_i.max() < 0:
            history_min[i] = 0
        else:
            history_min[i] = quantity_i[quantity_i > -1].min()
    # 生成最后提交结果
    final_out = out.copy()
    final_out['ciiquantity_month'] = 0
    for pred_y in ans:
        out['ciiquantity_month'] = pred_y
        # 将23个月均无数据且startdate,cooperatedate均在第24个月以后的清0
        for pid in invalid_pid:
            product_info_i = product_info.loc[pid]
            dat = min(product_info_i['startdate'],
                      product_info_i['cooperatedate']) - 191 + 23
            for j in range(23, dat.astype(np.int)):
                out.loc[(j - 23) * 4000 + pid - 1, 'ciiquantity_month'] = 0

        idx = out['ciiquantity_month'] < 0
        out.loc[idx, 'ciiquantity_month'] = np.array(
            history_min)[out.loc[idx, 'product_id']]
        final_out['ciiquantity_month'] += out['ciiquantity_month']
    out['ciiquantity_month'] = final_out['ciiquantity_month'] / len(ans)

    out.to_csv('l_bg46_lgb100_-1first.txt', index=False)
    print('predicting done')


if __name__ == "__main__":
    main()
