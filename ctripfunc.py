import pandas as pd
import numpy as np
import calendar
from datetime import datetime


def addDateInterval(date):
    """
    返回当前日期为从2000年1月开始的第几个月
    :dat str
        'YYYY-MM-DD'
    :return int
    """
    if date == '-1':
        return -1
    _date = pd.to_datetime(date)
    if _date.year < 2000:
        return -1
    month_interval = (_date.year - 2000) * 12 + _date.month - 1
    return month_interval


def addVotersDiscretization(x):
    """
    对voters属性进行6级离散化处理，缺失值为-1
    :x int
    :return int
    """
    if x > -1:
        if x < 100:
            return 0
        elif x < 500:
            return 1
        elif x < 1000:
            return 2
        elif x < 2500:
            return 3
        elif x < 10000:
            return 4
        else:
            return 5
    return -1


def addMeanPrice(product_quantity, product_info):
    """
    添加平均售价
    """
    new_pq = product_quantity[product_quantity.price != -1].copy()
    new_pq['sum'] = new_pq['price'] * new_pq['ciiquantity']
    sumc = new_pq.groupby('product_id')['ciiquantity'].sum()
    product_info.loc[product_info.index.isin(new_pq.product_id.unique()), 'price'] = (
        new_pq.groupby('product_id')['sum'].sum() / sumc)
    product_info.fillna(-1, inplace=True)
    return product_info


def addEval0(col):
    """
    根据评分人数和客户评分计算新的评级特征
    :col ['voters','eval3']
    :reuturn int
    """
    voters = col[0]
    eval3 = col[1]
    if voters < 0:
        return -1
    if voters > 2:
        if eval3 > 4:
            return 2
        elif eval3 < 3:
            return 0
    return 1


def transformProductInfo(product_info, product_quantity):
    product_info['index'] = product_info['product_id']
    product_info.set_index('index', inplace=True)
    product_info['startdate'] = product_info['startdate'].apply(
        addDateInterval)
    product_info['upgradedate'] = product_info['upgradedate'].apply(
        addDateInterval)
    product_info['cooperatedate'] = product_info['cooperatedate'].apply(
        addDateInterval)

    # 订单属性1 我们发现订单属性1对于产品来说其实是一个唯一属性,将其添加进info表
    oa1 = product_quantity.drop_duplicates('product_id')
    product_info.loc[oa1['product_id'],
                     'orderattribute1'] = oa1['orderattribute1'].tolist()
    product_info.fillna(-1, inplace=True)

    product_info['voters'] = product_info['voters'].apply(
        addVotersDiscretization)
    product_info = addMeanPrice(product_quantity, product_info)
    product_info['eval0'] = product_info[[
        'voters', 'eval3']].apply(addEval0, axis=1)

    return product_info


def change_dat2(col):
    """
    计算当前月份到产品开售月份(start_date)的月数
    : col   ['startdate','year','month']
    : return int
    """
    start_date, year, month = col
    current_date = addDateInterval('-'.join([str(year), str(month).zfill(2)]))
    if start_date > current_date:
        return -1
    return current_date - start_date


def get_holiday(col):
    '''
    计算指定年月的假期天数
    : col ['year','month']
    : return int
    '''
    holiday = [9, 11, 10, 9, 10, 10, 8, 10, 8, 12, 10, 8,
               10, 11, 9, 9, 11, 8, 8, 10, 9, 13, 9, 8,
               11, 11, 8, 10, 10, 9, 10, 8, 9, 13, 8, 9,
               12]
    year, month = col
    return holiday[(year - 2014) * 12 + month - 1]


def get_x(quantity, product_info):
    """
    根据product_id product_month 生成训练数据集
    : quantity dataframe
    : product_info datafrmae
    : return dataframe
    """
    x = product_info.loc[quantity['product_id']].copy()
    x.reset_index(drop=True, inplace=True)
    x['year'] = quantity['product_month'].str[:4].apply(int)
    x['month'] = quantity['product_month'].str[5:7].apply(int)
    x['startdate'] = x[['startdate', 'year', 'month']].apply(
        change_dat2, axis=1)
    x['upgradedate'] = x[['upgradedate', 'year', 'month']].apply(
        change_dat2, axis=1)
    x['cooperatedate'] = x[['cooperatedate', 'year', 'month']].apply(
        change_dat2, axis=1)
    x['holiday'] = x[['year', 'month']].apply(get_holiday, axis=1)
    x = pd.get_dummies(x, columns=['month'])
    x['month79'] = 1 - (1 - x['month_7']) * (1 - x['month_9'])

    return x
