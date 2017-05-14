# 出行产品未来14个月销量预测
##比赛描述

携程作为中国领先的综合性旅行服务公司，每天向超过2.5亿会员提供全方位的旅行服务，海量的网站访问产生了海量的数据，从中挖掘潜在的信息资源有着重要意义。调研发现，在出行产品预订业务中，不同区域的产品需求量级不一样，不同时段需求量会有高低起伏，相同区域相同时段各产品的需求量因产品特性不同又有差异。此次竞赛的目的正是为了深入了解产品需求量和产品特性、历史销量的关系，挖掘出影响需求量的关键因素，预测出行产品未来14个月每月的销量，从而指导产品的库存管理和定价策略，这对于收益管理和客户价值的提升有着重要作用。[比赛地址](https://www.kesci.com/apps/home_log/index.html#!/competition/58bfc27471db03332e1b8a36/content/0)
[数据地址](https://www.kesci.com/apps/home_log/index.html#!/lab/dataset/58bf9bb671db03332e1b85f3/document)  

赛后和其他选手交流后感觉比我们的方案做的很粗糙，还要太多需要学习的地方。

## 运行环境

- Windows10
- Python 3.5.2
- lightgbm 0.1
- scikit-learn 0.18.1
- Pandas 0.19.2
- Numpy 1.12.0

## 文件说明

- ctripfunc.py  特征处理函数
- solution.py   主函数
- prediction_lilei_20170320.txt 官方的提交样例，用于生成最终结果的格式
- best_result.txt 最终预测结果文件

## 运行说明

- 将原始数据文件，包括`product_info.txt`和`product_quantity.txt`放置在代码同级目录下，直接运行`solution.py`即可，生成的`l_bg46_lgb100_1first.txt`为最终提交的结果文件。
