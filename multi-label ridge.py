import pandas as pd
import numpy as np
import math
import jqdata
from jqfactor import standardlize
from jqfactor import winsorize_med
from jqdata import *
from sklearn.model_selection import KFold
from jqlib.technical_analysis import *
from pandas import DataFrame,Series

def initialize(context):
    set_params()
    set_backtest()
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5), type='stock')
    set_slippage(FixedSlippage(0))
##
def set_params():
    # 记录回测运行的天数
    g.days = 0
    # 当天是否交易
    g.if_trade = False

    # 股票池
    g.secCode = '000985.XSHG'  #中证全指
    #g.secCode = '000300.XSHG'
    #g.secCode = '000905.XSHG' #中证500
    # 调仓天数
    g.refresh_rate = 20
    # g.refresh_rate = 15
    # g.refresh_rate = 20
    # g.refresh_rate = 5
    # 线性回归：lr
    # 岭回归：ridge
    # 线性向量机：svr
    # 随机森林：rf
    g.method = 'svr'
    # 分组
    g.group = 10
    # 持仓数
    g.stocknum = 5
    # g.stocknum = 10
    # g.stocknum = 30
    # g.stocknum = 50
    g.invest_by_group = True
    g.quantile = 0.1 # 10%


def set_backtest():
    set_benchmark('000985.XSHG')   #中证全指
    set_option('use_real_price', True)
    log.set_level('order', 'error')

def process_initialize(context):
    g.tradeCount = 0
    g.feasible_stocks = []
    # 网格搜索是否开启
    g.gridserach = False
    g.trainlength = 3
    # 周期交易日
    g.intervals = 20

    # 离散值处理列表
    g.winsorizeList = ['log_NC','LEV']

    # 标准化处理列表
    g.standardizeList = ['log_mcap',
                        'log_NC',
                        'LEV',
                        'g',
                        'CYEL','CYES']

    # 聚宽一级行业
    g.industry_set = ['HY001', 'HY002', 'HY003', 'HY004', 'HY005', 'HY006', 'HY007', 'HY008', 'HY009',
          'HY010', 'HY011']

    # 因子列表(因子组合3)
    g.factorList = ['log_NC', 'LEV', 'g','CYEL','CYES',
                    'HY001', 'HY002', 'HY003', 'HY004', 'HY005', 'HY006', 'HY007', 'HY008', 'HY009', 'HY010', 'HY011']

#开盘前
def before_trading_start(context):
    # 当天是否交易
    g.if_trade = False
    if g.days % g.refresh_rate == 0:
        g.if_trade = True
        sample = get_index_stocks(g.secCode)
        g.feasible_stocks = set_feasible_stocks(sample,context)
        g.q = get_q_Factor(g.feasible_stocks)
    g.days+=1

# 设置可行股票池：过滤掉当日停牌的股票
def set_feasible_stocks(initial_stocks,context):
    current_data = get_current_data()
    security_list = [stock for stock in initial_stocks if not current_data[stock].paused]
    return security_list

# 交易时
def handle_data(context,data):
    if g.if_trade == True:
        g.tradeCount = g.tradeCount + 1
        # 训练集
        yesterday = context.previous_date
        today = context.current_dt
        df_train = get_df_train(g.q,yesterday,g.trainlength,g.intervals)
        df_train = initialize_df(df_train,yesterday)
        # 测试集
        df = get_fundamentals(g.q, date = None)
        df = get_CYE(df,today)
        df = initialize_df(df,today)
        # 离散值处理
        for fac in g.winsorizeList:
            df_train[fac] = winsorize_med(df_train[fac], scale=5, inclusive=True, inf2nan=True, axis=0)
            df[fac] = winsorize_med(df[fac], scale=5, inclusive=True, inf2nan=True, axis=0)
        # 标准化处理
        for fac in g.standardizeList:
            df_train[fac] = standardlize(df_train[fac], inf2nan=True, axis=0)
            df[fac] = standardlize(df[fac], inf2nan=True, axis=0)
        # 行业中性化
        df_train = neutralize(df_train,g.industry_set)
        df = neutralize(df,g.industry_set)

        #训练集（包括验证集）
        X_trainval = df_train[g.factorList]
        X_trainval = X_trainval.fillna(0)

        #训练集输出
        y_trainval = df_train[['log_mcap']]
        y_trainval = y_trainval.fillna(0)

        #测试集
        X = df[g.factorList]
        X = X.fillna(0)

        #测试集输出
        y = df[['log_mcap']]
        y.index = df['code']
        y = y.fillna(0)

        kfold = KFold(n_splits=5)
        if g.gridserach == False:
            #不带网格搜索的机器学习
            if g.method == 'svr': #SVR
                from sklearn.svm import SVR
                model = SVR(C=100, gamma=1)
            elif g.method == 'lr':
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
            elif g.method == 'ridge': #岭回归
                from sklearn.linear_model import Ridge
                model = Ridge(random_state=42,alpha=100)
            elif g.method == 'rf': #随机森林
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(random_state=42,n_estimators=500,n_jobs=-1)
        else:
            # 带网格搜索
            para_grid = {}
            if g.method == 'svr':
                from sklearn.svm import SVR
                para_grid = {'C':[10,100],'gamma':[0.1,1,10]}
                grid_search_model = SVR()
            elif g.method == 'lr':
                from sklearn.linear_model import LinearRegression
                grid_search_model = LinearRegression()
            elif g.method == 'ridge':
                from sklearn.linear_model import Ridge
                para_grid = {'alpha':[1,10,100]}
                grid_search_model = Ridge(random_state = 42)
            elif g.method == 'rf':
                from sklearn.ensemble import RandomForestRegressor
                para_grid = {'n_estimators':[100,500,1000]}
                grid_search_model = RandomForestRegressor(random_state = 42)

            from sklearn.model_selection import GridSearchCV
            model = GridSearchCV(grid_search_model,para_grid,cv=kfold,n_jobs=-1)

        # 生成模型
        model.fit(X_trainval,y_trainval)
        # 预测值
        y_pred = model.predict(X)

        # 实际值与预测值之差
        factor = y - pd.DataFrame(y_pred, index = y.index, columns = ['log_mcap'])

        if g.invest_by_group == True:
            len_secCodeList = len(list(factor.index))
            g.stocknum = int(len_secCodeList * g.quantile)
        # 残差进行排序
        factor = factor.sort_index(by = 'log_mcap')
        start = g.stocknum * (g.group-1)
        end = g.stocknum * g.group
        stockset = list(factor.index[start:end])

        current_data = get_current_data()

        #卖出
        sell_list = list(context.portfolio.positions.keys())
        for stock in sell_list:
            if stock not in stockset:
                if stock in g.feasible_stocks:
                    if current_data[stock].last_price == current_data[stock].high_limit:
                        pass
                    else:
                        stock_sell = stock
                        order_target_value(stock_sell, 0)

        #分配买入资金
        if len(context.portfolio.positions) < g.stocknum:
            num = g.stocknum - len(context.portfolio.positions)
            cash = context.portfolio.cash/num
        else:
            cash = 0
            num = 0

        #买入
        for stock in stockset[:g.stocknum]:
            if stock in sell_list:
                pass
            else:
                if current_data[stock].last_price == current_data[stock].low_limit:
                    pass
                else:
                    stock_buy = stock
                    order_target_value(stock_buy, cash)
                    num = num - 1
                    if num == 0:
                        break
# 获取初始特征值
def get_q_Factor(feasible_stocks):
    q = query(valuation.code, valuation.market_cap, balance.total_assets - balance.total_liability,
            balance.total_assets / balance.total_liability, indicator.inc_revenue_year_on_year).filter(valuation.code.in_(feasible_stocks))
    return q

# 训练集长度设置
def get_df_train(q,d,trainlength,interval):

    date1 = shift_trading_day(d,interval)
    date2 = shift_trading_day(d,interval*2)
    date3 = shift_trading_day(d,interval*3)

    d1 = get_fundamentals(q, date = date1)
    d2 = get_fundamentals(q, date = date2)
    d3 = get_fundamentals(q, date = date3)
    d1 = get_CYE(d1,date1)
    d2 = get_CYE(d2,date2)
    d3 = get_CYE(d3,date3)


    if trainlength == 1:
        df_train = d1
    elif trainlength == 3:
        # 3个周期作为训练集
        df_train = pd.concat([d1, d2, d3],ignore_index=True)
    elif trainlength == 4:
        date4 = shift_trading_day(d,interval*4)
        d4 = get_fundamentals(q, date = date4)
        d4 = get_CYE(d4,date4)
        # 4个周期作为训练集
        df_train = pd.concat([d1, d2, d3, d4],ignore_index=True)
    elif trainlength == 6:
        date4 = shift_trading_day(d,interval*4)
        date5 = shift_trading_day(d,interval*5)
        date6 = shift_trading_day(d,interval*6)

        d4 = get_fundamentals(q, date = date4)
        d5 = get_fundamentals(q, date = date5)
        d6 = get_fundamentals(q, date = date6)
        d4 = get_CYE(d4,date4)
        d5 = get_CYE(d5,date5)
        d6 = get_CYE(d6,date6)

        # 6个周期作为训练集
        df_train = pd.concat([d1,d2,d3,d4,d5,d6],ignore_index=True)
    elif trainlength == 9:
        date4 = shift_trading_day(d,interval*4)
        date5 = shift_trading_day(d,interval*5)
        date6 = shift_trading_day(d,interval*6)
        date7 = shift_trading_day(d,interval*7)
        date8 = shift_trading_day(d,interval*8)
        date9 = shift_trading_day(d,interval*9)

        d4 = get_fundamentals(q, date = date4)
        d5 = get_fundamentals(q, date = date5)
        d6 = get_fundamentals(q, date = date6)
        d7 = get_fundamentals(q, date = date7)
        d8 = get_fundamentals(q, date = date8)
        d9 = get_fundamentals(q, date = date9)
        d4 = get_CYE(d4,date4)
        d5 = get_CYE(d5,date5)
        d6 = get_CYE(d6,date6)
        d7 = get_CYE(d7,date7)
        d8 = get_CYE(d8,date8)
        d9 = get_CYE(d9,date9)

        # 9个周期作为训练集
        df_train = pd.concat([d1,d2,d3,d4,d5,d6,d7,d8,d9],ignore_index=True)
    else:
        pass

    return df_train

# 某一日的前shift个交易日日期
def shift_trading_day(date,shift):
    # 获取所有的交易日，返回一个包含所有交易日的 list,元素值为 datetime.date 类型.
    tradingday = get_all_trade_days()
    # 得到date之后shift天那一天在列表中的行标号 返回一个数
    shiftday_index = list(tradingday).index(date) - shift
    # 根据行号返回该日日期 为datetime.date类型
    return tradingday[shiftday_index]

def get_CYE(df,date):
    df.columns = ['code',
                'mcap',
                'NC',
                'LEV',
                'g',
                ]
    CYEL,CYES = CYE(df.code.tolist(),check_date = date)
    df['CYEL'] = Series(CYEL)
    df['CYES'] = Series(CYES)
    return df

# 特征值提取
def initialize_df(df,date):
    df['log_mcap'] = np.log(df['mcap'])
    df['log_NC'] = np.log(df['NC'])
    del df['mcap']
    del df['NC']
    CYEL,CYES = CYE(df.code.tolist(),check_date = date)
    df['CYEL'] = Series(CYEL)
    df['CYES'] = Series(CYES)
    df = df.fillna(0)
    return df

# 中性化
def neutralize(df,industry_set):
    for i in range(len(industry_set)):
        industry = get_industry_stocks(industry_set[i], date = None)
        s = pd.Series([0]*len(df), index=df.index)
        s[set(industry) & set(df.index)]=1
        df[industry_set[i]] = s

    return df

# 收盘后
def after_trading_end(context):
    return
