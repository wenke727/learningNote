# Pandas

## 学习资源

- [全平台支持的pandas运算加速神器](https://www.cnblogs.com/feffery/p/13049547.html)　:airplane:[仓库路径](https://github.com/CNFeffery/DataScienceStudyNotes/tree/master/%EF%BC%88%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%AD%A6%E4%B9%A0%E6%89%8B%E6%9C%AD86%EF%BC%89%E5%85%A8%E5%B9%B3%E5%8F%B0%E6%94%AF%E6%8C%81%E7%9A%84pandas%E8%BF%90%E7%AE%97%E5%8A%A0%E9%80%9F%E7%A5%9E%E5%99%A8)
- [掌握pandas中的transform](https://www.cnblogs.com/feffery/p/13816362.html)　:airplane:[仓库路径](https://github.com/CNFeffery/DataScienceStudyNotes/tree/master/%EF%BC%88%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E5%AD%A6%E4%B9%A0%E6%89%8B%E6%9C%AD97%EF%BC%89%E6%8E%8C%E6%8F%A1pandas%E4%B8%AD%E7%9A%84transform)

## 格式设置

在使用dataframe时遇到datafram在列太多的情况下总是自动换行显示的情况，导致数据阅读困难
在代码中设置显示的长宽等, [REF](https://blog.csdn.net/lihuarongaini/article/details/101298171)

``` python
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
```

## 常用函数

- 多重索引定位：一级一级的链式访问方式
  
    ``` python
    gt = graph_t.drop_duplicates(['e_0','s_1']).set_index(['e_0', 's_1'])
    gt.loc[2525990691].loc[5834799167]
    ```

- 创建空白

    ``` python
    df = pd.DataFrame(columns = ["ebayno", "p_sku", "sale", "sku"]) 
    ```

- 时间切换

    ``` python
    # string变成datetime格式 
    temp['dates'] = pd.to_datetime( temp['date']  , format = '%Y%m%d') 
    # datetime变回string格式 
    temp['date'] = temp['time'].apply(lambda x: x.strftime('%Y-%m-%d'))
    ```

- 增删改查

    ``` python
    # 增加
    df.append(row, ignore_index=True)
    ```

- pivot

    ``` python
    link_emission = pd.pivot_table(
            data=moves_output, 
            index=['linkID', 'sourceTypeID'], 
            columns=['pollutantID'], 
            values='emissionQuant', 
            aggfunc=np.sum, 
            fill_value=0
        )

    ```

- 按类别排序

    ``` python
    test["city_o"]= test["city_o"].astype("category")
    test['city_o'].cat.set_categories(['香港','澳门','广州','深圳','珠海','佛山','惠州','东莞','中山','江门','肇庆'],inplace=True)
    ```

- 笛卡尔积

    ``` python
    # Cartesian product
    base_atts = ['pid', 'rindex','s', 'e', 'offset']
    a, b = tList[i][base_atts], tList[i+1][base_atts+['observ_prob']]
    a.loc[:, 'tmp'], b.loc[:, 'tmp'] = 1, 1
    graph_t.append(a.merge(b, on='tmp', suffixes=["_0", '_1']).drop(columns='tmp') )
    ```

- to_dict

    ``` python
    df = pd.DataFrame({'col1': [1, 2],
                   'col2': [0.5, 0.75]},
                  index=['row1', 'row2'])
    df.to_dict()
    {'col1': {'row1': 1, 'row2': 2}, 'col2': {'row1': 0.5, 'row2': 0.75}}

    df.to_dict('records')
    [{'col1': 1, 'col2': 0.5}, {'col1': 2, 'col2': 0.75}]

    df.to_dict('index')
    {'row1': {'col1': 1, 'col2': 0.5}, 'row2': {'col1': 2, 'col2': 0.75}}

    ```

- idxmax
Return index of first occurrence of maximum over requested axis.

    ``` python
    # axis{0 or ‘index’, 1 or ‘columns’}, default 0
    # The axis to use. 0 or ‘index’ for row-wise, 1 or ‘columns’ for column-wise.

    # skipnabool, default True
    # Exclude NA/null values. If an entire row/column is NA, the result will be NA.
    ```

- [apply](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html)

    ``` python
    # Returning a Series inside the function is similar to passing result_type='expand'. The resulting column names will be the Series index.
    df.apply(lambda x: pd.Series([1, 2], index=['foo', 'bar']), axis=1)
    
    df_candidates[['offset', 'seg_first', 'seg_last']] = df_candidates.apply(lambda x: 
        cal_relative_offset(traj.loc[x.pid].geometry, net.df_edges.loc[x.rindex].geometry), axis=1, result_type='expand')

    ```

- test

    ``` python
    ```