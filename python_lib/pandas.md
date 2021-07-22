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
