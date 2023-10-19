# SpikePlotter
using nwb/bids data format, refer the elephant-repo, wtrite a relative-universal back-end plot pack ---- it can simply plugin with the front qt widget 

app/ 存放代码
doc/ 定期整理的指示性文档(markdown)
log/ 日志, 以天为分辨率, 记录查阅的资料, 遇到的问题, 以及一些思考
data/ 测试用的数据

总体的目标, 希望能够读取常见的侵入式脑电数据集, 这至少包括NWB/BIDS等文件格式.
读取进来后, 需要和另一位组员确定好神经发放信息和事件信息的基本数据结构（？能确定下来吗）
基于这样的中间数据结构, 我需要据此绘制图形, 另一位组员需要进行解码分析, 其分析结果亦需进行可视化

另一方面已知的是, 有一些库可以用于参考, 主要是 elephant, 实现形式类似于功能函数的集合库, 这些功能函数
依据分析对象(输入)的不同需要分为多个类型, 每个类型下面又有很多功能, 其目的在于: 同一类型中, 其功能函数的
参数接口应当相同, 这可以参考 elephant的实现.

还有一点需要考虑, 这些代码完成后, 它将类似于一个算法库, 后台调用它能够方便的和前台(界面交互), 也即这些功能函数
返回的应当是绘图所需要的直接的(x, y)数据.

## 10-18
将文件中的数据处理为合适的 SpikeTrain(s)
输入: SpikeTrain
输出: nd-array or AnalogSignal


