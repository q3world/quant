# quant

多品种实时量化计算

（软件内实现的量化策略，仅可作为参考，:warning:投资有风险，交易请谨慎）

### run

1. 在 quant.cfg 内，按格式，设定选择的代码（"000001.0"，尾缀代表市场：0 为深市；1 为沪市）
2. 运行 python quant.py 启动程序

## interface

quant.png

提示数据包括：品种代码、品种名称、当前价格、时间频、策略名称、策略数据（均值、价格均值差、价格均值差率）

## log

### 2021/11/9

增加数据源 pytdx （约定数据源）

（akshare 比较慢，pytdx 快n多）

### 2021/11/6

quant 运行后，直接启动一个线程循环执行；

在交易时间段内，会不断提取数据，计算量化策略（当前数据通过 akshare 提取）；

初始时，提取各时间频 k线数据，若数据源无此时间频数据，则用小时间频数据生成（10m线由 5m线生成、2h线由 1h线生成）；

每次循环执行时，仅提取有限最小时间频数据，然后更新各时间频数据；

当前，仅实现 MA策略：当价格回落到均线（ma5，ma10， ma20， ma30， ma60， ma120，ma240）不足一个点，且该均线向上时，则输出并鸣响提示。

## ref

akshare： https://github.com/akfamily/akshare

pytdx： https://github.com/rainx/pytdx
