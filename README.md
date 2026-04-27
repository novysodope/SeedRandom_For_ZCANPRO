# SeedRandom_For_ZCANPRO
与ZCANPRO结合，获得UDS种子进行种子随机性测试，用来验证种子是否真随机

支持16种测试并会生成相关图片

1、需要先让开发提供一个没有惩罚机制的版本，不然会因为请求太多触发0x36、0x37；
2、修改collector的配置区，然后把修改后的`collector.py`导入到周立功，运行获取种子
3、最后修改`analyzer.py`里的配置，使用`python3 analyzer.py`运行等待结果
