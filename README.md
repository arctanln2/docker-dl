# docker-dl


```bash
- sig-mnist
    Dockerfile
    main.py
- ddp-mnist
    Dockerfile
    main.py
```

## 单GPU
以sig-mnist为例，这里挂载可以将下载的数据集、运行过程中的log保存到相应的文件夹里，不用再次下载和方便之后查看结果。
```bash
cd sig-mnist
docker build -t sig-mnist:v1 .
docker run --rm -v "${PWD}:/opt/mnist/src/" sig-mnist:v1 python main.py
```

## 多GPU
以ddp-mnist为例，当有两个GPU时
```bash
cd ddp-mnist
docker build -t ddp-mnist:v1 .
docker run --rm -v "${PWD}:/opt/mnist/src/" ddp-mnist:v1 torchrun --nnodes 1 --nproc_per_node=2 main.py
```