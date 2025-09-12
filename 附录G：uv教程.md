
```bash
$ pip install uv
$ uv rlhf-env # 创建虚拟环境
$ source rlhf-env/bin/activate # 进入虚拟环境
$ uv pip install jinja2
$ uv pip install pandas
$ uv pip install pyarrow
$ uv pip install pyyaml
$ uv pip install safetensors
$ uv pip install tensorboard
$ uv pip install tokenizers
$ uv pip install torch
$ uv pip install transformers
```

配置国内源

```bash
# 推荐使用清华源
echo 'export UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"'>> ~/.bashrc

# 让配置立即生效
source ~/.bashrc

# 检查环境变量
echo $UV_DEFAULT_INDEX
```
