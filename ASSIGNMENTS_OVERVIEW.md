# CS336 Assignment 1: 作业概览文档

## 课程信息

- **课程**: CS336 Spring 2025 - Building a Transformer LM
- **版本**: 1.0.6
- **目标**: 从零实现一个完整的Transformer语言模型

---

## 作业总览

本作业包含4大部分：

1. **BPE分词器** (§2) - 实现字节对编码分词器
2. **Transformer语言模型** (§3) - 实现模型架构
3. **训练组件** (§4) - 交叉熵损失、AdamW优化器、训练循环
4. **实验与生成** (§5-7) - 训练模型、文本生成、消融实验

---

## 运行测试命令

```bash
# 运行所有测试
uv run pytest

# 运行特定测试文件
uv run pytest tests/test_tokenizer.py
uv run pytest tests/test_train_bpe.py
uv run pytest tests/test_nn_utils.py
uv run pytest tests/test_model.py
uv run pytest tests/test_optimizer.py
uv run pytest tests/test_data.py
uv run pytest tests/test_serialization.py

# 运行特定测试
uv run pytest -k test_linear
uv run pytest -k test_embedding
```

---

## 第一部分：BPE分词器 (Byte-Pair Encoding Tokenizer)

### 1.1 Unicode理解 (Problem: unicode1) - 1分

**目标**: 理解Unicode字符编码基础

- chr(0)返回什么字符
- repr()与print()的区别
- 特殊字符在文本中的行为

**提交形式**: 书面回答 (writeup.pdf)

---

### 1.2 Unicode编码 (Problem: unicode2) - 3分

**目标**: 理解UTF-8编码原理

**提交形式**: 书面回答 (writeup.pdf)

---

### 1.3 BPE分词器训练 (Problem: train_bpe) - 15分

**实现adapter**: `tests/adapters.py` 中的 `run_train_bpe`

```python
def run_train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """训练BPE分词器"""
    raise NotImplementedError
```

**pytest测试** (`tests/test_train_bpe.py`):

| 测试函数                        | 验证内容                                |
| ------------------------------- | --------------------------------------- |
| `test_train_bpe_speed`          | BPE训练速度 < 1.5秒 (小数据集)          |
| `test_train_bpe`                | 训练结果与参考实现一致 (vocab + merges) |
| `test_train_bpe_special_tokens` | 特殊token正确添加且不被合并             |

**关键实现要求**:

1. 词汇表初始化：256个字节 + 特殊token
2. 预分词：使用GPT-2正则表达式 `regex` 库
3. 合并计算：迭代找出最高频字节对进行合并
4. 相同频率时：选择字典序(lexicographically)最大的pair
5. 支持特殊token：预分词前需要先分割特殊token

**运行测试**:

```bash
uv run pytest tests/test_train_bpe.py -v
```

---

### 1.4 BPE训练实验 (Problems: train_bpe_tinystories, train_bpe_expts_owt) - 4分

**提交形式**: 书面回答 (writeup.pdf)

**任务**:

- TinyStories数据集：vocab_size=10000，训练时间≤30分钟
- OpenWebText数据集：vocab_size=32000，训练时间≤12小时
- 分析最长token、训练时间、内存使用、瓶颈

---

### 1.5 分词器实现 (Problem: tokenizer) - 15分

**实现adapter**: `tests/adapters.py` 中的 `get_tokenizer`

```python
def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None
) -> Tokenizer:
    """从vocab和merges创建分词器"""
    raise NotImplementedError
```

**需要实现的Tokenizer类方法**:

```python
class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None)
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None)
    def encode(self, text: str) -> list[int]
    def encode_iterable(self, iterable: Iterator[str]) -> Iterator[int]
    def decode(self, ids: list[int]) -> str
```

**pytest测试** (`tests/test_tokenizer.py`):

| 测试函数                                            | 验证内容                               |
| --------------------------------------------------- | -------------------------------------- |
| `test_roundtrip_empty`                              | 空字符串encode/decode往返一致          |
| `test_single_character`                             | 单字符往返一致                         |
| `test_single_unicode_character`                     | Unicode字符往返一致 (如 🙃)            |
| `test_roundtrip_ascii_string`                       | ASCII字符串往返一致                    |
| `test_roundtrip_unicode_string`                     | Unicode字符串往返一致                  |
| `test_roundtrip_unicode_string_with_special_tokens` | 含特殊token往返一致                    |
| `test_overlapping_special_tokens`                   | 重叠特殊token处理                      |
| `test_address_roundtrip`                            | 地址文件往返一致                       |
| `test_german_roundtrip`                             | 德语文本往返一致                       |
| `test_tinystories_sample_roundtrip`                 | TinyStories样本往返一致                |
| `test_*_matches_tiktoken`                           | 各种测试与tiktoken参考实现比对         |
| `test_encode_iterable_*`                            | 流式编码功能                           |
| `test_encode_iterable_memory_usage`                 | 流式编码内存效率 (Linux only, 1MB限制) |

**运行测试**:

```bash
uv run pytest tests/test_tokenizer.py -v
```

---

### 1.6 分词器实验 (Problem: tokenizer_experiments) - 4分

**提交形式**: 书面回答 (writeup.pdf)

---

## 第二部分：Transformer语言模型架构

### 2.1 Linear层 (Problem: linear) - 1分

**实现adapter**: `tests/adapters.py` 中的 `run_linear`

```python
def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, "d_out d_in"],
    in_features: Float[Tensor, "... d_in"]
) -> Float[Tensor, "... d_out"]:
    """线性变换 y = Wx (无偏置)"""
    raise NotImplementedError
```

**pytest测试**: `test_linear` (使用snapshot testing)

**要求**:

- 权重形状: `[d_out, d_in]`
- 初始化: `N(0, 2/(d_in + d_out))` 截断在±3σ

**运行测试**:

```bash
uv run pytest -k test_linear -v
```

---

### 2.2 Embedding层 (Problem: embedding) - 1分

**实现adapter**: `tests/adapters.py` 中的 `run_embedding`

```python
def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, "vocab_size d_model"],
    token_ids: Int[Tensor, "..."]
) -> Float[Tensor, "... d_model"]:
    """Token嵌入查找"""
    raise NotImplementedError
```

**pytest测试**: `test_embedding` (使用snapshot testing)

**要求**:

- 权重形状: `[vocab_size, d_model]`
- 初始化: `N(0, 1)` 截断在±3

**运行测试**:

```bash
uv run pytest -k test_embedding -v
```

---

### 2.3 RMSNorm (Problem: rmsnorm) - 1分

**实现adapter**: `tests/adapters.py` 中的 `run_rmsnorm`

```python
def run_rmsnorm(
    d_model: int,
    eps: float = 1e-5,
    weights: Float[Tensor, "d_model"],
    in_features: Float[Tensor, "... d_model"]
) -> Float[Tensor, "... d_model"]:
    """RMS归一化"""
    raise NotImplementedError
```

**pytest测试**: `test_rmsnorm` (使用snapshot testing, atol=1e-6)

**公式**:

```
RMSNorm(x_i) = x_i / RMS(x) * g_i
RMS(x) = sqrt(1/d_model * sum(x_i^2) + eps)
```

**要求**:

- 升到float32计算以防溢出
- 返回原始dtype

**运行测试**:

```bash
uv run pytest -k test_rmsnorm -v
```

---

### 2.4 SwiGLU / Position-wise Feed-Forward (Problem: positionwise_feedforward) - 2分

**实现adapter**: `tests/adapters.py` 中的 `run_swiglu`

```python
def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, "d_ff d_model"],
    w2_weight: Float[Tensor, "d_model d_ff"],
    w3_weight: Float[Tensor, "d_ff d_model"],
    in_features: Float[Tensor, "... d_model"]
) -> Float[Tensor, "... d_model"]:
    """SwiGLU前馈网络"""
    raise NotImplementedError
```

**pytest测试**: `test_swiglu` (使用snapshot testing, atol=1e-5)

**公式**:

```
FFN(x) = W_2(SiLU(W_1 x) ⊙ W_3 x)
其中 d_ff ≈ (8/3) * d_model，且是64的倍数
```

**运行测试**:

```bash
uv run pytest -k test_swiglu -v
```

---

### 2.5 Softmax (Problem: softmax) - 1分

**实现adapter**: `tests/adapters.py` 中的 `run_softmax`

```python
def run_softmax(
    in_features: Float[Tensor, "..."],
    dim: int
) -> Float[Tensor, "..."]:
    """数值稳定的softmax"""
    raise NotImplementedError
```

**pytest测试** (`tests/test_nn_utils.py`):

| 测试函数                       | 验证内容                                |
| ------------------------------ | --------------------------------------- |
| `test_softmax_matches_pytorch` | 与PyTorch F.softmax结果一致 (atol=1e-6) |
|                                | 验证数值稳定性 (输入+100)               |

**运行测试**:

```bash
uv run pytest -k test_softmax -v
```

---

### 2.6 RoPE (Problem: rope) - 2分

**实现adapter**: `tests/adapters.py` 中的 `run_rope`

```python
def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, "... sequence_length d_k"],
    token_positions: Int[Tensor, "... sequence_length"]
) -> Float[Tensor, "... sequence_length d_k"]:
    """旋转位置嵌入"""
    raise NotImplementedError
```

**pytest测试**: `test_rope` (使用snapshot testing, atol=1e-6)

**要求**:

- 预计算sin/cos buffer
- 支持任意batch维度
- 使用token_positions切片

**运行测试**:

```bash
uv run pytest -k test_rope -v
```

---

### 2.7 Scaled Dot-Product Attention (Problem: scaled_dot_product_attention) - 5分

**实现adapter**: `tests/adapters.py` 中的 `run_scaled_dot_product_attention`

```python
def run_scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... values d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    """缩放点积注意力"""
    raise NotImplementedError
```

**pytest测试**:

| 测试函数                               | 验证内容                                 |
| -------------------------------------- | ---------------------------------------- |
| `test_scaled_dot_product_attention`    | 3D输入 (batch, seq, d) (atol=1e-6)       |
| `test_4d_scaled_dot_product_attention` | 4D输入 (batch, head, seq, d) (atol=1e-6) |

**公式**:

```
Attention(Q, K, V) = softmax(Q^T K / sqrt(d_k)) V
```

**要求**:

- 支持任意batch维度
- mask中True=可attend, False=不可attend

**运行测试**:

```bash
uv run pytest -k test_scaled_dot_product_attention -v
```

---

### 2.8 Multi-Head Self-Attention (Problem: multihead_self_attention) - 5分

**实现adapter**: `tests/adapters.py` 中的 `run_multihead_self_attention`

```python
def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, "d_k d_in"],
    k_proj_weight: Float[Tensor, "d_k d_in"],
    v_proj_weight: Float[Tensor, "d_v d_in"],
    o_proj_weight: Float[Tensor, "d_model d_v"],
    in_features: Float[Tensor, "... sequence_length d_in"]
) -> Float[Tensor, "... sequence_length d_out"]:
    """多头自注意力 (无RoPE)"""
    raise NotImplementedError
```

**pytest测试**:

| 测试函数                        | 验证内容                   |
| ------------------------------- | -------------------------- |
| `test_multihead_self_attention` | 批量多头注意力 (atol=1e-6) |

**要求**:

- 单次矩阵乘法计算所有头的QKV投影
- 因果掩码 (causal mask)
- 不使用RoPE

**运行测试**:

```bash
uv run pytest -k test_multihead_self_attention -v
```

---

### 2.9 Multi-Head Self-Attention with RoPE (Problem: multihead_self_attention) - 包含在上文中

**实现adapter**: `tests/adapters.py` 中的 `run_multihead_self_attention_with_rope`

```python
def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, "d_k d_in"],
    k_proj_weight: Float[Tensor, "d_k d_in"],
    v_proj_weight: Float[Tensor, "d_v d_in"],
    o_proj_weight: Float[Tensor, "d_model d_v"],
    in_features: Float[Tensor, "... sequence_length d_in"],
    token_positions: Int[Tensor, "... sequence_length"] | None = None,
) -> Float[Tensor, "... sequence_length d_out"]:
    """多头自注意力 (带RoPE)"""
    raise NotImplementedError
```

**pytest测试**: `test_multihead_self_attention_with_rope` (atol=1e-6)

**运行测试**:

```bash
uv run pytest -k test_multihead_self_attention_with_rope -v
```

---

### 2.10 Transformer Block (Problem: transformer_block) - 3分

**实现adapter**: `tests/adapters.py` 中的 `run_transformer_block`

```python
def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, "batch sequence_length d_model"]
) -> Float[Tensor, "batch sequence_length d_model"]:
    """Pre-norm Transformer块"""
    raise NotImplementedError
```

**pytest测试**: `test_transformer_block` (使用snapshot testing, atol=1e-6)

**结构** (Pre-norm):

```
y = x + MultiHeadSelfAttention(RMSNorm(x))
z = y + SwiGLU(RMSNorm(y))
```

**运行测试**:

```bash
uv run pytest -k test_transformer_block -v
```

---

### 2.11 Transformer LM (Problem: transformer_lm) - 3分

**实现adapter**: `tests/adapters.py` 中的 `run_transformer_lm`

```python
def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, "batch_size sequence_length"]
) -> Float[Tensor, "batch_size sequence_length vocab_size"]:
    """完整Transformer语言模型"""
    raise NotImplementedError
```

**pytest测试**:

| 测试函数                              | 验证内容                              |
| ------------------------------------- | ------------------------------------- |
| `test_transformer_lm`                 | 完整LM前向传播 (atol=1e-4, rtol=1e-2) |
| `test_transformer_lm_truncated_input` | 截断输入处理                          |

**结构**:

```
x = TokenEmbedding(input_ids)
for each layer in num_layers:
    x = TransformerBlock(x)
x = RMSNorm(x)
logits = OutputProjection(x)
```

**运行测试**:

```bash
uv run pytest -k test_transformer_lm -v
```

---

### 2.12 资源计算 (Problem: transformer_accounting) - 5分

**提交形式**: 书面回答 (writeup.pdf)

**任务**:

- 计算GPT-2 XL参数量和内存
- 列出所有矩阵乘法及FLOPs
- 分析各组件FLOPs占比
- 对比不同模型规模

---

## 第三部分：训练组件

### 3.1 Cross-Entropy (Problem: cross_entropy) - 2分

**实现adapter**: `tests/adapters.py` 中的 `run_cross_entropy`

```python
def run_cross_entropy(
    logits: Float[Tensor, "... vocab_size"],
    targets: Int[Tensor, "..."]
) -> Float[Tensor, "..."]:
    """交叉熵损失"""
    raise NotImplementedError
```

**pytest测试** (`tests/test_nn_utils.py`):

| 测试函数             | 验证内容                                      |
| -------------------- | --------------------------------------------- |
| `test_cross_entropy` | 与PyTorch F.cross_entropy结果一致 (atol=1e-4) |
|                      | 验证数值稳定性 (输入\*1000)                   |

**公式**:

```
ℓ_i = -log softmax(o_i)[x_{i+1}]
```

**要求**:

- 减最大值以数值稳定
- 抵消log/exp
- 处理任意batch维度

**Perplexity (困惑度)**:

```
perplexity = exp(1/m * sum(ℓ_i))
```

**运行测试**:

```bash
uv run pytest -k test_cross_entropy -v
```

---

### 3.2 SGD Optimizer Example - 学习优化器实现

**目的**: 学习如何继承`torch.optim.Optimizer`实现自定义优化器

**示例SGD实现** (学习率衰减版本):

```python
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1
        return loss
```

**关键要点**:

- `__init__`: 接收params和超参数，传给基类构造函数
- `step`: 遍历param_groups，对每个参数原地更新
- `self.state[p]`: 存储每个参数的状态

---

### 3.3 Learning Rate Tuning (Problem: learning_rate_tuning) - 1分

**任务**: 在toy SGD示例上测试不同学习率的影响

**提交形式**: 书面回答 (writeup.pdf)

**要求**:

- 使用学习率 1e1, 1e2, 1e3 各运行10次迭代
- 观察loss是更快下降、更慢下降还是发散

---

### 3.4 AdamW (Problem: adamw) - 2分

**实现adapter**: `tests/adapters.py` 中的 `get_adamw_cls`

```python
def get_adamw_cls() -> type[torch.optim.Optimizer]:
    """返回AdamW优化器类"""
    raise NotImplementedError
```

**pytest测试** (`tests/test_optimizer.py`):

| 测试函数     | 验证内容                                      |
| ------------ | --------------------------------------------- |
| `test_adamw` | 与PyTorch AdamW或参考实现结果一致 (atol=1e-4) |

**要求**:

- 继承torch.optim.Optimizer
- 实现AdamW算法 (参考论文Algorithm 2)
- 保持m和v的state

**运行测试**:

```bash
uv run pytest -k test_adamw -v
```

---

### 3.5 AdamW资源计算 (Problem: adamwAccounting) - 2分

**提交形式**: 书面回答 (writeup.pdf)

**任务**:

(a) 计算AdamW运行所需峰值内存

- 分解为: 参数、激活、梯度、优化器状态
- 用batch_size和模型超参数表示
- 简化考虑的激活组件: Transformer block (RMSNorm, MHA子层, FFN), 最终RMSNorm, 输出embedding, cross-entropy

(b) 对GPT-2 XL形状模型，求最大batch_size (80GB内存限制)

(c) AdamW一步需要多少FLOPs?

(d) Model FLOPs Utilization (MFU) 计算

- 假设50% MFU，A100上训练GPT-2 XL 400K步需要多少天?
- 假设backward FLOPs是forward的2倍

---

### 3.6 Cosine Schedule (Problem: learning_rate_schedule) - 2分

**实现adapter**: `tests/adapters.py` 中的 `run_get_lr_cosine_schedule`

```python
def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int
) -> float:
    """余弦学习率调度"""
    raise NotImplementedError
```

**pytest测试** (`tests/test_optimizer.py`):

| 测试函数                      | 验证内容                 |
| ----------------------------- | ------------------------ |
| `test_get_lr_cosine_schedule` | 与预期学习率序列完全一致 |

**预期学习率序列** (warmup=7, cosine=21):

```
[0, 0.143, 0.286, 0.429, 0.571, 0.714, 0.857, 1.0, 0.989, 0.955, 0.902, 0.831,
 0.745, 0.650, 0.55, 0.450, 0.355, 0.270, 0.198, 0.145, 0.111, 0.1, 0.1, 0.1, 0.1]
```

**运行测试**:

```bash
uv run pytest -k test_get_lr_cosine_schedule -v
```

---

### 3.7 Gradient Clipping (Problem: gradient_clipping) - 1分

**实现adapter**: `tests/adapters.py` 中的 `run_gradient_clipping`

```python
def run_gradient_clipping(
    parameters: list[torch.nn.Parameter],
    max_norm: float
) -> None:
    """梯度裁剪 (原地修改)"""
    raise NotImplementedError
```

**pytest测试** (`tests/test_nn_utils.py`):

| 测试函数                 | 验证内容                                      |
| ------------------------ | --------------------------------------------- |
| `test_gradient_clipping` | 与PyTorch clip*grad_norm*结果一致 (atol=1e-6) |

**公式**: 如果 ||g||₂ > M，则 g = g \* M/(||g||₂ + ε)

**运行测试**:

```bash
uv run pytest -k test_gradient_clipping -v
```

---

### 3.8 Data Loading (Problem: data_loading) - 2分

**实现adapter**: `tests/adapters.py` 中的 `run_get_batch`

```python
def run_get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """从数据集采样batch"""
    raise NotImplementedError
```

**数据格式**:

- 输入: 整数token ID的numpy数组
- 输出: (x, y) 两个tensor，shape均为(batch_size, context_length)
- y是x的下一个token (y[i, j] = x[i, j+1])

**大数据集处理**:

- 使用`np.memmap`或`np.load(mmap_mode='r')`延迟加载
- 避免将整个数据集加载到内存
- 注意指定正确的dtype (如uint16)

**CPU/MPS设备**:

- CPU: 使用`'cpu'`
- Apple Silicon: 使用`'mps'`

**pytest测试** (`tests/test_data.py`):

| 测试函数         | 验证内容                               |
| ---------------- | -------------------------------------- |
| `test_get_batch` | shape检查 (batch_size, context_length) |
|                  | y = x + 1 验证                         |
|                  | 随机采样均匀分布                       |
|                  | 无效device报错                         |

**要求**:

- 随机起始位置
- y是x的下一个token
- 支持numpy memmap

**运行测试**:

```bash
uv run pytest -k test_get_batch -v
```

---

### 3.9 Checkpointing (Problem: checkpointing) - 1分

**实现adapter**: `tests/adapters.py` 中的 `run_save_checkpoint` 和 `run_load_checkpoint`

```python
def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO
) -> None:
    """保存检查点"""
    raise NotImplementedError

def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    """加载检查点，返回iteration"""
    raise NotImplementedError
```

**pytest测试** (`tests/test_serialization.py`):

| 测试函数             | 验证内容                         |
| -------------------- | -------------------------------- |
| `test_checkpointing` | 保存/加载后模型和优化器state一致 |

**运行测试**:

```bash
uv run pytest -k test_checkpointing -v
```

---

### 3.10 训练脚本 (Problem: training_together) - 4分

**提交形式**: 代码 + 实验结果

**要求**:

- 命令行参数配置
- 内存高效加载 (np.memmap)
- Checkpoint保存
- 日志记录 (wandb可选)

---

## 第四部分：文本生成与实验

### 4.1 Text Decoding (Problem: decoding) - 3分

**提交形式**: 代码实现

**要求**:

- 从prompt开始生成 (输入 x\_{1:t}，采样直到遇到 `<|endoftext|>`)
- 用户可控制最大生成token数
- Temperature scaling:
  ```
  softmax(v, τ)_i = exp(v_i/τ) / Σ exp(v_j/τ)
  ```

  - τ→0: 最大元素主导，softmax变为one-hot
- Top-p (nucleus) sampling:
  ```
  P(x_{t+1}=i|q) = q_i / Σ_{j∈V(p)} q_j  if i ∈ V(p)
                  = 0                     otherwise
  ```

  - V(p)是概率累计≥p的最小token集合
  - 通过排序后选取实现

**解码公式**:

```
P(x_{t+1}=i|x_{1:t}) = exp(v_i) / Σ exp(v_j)
v = TransformerLM(x_{1:t})_t ∈ R^{vocab_size}
```

---

### 4.2 Experiment Logging (Problem: experiment_log) - 3分

**提交形式**: 代码 + 实验日志

**要求**: 创建实验跟踪基础设施，记录梯度步数和墙钟时间对应的loss曲线

---

### 4.3 TinyStories基准超参数

**固定超参数**:

| 参数           | 值          | 说明                              |
| -------------- | ----------- | --------------------------------- |
| vocab_size     | 10000       | 典型值tens到hundreds of thousands |
| context_length | 256         | 简单数据集可用较短序列            |
| d_model        | 512         | 比标准768略小，加速               |
| d_ff           | 1344        | ≈8/3 × d_model, 64的倍数          |
| RoPE theta     | 10000       |                                   |
| num_layers     | 4           |                                   |
| num_heads      | 16          |                                   |
| total_tokens   | 327,680,000 | batch × steps × context_length    |

**需调试的超参数**: 学习率、warmup、AdamW参数(β₁, β₂, ε)、weight decay

**预期运行时间**: 30-40分钟 (H100)

**模型规模**: ~17M non-embedding parameters

**调试技巧**:

- 先overfit单个minibatch验证实现正确性
- 使用debugger检查中间tensor形状
- 监控activation、weight、gradient的norm防止爆炸/消失

---

### 4.4 Learning Rate Tuning (Problem: learning_rate) - 3分 (4 H100 hrs)

**提交形式**: 学习曲线 + 模型

**任务**:

(a) 对学习率进行超参数扫描，报告最终loss (或记录发散情况)

- **提交**: 多个学习率的学习曲线 + 超参数搜索策略说明

(b) 研究"边缘稳定性"——最佳学习率与发散点的关系

- **提交**: 包含至少一个发散run的学习曲线 + 分析

**要求**:

- 验证loss ≤ 1.45 (H100) 或 ≤ 2.00 (CPU/MPS)

**低资源配置 (CPU/MPS)**:

- 总tokens: 40,000,000 (而非327,680,000)
- batch_size × total_steps × context_length = 32 × 5000 × 256 = 40,960,000
- M3 Max (36GB RAM): ~36分钟 (MPS), ~1小时22分钟 (CPU)
- 验证loss约1.80 (step 5000)

**技巧**:

- cosine schedule应在最后一步恰好达到min learning rate
- MPS上不要使用TF32 (torch.set_float32_matmul_precision)
- 可用torch.compile加速:
  - CPU: `model = torch.compile(model)`
  - MPS: `model = torch.compile(model, backend="aot_eager")`

---

### 4.5 Batch Size Experiment (Problem: batch_size_experiment) - 1分 (2 H100 hrs)

**任务**: 从batch_size=1到GPU内存上限，测试不同batch size的影响

**提交形式**:

- 不同batch size的学习曲线 (必要时重新调学习率)
- 几句话讨论batch size对训练的影响

---

### 4.6 Text Generation (Problem: generate) - 1分

**提交形式**: 生成的文本 (≥256 tokens) + 简短评论

**要求**:

- 使用训练好的checkpoint生成文本
- 可能需要调整temperature、top-p等参数获得流畅输出
- 评论输出的流畅性 + 影响输出质量的至少两个因素

**TinyStories参考输出** (327M tokens训练):

```
Once upon a time, there was a pretty girl named Lily. She loved to eat gum, especially the big black one. One day, Lily's mom asked her to help cook dinner...
```

**低资源配置输出** (40M tokens训练):

```
Once upon a time, there was a little girl named Sue. Sue had a tooth that she loved very much. It was his best head...
```

---

### 4.7 消融实验 - 4分 (共4个实验，每个1 H100 hr)

#### 4.7.1 Layer Norm Ablation (Problem: layer_norm_ablation) - 1分

**任务**: 移除所有RMSNorm，观察训练行为

**提交形式**:

- 移除RMSNorm的学习曲线 vs 最佳学习率的学习曲线
- 几句话评论RMSNorm的影响

#### 4.7.2 Pre-norm vs Post-norm (Problem: pre_norm_ablation) - 1分

**Pre-norm (我们使用的)**:

```
z = x + MultiHeadSelfAttention(RMSNorm(x))
y = z + FFN(RMSNorm(z))
```

**Post-norm (原始Transformer)**:

```
z = RMSNorm(x + MultiHeadSelfAttention(x))
y = RMSNorm(z + FFN(z))
```

**提交形式**: Post-norm vs Pre-norm 学习曲线对比

#### 4.7.3 No Position Embedding (Problem: no_pos_emb) - 1分

**任务**: 完全移除位置嵌入 (RoPE)，观察模型性能

**背景**: 有研究表明causal mask本身可以隐式推断位置信息

**提交形式**: RoPE vs NoPE 学习曲线对比

#### 4.7.4 SwiGLU vs SiLU (Problem: swiglu_ablation) - 1分

**SwiGLU**:

```
FFN(x) = W2(SiLU(W1 x) ⊙ W3 x), d_ff ≈ 8/3 * d_model
```

**SiLU (无GLU)**:

```
FFN_SiLU(x) = W2(SiLU(W1 x)), d_ff = 4 * d_model
```

**提交形式**:

- 参数量近似匹配的学习曲线对比
- 几句话讨论发现

---

### 4.8 OpenWebText Experiment (Problem: main_experiment) - 2分 (3 H100 hrs)

**提交形式**: 学习曲线 + 生成文本

**任务**: 用相同的模型架构和训练迭代次数，在OpenWebText上训练

**要求**:

- 描述TinyStories和OpenWebText loss的差异及解释
- 生成的文本示例 + 流畅性分析
- 解释为何同样模型和计算量，输出质量不如TinyStories

**注意**: 可能需要重新调参 (学习率、batch size等)

---

### 4.9 Leaderboard (Problem: leaderboard) - 6分 (10 H100 hrs, 可选)

**规则**:

- 最多1.5小时H100训练
- 仅使用OpenWebText数据
- 验证loss < 5.0

**提交**: https://github.com/stanford-cs336/assignment1-basics-leaderboard

**改进思路**:

- 参考Llama 3、Qwen 2.5等开源模型
- NanoGPT speedrun repository的优化技巧
- 权重共享 (tie input/output embeddings) - 需减小embedding初始化std

---

## 测试容差说明

作业使用 **snapshot testing**，容差设置：

| 组件              | 绝对容差 (atol) | 相对容差 (rtol) |
| ----------------- | --------------- | --------------- |
| Linear/Embedding  | -               | -               |
| RMSNorm           | 1e-6            | -               |
| SwiGLU            | 1e-5            | -               |
| Softmax           | 1e-6            | -               |
| RoPE              | 1e-6            | -               |
| SDPA              | 1e-6            | -               |
| MHA               | 1e-6            | -               |
| Transformer Block | 1e-6            | -               |
| Transformer LM    | 1e-4            | 1e-2            |
| Cross-entropy     | 1e-4            | -               |
| AdamW             | 1e-4            | -               |
| Gradient Clipping | 1e-6            | -               |

---

## 完整测试运行

```bash
# 运行所有测试
uv run pytest

# 按文件运行
uv run pytest tests/test_tokenizer.py
uv run pytest tests/test_train_bpe.py
uv run pytest tests/test_nn_utils.py
uv run pytest tests/test_model.py
uv run pytest tests/test_optimizer.py
uv run pytest tests/test_data.py
uv run pytest tests/test_serialization.py

# 按组件运行
uv run pytest -k "linear or embedding"
uv run pytest -k "softmax or cross_entropy"
uv run pytest -k "rope or attention or transformer"
```

---

## 关键技术点总结

### 禁止使用

- `torch.nn` 中的具体层 (除 Parameter, Module, ModuleList等容器)
- `torch.nn.functional` 中的函数
- `torch.optim` 中的优化器实现

### 权重形状约定

- Linear: `[d_out, d_in]`
- Embedding: `[vocab_size, d_model]`
- 输出投影: `[d_model, vocab_size]`

### 数值稳定性

- Softmax: 减最大值
- RMSNorm: 升到float32
- Cross-entropy: 减最大值，抵消log/exp

### 性能优化

- 多头注意力: 单次矩阵乘法处理所有头
- BPE训练: 增量更新pair counts
- 预计算RoPE的sin/cos

---

## 问题总表与分值

### 第一部分：BPE分词器 (38分)

| Problem               | 分值 | 类型     |
| --------------------- | ---- | -------- |
| unicode1              | 1    | 书面回答 |
| unicode2              | 3    | 书面回答 |
| train_bpe             | 15   | 代码实现 |
| train_bpe_tinystories | 2    | 书面回答 |
| train_bpe_expts_owt   | 2    | 书面回答 |
| tokenizer             | 15   | 代码实现 |
| tokenizer_experiments | 4    | 书面回答 |

### 第二部分：Transformer架构 (24分)

| Problem                      | 分值 | 类型     |
| ---------------------------- | ---- | -------- |
| linear                       | 1    | 代码实现 |
| embedding                    | 1    | 代码实现 |
| rmsnorm                      | 1    | 代码实现 |
| positionwise_feedforward     | 2    | 代码实现 |
| softmax                      | 1    | 代码实现 |
| rope                         | 2    | 代码实现 |
| scaled_dot_product_attention | 5    | 代码实现 |
| multihead_self_attention     | 5    | 代码实现 |
| transformer_block            | 3    | 代码实现 |
| transformer_lm               | 3    | 代码实现 |
| transformer_accounting       | 5    | 书面回答 |

### 第三部分：训练组件 (15分)

| Problem                | 分值 | 类型     |
| ---------------------- | ---- | -------- |
| cross_entropy          | 2    | 代码实现 |
| learning_rate_tuning   | 1    | 书面回答 |
| adamw                  | 2    | 代码实现 |
| adamwAccounting        | 2    | 书面回答 |
| learning_rate_schedule | 2    | 代码实现 |
| gradient_clipping      | 1    | 代码实现 |
| data_loading           | 2    | 代码实现 |
| checkpointing          | 1    | 代码实现 |
| training_together      | 4    | 代码实现 |

### 第四部分：实验与生成 (23分 + 6分可选)

| Problem               | 分值 | 类型      |
| --------------------- | ---- | --------- |
| decoding              | 3    | 代码实现  |
| experiment_log        | 3    | 代码+日志 |
| learning_rate         | 3    | 实验      |
| batch_size_experiment | 1    | 实验      |
| generate              | 1    | 实验      |
| layer_norm_ablation   | 1    | 实验      |
| pre_norm_ablation     | 1    | 实验      |
| no_pos_emb            | 1    | 实验      |
| swiglu_ablation       | 1    | 实验      |
| main_experiment       | 2    | 实验      |
| leaderboard (可选)    | 6    | 比赛      |

**总分**: 100分 (不含可选) + 6分可选
