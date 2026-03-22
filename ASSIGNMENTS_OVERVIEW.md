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

### 2.4 SwiGLU (Problem: swiglu) - 2分

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

**运行测试**:

```bash
uv run pytest -k test_cross_entropy -v
```

---

### 3.2 AdamW (Problem: adamw) - 2分

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

### 3.3 AdamW资源计算 (Problem: adamwAccounting) - 2分

**提交形式**: 书面回答 (writeup.pdf)

---

### 3.4 Cosine Schedule (Problem: learning_rate_schedule) - 2分

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

### 3.5 Gradient Clipping (Problem: gradient_clipping) - 1分

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

### 3.6 Data Loading (Problem: data_loading) - 2分

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

### 3.7 Checkpointing (Problem: checkpointing) - 1分

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

### 3.8 训练脚本 (Problem: training_together) - 4分

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

- 从prompt开始生成
- 支持temperature采样
- 支持top-p (nucleus)采样
- 遇到<|endoftext|>停止

---

### 4.2 Experiment Logging (Problem: experiment_log) - 3分

**提交形式**: 代码 + 实验日志

---

### 4.3 Learning Rate Tuning (Problem: learning_rate) - 3分

**提交形式**: 学习曲线 + 模型

**要求**:

- 验证loss ≤ 1.45 (H100) 或 ≤ 2.00 (CPU/MPS)
- 分析"边缘稳定性"

---

### 4.4 Batch Size Experiment (Problem: batch_size_experiment) - 1分

**提交形式**: 学习曲线 + 分析

---

### 4.5 Text Generation (Problem: generate) - 1分

**提交形式**: 生成的文本 (≥256 tokens)

---

### 4.6 消融实验 - 4分

| Problem               | 消融内容              |
| --------------------- | --------------------- |
| `layer_norm_ablation` | 移除RMSNorm           |
| `pre_norm_ablation`   | Pre-norm vs Post-norm |
| `no_pos_emb`          | RoPE vs NoPE          |
| `swiglu_ablation`     | SwiGLU vs SiLU        |

---

### 4.7 OpenWebText Experiment (Problem: main_experiment) - 2分

**提交形式**: 学习曲线 + 生成文本

---

### 4.8 Leaderboard (Problem: leaderboard) - 6分 (可选)

**规则**:

- 最多1.5小时H100训练
- 仅使用OpenWebText数据
- 验证loss < 5.0

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
