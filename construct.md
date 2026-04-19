# NCSN++（NCSNpp）架构解析

## 整体结构：带渐进式输入/输出的 U-Net

NCSN++ 是一个经典的 **对称 U-Net**，由 **下采样路径 → 瓶颈层 → 上采样路径** 三部分组成，并附带时间条件嵌入。本仓库在原版基础上做了若干增改（支持双时间条件、复数频域输入输出）。

---

## 1. 时间条件嵌入（Time Embedding）

| 分支 | 作用 |
|---|---|
| **t 嵌入** | `GaussianFourierProjection(nf)` → `Linear(2*nf, 4*nf)` → SiLU → `Linear(4*nf, 4*nf)` |
| **d = t−r 嵌入**（新增） | 独立的 `GaussianFourierProjection` → 两层 Linear，与 t 嵌入**相加** |

- `GaussianFourierProjection`：将标量时间 t 映射为 [sin(2πWt), cos(2πWt)]，维度 2×nf，其中 W 是冻结的随机高斯权重。
- 最终得到维度为 4×nf = 512 的 `temb` 向量，注入到每个 ResNet Block 中。

---

## 2. 输入预处理

```
x ∈ complex → [x_real, x_imag, y_real, y_imag]  (4 通道)
         ↓
   Conv3×3: 4ch → nf(128)ch
```

---

## 3. 下采样路径（Encoder / Down Path）

配置：`ch_mult = (1,1,2,2,2,2,2)`，共 **7 个分辨率级别**，每级 `num_res_blocks=2` 个 ResNet Block。

每个分辨率级别：

```
┌─ ResnetBlockBigGAN(in_ch, out_ch=nf*ch_mult[i], temb) ─┐
│  GroupNorm → SiLU → Conv3×3                              │
│  + temb (Dense 投影后加偏置)                               │
│  GroupNorm → SiLU → Dropout → Conv3×3                    │
│  + skip_rescale 残差: (x+h)/√2                           │
└──────────────────────────────────────────────────────────┘
× num_res_blocks

[可选] AttnBlockpp  (当 resolution ∈ attn_resolutions={16})
       GroupNorm → Q,K,V (NIN) → 全空间 self-attention → NIN → 残差

[下采样] ResnetBlockBigGAN(down=True)  // FIR 多相下采样2×
[可选] input_skip: pyramid_downsample(原始输入) + h  // 渐进式输入注入
```

所有中间特征 `hs` 被保存用于跳跃连接。

---

## 4. 瓶颈层（Bottleneck）

```
ResnetBlockBigGAN(in_ch, temb)
     ↓
AttnBlockpp(channels)       ← 唯一一处必加注意力
     ↓
ResnetBlockBigGAN(in_ch, temb)
```

---

## 5. 上采样路径（Decoder / Up Path）

与下采样对称，但每级有 `num_res_blocks + 1 = 3` 个 ResBlock（多一个用于融合跳跃连接）：

```
for each level (从最低分辨率到最高):
   × (num_res_blocks+1):
       concat([h, hs.pop()], dim=1)  ← U-Net 跳跃连接
       ResnetBlockBigGAN(in_ch + skip_ch, out_ch, temb)

   [可选] AttnBlockpp

   [渐进式输出 output_skip]:
       GroupNorm → SiLU → Conv3×3(in_ch → 4ch)
       pyramid = Upsample(pyramid) + 当前投影   ← 逐级聚合多尺度输出

   [上采样] ResnetBlockBigGAN(up=True)  // FIR 多相上采样2×
```

---

## 6. 输出层

```
h = pyramid                          # 渐进式输出聚合的结果 (4ch)
     ↓
Conv2d(4ch → 2ch, kernel=1×1)        # output_layer
     ↓
reshape → view_as_complex → (B,1,F,T)  # 返回复数频谱
```

---

## 7. 核心组件一览

| 组件 | 文件位置 | 说明 |
|---|---|---|
| `GaussianFourierProjection` | `layerspp.py` L34-42 | 标量→傅里叶特征 |
| `ResnetBlockBigGANpp` | `layerspp.py` L228-303 | 核心残差块，支持 up/down |
| `AttnBlockpp` | `layerspp.py` L62-86 | 全空间 channel-wise 自注意力 |
| `Upsample` / `Downsample` | `layerspp.py` L89-169 | FIR 多相采样 |
| `Combine` | `layerspp.py` L45-59 | 渐进式输入的跳跃合并 |

---

## 8. 简化示意图

```
Input (complex) → 4ch
  │
  ├── Conv3×3 → 128ch
  │
  ├── ══ Down Level 0 ══  (×2 ResBlock, ch=128)  ──┐ skip
  ├── ══ Down Level 1 ══  (×2 ResBlock, ch=128)  ──┤
  ├── ══ Down Level 2 ══  (×2 ResBlock, ch=256)  ──┤
  ├── ══ Down Level 3 ══  (×2 ResBlock, ch=256)  ──┤
  ├── ══ Down Level 4 ══  (×2 ResBlock, ch=256)  ──┤
  ├── ══ Down Level 5 ══  (×2 ResBlock, ch=256)  ──┤
  ├── ══ Down Level 6 ══  (×2 ResBlock, ch=256)  ──┤
  │                                                 │
  ├── Bottleneck: ResBlock → Attention → ResBlock   │
  │                                                 │
  ├── ══ Up Level 6 ══  (×3 ResBlock + skip) ◄──────┤
  ├── ══ Up Level 5 ══  (×3 ResBlock + skip) ◄──────┤
  ├──        ...                             ◄──────┤
  ├── ══ Up Level 0 ══  (×3 ResBlock + skip) ◄──────┘
  │
  ├── Progressive output aggregation → 4ch
  └── Conv1×1 → 2ch → complex output
```

---

## 总结

NCSN++ 本质是一个 **7 级、带 BigGAN 风格残差块 + FIR 多相采样 + 渐进式输入/输出 + 高斯傅里叶时间嵌入的 U-Net**，在此仓库中被改造为接收复数 STFT 频谱并输出复数预测，同时支持双时间条件 (t, d) 的注入。

---
---

# Flow Matching 条件速度场目标 v_target 解析

## 前置：前向插值路径

Flow Matching 定义了一条从 **干净语音** x_clean（t=0）到 **带噪语音** y（t=1）的线性插值路径：

$$\mu_t = (1-t) \cdot x_{\text{clean}} + t \cdot y$$

$$\sigma_t = (1-t) \cdot \sigma_{\min} + t \cdot \sigma_{\max}$$

$$x_t = \mu_t + \sigma_t \cdot z, \quad z \sim \mathcal{N}(0, I)$$

其中 x_t 是时刻 t 的"中间态"：均值沿 x_clean → y 线性移动，标准差从 σ_min 线性增大到 σ_max，z 是采样时固定的噪声。

---

## 速度场目标的推导

对 x_t = μ_t + σ_t · z 关于 t 求导：

$$v_{\text{target}} = \frac{dx_t}{dt} = \underbrace{\frac{d\mu_t}{dt}}_{\text{均值漂移}} + \underbrace{\frac{d\sigma_t}{dt} \cdot z}_{\text{扩散变化}}$$

代入具体表达式：

| 项 | 计算 | 结果 |
|---|---|---|
| dμ_t/dt | d/dt[(1-t) x_clean + t·y] | y − x_clean |
| dσ_t/dt | d/dt[(1-t)σ_min + t·σ_max] | σ_max − σ_min |

因此：

$$\boxed{v_{\text{target}} = (y - x_{\text{clean}}) + (\sigma_{\max} - \sigma_{\min}) \cdot z}$$

---

## 两项的物理含义

1. **(y − x_clean)** — **确定性漂移**：从干净到带噪的方向差异，即"噪声残差"。这一项与 t 和 z 无关，全程恒定。

2. **(σ_max − σ_min) · z** — **随机扩散贡献**：路径上标准差的变化率乘以噪声采样。当 σ_min=0, σ_max=0.487 时，这一项 = 0.487·z，为速度场增加了与特定噪声实例相关的分量。

---

## 训练时怎么用

网络的任务是：**给定 x_t、t、y，预测 v_target**。

推理时反向：从 t=T_rev（带噪端）用 ODE 积分回 t=t_ε（干净端），每步沿预测的速度场 v 走，逐步去噪：

$$x \leftarrow x - \Delta t \cdot v(x, t, y)$$

简言之，v_target 就是"沿插值路径上每一点 x_t 应该走的瞬时速度"，网络学会它之后就能反向积分完成语音增强。
