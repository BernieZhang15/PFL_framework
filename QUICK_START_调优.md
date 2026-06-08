# FedFourierFT Non-IID CIFAR-100 快速调优指南

## 🚀 快速开始（3步）

### Step 1: 修改脚本参数
编辑 `fourier_cifar100.sh`，将：
```bash
$PY "$MAIN" -m "Fourier_bayes_cnn" -data "$ds" -lr "$LR" -fr 1 -sl 0 -nb "$NB"
```

改为：
```bash
$PY "$MAIN" -m "Fourier_bayes_cnn" -data "$ds" \
    -lr 0.001 \
    -fr 0.5 \
    -sl 0.5 \
    -nb "$NB" \
    -ls 20 \
    -gr 1000 \
    -eg 50
```

### Step 2: 运行训练
```bash
cd system
bash fourier_cifar100.sh
```

### Step 3: 监控结果
在TensorBoard中查看：
```bash
tensorboard --logdir=runs
```

---

## 📋 参数调整说明

### **最关键的3个参数**

#### 1️⃣ **学习率 (-lr)**: `0.0005` → `0.001`
**为什么**：
- Non-IID数据分布差异大，梯度方向不稳定
- 需要更强的更新来跳过局部最优
- 小学习率会导致收敛缓慢甚至停滞

#### 2️⃣ **频率比例 (-fr)**: `1.0` → `0.5`
**为什么**：
- 频率分量从1024→512
- 减少模型复杂度，降低过拟合风险
- 保留重要的低频和中频成分

**代码位置** [FourierFTModel.py#L138-142](FourierFTModel.py#L138-142)：
```python
# 当前
base_freq1, base_freq2, base_freq3 = 1024, 1024, 512

# 当 freq_ratio=0.5 时自动变成
n_freq1 = 1024 * 0.5 = 512
n_freq2 = 1024 * 0.5 = 512
n_freq3 = 512 * 0.5 = 256
```

#### 3️⃣ **谱规则化 (-sl)**: `0` → `0.5`
**为什么**：
- 防止高频分量无限增长
- 稳定集成学习
- 提高模型的泛化能力

**代码位置** [clientFourierFT.py#L37](clientFourierFT.py#L37)：
```python
# 当前使用
loss += self.spec_lambda * spec_loss + lambda_kl * kl / self.train_samples
```

---

## 🔧 如果效果仍未改善，依次尝试：

### 方案A: 进一步增加学习率
```bash
-lr 0.002  # 增大2倍
```

### 方案B: 更激进地压缩频率
```bash
-fr 0.3    # 频率分量减至30%
-sl 1.0    # 更强的规则化
```

### 方案C: 修改代码增加集成数
编辑 `FourierFTModel.py` 第136行：
```python
# 原来
def __init__(self, in_features=3, num_classes=10, ens_num=4, ...):

# 改为
def __init__(self, in_features=3, num_classes=10, ens_num=8, ...):
```

同时运行：
```bash
-lr 0.001 -fr 0.5 -sl 0.5
```

---

## 📊 期望结果

| 指标 | 原始配置 | 预期改善后 |
|------|---------|-----------|
| 收敛速度 | 慢 | 快（更早达到高准确率） |
| 最终准确率 | 低 | +1-3% |
| ECE | 高 | -10-20% |
| 训练稳定性 | 不稳定 | 稳定 |

---

## 🧪 实验跟踪

建议记录：
```
日期: 2026-01-21
版本: v1 (LR=0.001, FR=0.5, SL=0.5)
轮数: 500
最终准确率: ???
最终ECE: ???
Notes: ...
```

---

## ⚠️ 常见问题

**Q: 为什么准确率下降了？**
A: 可能学习率过大，尝试 0.001 而非 0.002

**Q: ECE反而增加了？**
A: 增加 spec_lambda 的值（0.5 → 1.0）

**Q: 训练速度变慢了？**
A: 这是正常的，因为计算了谱规则化项。如果不可接受，可将 -sl 减小到 0.3

**Q: 什么时候应该停止调整？**
A: 当准确率不再上升3个连续评估轮次时，或达到理想性能目标

---

## 📝 调优日志记录

建议创建文件 `tune_log.txt`：
```
[版本v0] 原始参数
-lr 0.0005 -fr 1.0 -sl 0
最终准确率: XX%
最终ECE: XX
备注: 性能不佳

[版本v1] 基础调整 (推荐首先尝试)
-lr 0.001 -fr 0.5 -sl 0.5
最终准确率: XX%
最终ECE: XX
备注: 性能改善情况

...
```

---

## 💡 高级调优提示

### 1. 学习率热身 (Learning Rate Warm-up)
在 `main.py` 中添加：
```python
parser.add_argument('-warmup', type=int, default=10,
                   help='学习率预热轮数')
```

### 2. 动态调整频率
根据训练进度动态调整 `spec_lambda`：
```python
# 前期加入更强的规则化，后期逐步减弱
if global_round < 500:
    spec_lambda = 1.0
else:
    spec_lambda = 0.5
```

### 3. 客户端采样策略
修改 `main.py`：
```bash
-jr 0.3    # 提高每轮参与客户端比例从0.2→0.3
-cdr 0.0   # 降低客户端掉线率
```

---

## 📞 技术支持清单

如果调整后仍未改善，检查：
- [ ] 是否正确保存了修改
- [ ] 日志中是否有错误信息
- [ ] GPU内存是否足够（ens_num=8会增加内存需求）
- [ ] 数据集是否正确加载（CIFAR-100 vs CIFAR-10）
- [ ] 是否有其他进程占用GPU

---

**建议**: 从 **v1 配置**开始运行，观察至少100个全局轮次（或500轮），再决定是否进一步调整。
