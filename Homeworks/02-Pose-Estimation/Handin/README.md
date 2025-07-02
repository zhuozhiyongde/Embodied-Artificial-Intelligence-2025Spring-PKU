# 02-Pose-Estimation

## Result

### Task 1

```
Error: trans 0.008078088983893394 rot 0.10565884014222496
```

### Task 2

```
Error: trans 0.005475909449160099 rot 0.08038749273792133
```

### Task 3

Estimate Pose:

```
Current success rate: 477/500 = 0.954
```

Esitimate Coordinates:

```
Without RANSAC:
Current success rate: 484/500 = 0.968

With RANSAC:
Current success rate: 479/500 = 0.958
```

我推测这可能是因为这个 Coord 预测已经非常准确，而使用 RANSAC 反而会降低准确性。

默认情况下，RANSAC 是禁用的。如果你想启用它，需要注释掉`src/model/est_coord.est`中的相关代码。

## Tricks

1. 为什么 PointNet 提取到每个点的全局特征之后，要做 `torch.max(x, 2, keepdim=True)[0]` 操作？
   因为我们期望对点云有一个轮换不变性，点云的顺序变化应当不影响最终的输出，通过 `max` 操作，我们实现了这一点。
2. 为什么 Est Coord 中使用了一个类似 U-Net 的特征传播？
   因为我们最重要预测每个点的坐标，这是一个非常密集（dense）的任务，通过将初级特征对应传到特征图中，我们可以更好地利用这些信息，也即记住最初的结构信息。
