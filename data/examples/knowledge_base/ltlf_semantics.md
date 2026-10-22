# LTLf Semantics

## 有限轨迹语义

LTLf 公式解释在有限轨迹上。一条轨迹可以写成 t0、t1、t2 等时间步，每个时间步包含当前为真的原子命题。

## F 算子

F phi 表示 eventually phi。从当前位置到有限轨迹结束，只要某个位置满足 phi，就认为 F phi 成立。F 允许当前时刻成立。

## G 算子

G phi 表示 globally phi。从当前位置到有限轨迹结束，每个位置都必须满足 phi。

## X 算子

X phi 表示强 next。只有存在下一步并且下一步满足 phi 时才成立。在有限轨迹最后一个位置，X phi 为假。
