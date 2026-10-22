# Temporal Patterns

## Response 模式

Whenever A occurs, B should eventually occur 通常对应 response 模式。受限骨架是 A -> B，补全后的候选公式包括 G(A -> B)、G(A -> X B)、G(A -> F B)、G(A -> X F B)。

## Response 语义线索

whenever 通常表示全局约束 G。eventually 通常表示 F。边界轨迹需要判断 A 与 B 同时发生时是否满足需求。

## Until 模式

如果自然语言显式出现 until、直到、在……之前一直，可以考虑 A U B。受限骨架阶段允许 U，但不能用 U 伪装 F 或 G。
