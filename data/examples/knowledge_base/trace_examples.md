# Trace Examples

## Response 正轨迹

对需求 whenever request occurs, grant should eventually occur，正轨迹可以是：t0 request，t1 空，t2 grant。

## Response 反轨迹

对 response 模式，反轨迹可以是：t0 request，t1 空，t2 空。因为 request 发生后直到有限轨迹结束都没有 grant。

## Response 边界轨迹

边界轨迹至少保留两条不同有限轨迹。例如：

- t0 request 与 grant 同时发生。
- t0 空，t1 空，t2 request 与 grant 同时发生。

它们共同用于区分 G(request -> F grant) 与 G(request -> X F grant)，并避免补全阶段只依赖单一边界样本。
