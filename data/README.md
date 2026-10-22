# Traj-LTLf Data: Input, Examples, Output

参照 DSVA-Framework 的数据布局组织自然语言需求、通用知识和运行结果。完整评估数据放在根目录，实际运行读取 `input` 下不含参考答案的 XLSX 文件。

## 🌟 Overview

| 数据集 | 需求数 | 在线输入 | 评估表工作表 |
|---|---:|---|---|
| A72 | 72 | [A72.xlsx](input/A72.xlsx) | `A72` |
| B240 | 240 | [B240.xlsx](input/B240.xlsx) | `B240` |
| C1026 | 1,026 | [C1026.xlsx](input/C1026.xlsx) | `C1026` |
| 合计 | 1,338 | | |

原始顺序、ID、需求原文和参考公式均保留，未对重复自然语言条目去重；相同原文可能对应不同的来源 ID。`example.xlsx` 另含两条演示需求，不计入上述数据集总数。

## 📁 Structure

```text
data/
├── LTLf_Evaluation_Dataset.xlsx  # 完整需求及参考公式，仅供离线比较
├── manifest.json                # 工作簿、原始来源与评估内容的摘要
├── README.md
├── input/
│   ├── example.xlsx             # 默认演示输入
│   ├── A72.xlsx
│   ├── B240.xlsx
│   ├── C1026.xlsx
│   ├── manifest.json            # 输入摘要、稳定 ID 与原文摘要
│   └── sources/                # 原始 Markdown 来源，仅供溯源
│       ├── aerospace_db/
│       ├── controll_db/
│       └── construct_db/
├── examples/
│   └── knowledge_base/          # 通用 LTLf 知识和说明，不是评估答案库
└── output/                     # 运行生成的公式、审核记录和断点
```

## 📊 Workbook Format

输入文件只有一个工作表 `Requirements`，每行一条需求，首行为表头：

| Natural Language | Requirement ID |
|---|---|
| Globally it is the case that if a holds then eventually a and b hold. | A72:REQ-000002 |

这两列分别是原始需求和稳定 ID。程序将 `A72:REQ-000002` 映射为运行记录中的 `A72-REQ-000002`，保留需求原文中的大小写、空格及换行。输入中的 AP 仍需人工审核，不由表格预先批准。

完整评估表 [LTLf_Evaluation_Dataset.xlsx](LTLf_Evaluation_Dataset.xlsx) 分为 `A72 / B240 / C1026` 三个工作表，包含四列：

| 列名 | 内容 |
|---|---|
| `Natural Language` | 与在线输入逐字一致的需求原文 |
| `Requirement ID` | 稳定来源 ID，用于关联输出与参考答案 |
| `Raw Formula` | 原始参考公式 |
| `Normalized Formula` | 规范化后的参考公式 |

例如 `A72:REQ-000002` 的参考式为 `G(a -> F(a & b))`。公式按文本保存，不作为 Excel 计算公式。完整评估表不作为在线审核输入；输入读取器拒绝多工作表或含额外参考列的工作簿。

## 🚀 Quick Start

在 `LTLf-main` 根目录配置好环境和 `.env` 后运行：

```powershell
# 使用默认 example.xlsx
python -B main.py --review manual --experiment-mode human --llm api --segmentation-review manual --plain-reading-review manual

# 审核 A72，将结果保存到独立运行目录
python -B main.py --review manual --experiment-mode human --llm api --segmentation-review manual --plain-reading-review manual --requirements-file data/input/A72.xlsx --run-dir data/output/A72
```

人工只审核表面切分、AP 和情景判断。结果写入所选运行目录的 `outputs/`，原始人工作答保存在 `review_records/`。现有运行结果保留，数据目录整理不会修改这些记录。

## 🔍 Reproducibility

```powershell
python -B scripts/prepare_inputs.py --verify
python -B scripts/check_reproducibility.py --check-dependencies
```

离线导出工具从完整评估表中只选择 `Natural Language / Requirement ID` 两列，确定性生成三份输入表及输入清单。改变参考答案列不会影响输入导出；在线运行只打开所指定的输入表和通用知识库。

缺少三份数据集输入文件时可运行 `python -B scripts/prepare_inputs.py --write` 重建；该命令拒绝覆盖已有不同内容。需要自定义需求时，复制输入表并另存为新文件，保持两列表头、唯一的文本 ID 以及纯文本单元格；也可继续使用自定义 Markdown 需求文件。

`input/sources` 保留 10 份原始来源文件，包括原始分段、类别及公式。它们的字节摘要与整理前一致，用于追溯数据来源，不属于默认在线输入或默认知识库。
