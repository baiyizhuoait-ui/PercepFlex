# PercepFlex 标号统一规范 v3 —— 全量版（含历史阶段）

> 状态：**已定案，待执行**（执行窗口 = Round 2 链完全结束之后）
> 决策人：用户（2026-09-10）："历史阶段的那些冲突的名字也要改，必须统一，改完后我再提交；
> 等 20ep 做完后统一改一遍。不要影响现在和未来的进程。"
> v1/v2 只覆盖 Phase 6 活跃阶段；**v3 取消历史冻结，Phase 0–5 全部纳入**。

---

## 0. 一句话

标号体系从"双轨/多形态并存"收敛为**一套四层语法 + 三类路径范式**；
除"非文本产物"与"命名工具自指"外，**没有任何历史名被豁免**。

---

## 1. 命名语法（唯一权威）

### 1.1 标号层（出现在正文 / 注释 / CSV 单元格）

| 层 | 规范形式 | 正则 | 反例（禁止） |
|---|---|---|---|
| 大阶段 | `Phase <n>` | `Phase [0-9]+` | `PHASE 6`（正文大写） |
| 子阶段／轮次 | `Phase <n><A-Z>` | `Phase [0-9]+[A-Z]\b` | `Round 1`、`Round-1`、`6R1` |
| 实验单元 | `EXP-<nn>`（全局唯一、零填充） | `EXP-[0-9]{2}` | `EXP-1`、`EXP1` |
| 阶段内实验引用 | `P<n><L>-EXP-<nn>` | `P[0-9]+[A-Z]-EXP-[0-9]{2}` | `4B-3`、`P4B/EXP-03` |
| 执行步 | `P<n><L>-STEP<n>`（可带子步字母） | `P[0-9]+[A-Z]-STEP[0-9]+[a-z]?` | `STEP1`、`STEP A`、`GPU-1` |
| 假设 | `H-<nn>` 可带小写后缀或 `-alt` | `H-[0-9]{2}[a-z]?(-alt)?` | `H1`、`H5b`、`H-A` |
| 闸门 | `GATE-<phase>.<n>` | `GATE-[0-9]+[A-Z]\.[0-9]+` | `GATE1`、`GATE-2` |

**关键规则（易错点）**
- 实验单元全局编号、**零填充两位**、跨阶段不复用。
- 阶段内引用必须带阶段限定（`P4B-EXP-03`），否则与全局 `EXP-03` 混淆。
- 执行步**必须带阶段前缀**：裸 `STEP1` 一律非法（历史上它是 Phase 5 闭链步，现为 `P5-STEP1`）。
- 假设编号**全局连续**：Phase 1–5 占 `H-01 … H-21`（含 `H-05a/H-05b/H-11-alt` 等子/变体后缀），
  Phase 6 占 `H-22 … H-35`（旧写 `H-A … H-N`）。
- 子阶段字母**只在有子阶段时使用**；无子阶段的 Phase（如 5）写作 `P5-STEP1`。

### 1.2 路径层

| 目录 | 范式 | 例 |
|---|---|---|
| 阶段报告 | `docs/PHASE<n><L>_<SLUG>.md` | `docs/PHASE4A_DECISION_REPORT.md` |
| 合体报告 | `docs/PHASE<n><L><L>_<SLUG>.md` | `docs/PHASE4BC_LEVEL2_DECISION_REPORT.md`（= 4B + 4C） |
| 配置 | `configs/phase<n><L>_<slug>.yaml` | `configs/phase4a_r2_z16.yaml` |
| 脚本 | `scripts/phase<n><L>_<slug>.{sh,py}` | `scripts/phase5_lane8_decide.py` |
| 实验输出 | `experiments/phase<n><L>/<slug>/` | `experiments/phase5/exp6_danc20/` |
| 非实验工具目录 | `experiments/_<slug>/` | `experiments/_template/`、`experiments/_figures/` |

**方向性规则**
- `docs/` 文件名用**大写** `PHASE…`（人读面）；`configs/`、`scripts/`、`experiments/` 用**小写** `phase…`（机器面）。
- 机器面文件名**禁止驼峰与多余大写**（`singleKD` → `singlekd`）。
- 报告一律归 `docs/`，**仓库根与 `experiments/` 根不得放报告**。

---

## 2. 冲突全量清单与处置（v3 新增部分以 ★ 标注）

| # | 冲突 | 处置 | 手段 | 状态 |
|---|---|---|---|---|
| C1 | `Phase 6 Round 1/2`、`Round-1` | → `Phase 6A/6B` | 文本 | ✅ v2 已完成 |
| C2 | `H-A … H-N` | → `H-22 … H-35`（`alias` 列回填） | 文本+CSV | ✅ v2 已完成 |
| C3 | `EXP-1 … EXP-8` | → `EXP-01 … EXP-08` | 文本 | ✅ v2 已完成 |
| C4 | `GPU-1/2/3` | → `P6A-STEP1/2/3` | 文本 | ✅ v2 已完成 |
| C5 | `GATE1/GATE2` | → `GATE-6A.1/.2` | 文本 | ✅ v2 已完成 |
| C6 | `STEP A–D` | → `P6A-STEP2a–d` | 文本 | ✅ v2 已完成 |
| C7 | `PHASE4BC_` 连写 | **规则化**（定义为合体语法，不再算冲突） | 规则 | ✅ v2 已完成 |
| ★C8 | `H1 … H21`（含 `H5a/H5b/H5c/H7b/H11-alt`） | → `H-01 … H-21`（后缀保留） | 文本 | ⏳ v3 |
| ★C9 | `4B-2 … 4B-6`、`4C-n` | → `P4B-EXP-02 … 06` / `P4C-EXP-0n` | 文本 | ⏳ v3 |
| ★C10 | `P4B/EXP-0n`（斜杠形） | → `P4B-EXP-0n`（连字符形） | 文本 | ⏳ v3 |
| ★C11 | 裸 `STEP1 … STEP5`（Phase 5 闭链） | → `P5-STEP1 … P5-STEP5` | 文本 | ⏳ v3 |
| ★C12 | `experiments/expA_*`、`expD_*`、`expF_*`、`expI_*` | 归位 `experiments/phase2b|2c/` | git mv + 引用同步 | ⏳ v3 |
| ★C13 | `experiments/exp_train_*`、`exp_001`、`revised`、`exp_static_vs_dynamic` 等 | 归位 `experiments/phase1b/` | git mv + 引用同步 | ⏳ v3 |
| ★C14 | 孤儿目录 `expB/expC/expE/expG_dynamic|latency|probe|multi_teacher`、`exp_smoke` | 归位 `experiments/phase1b/` | git mv | ⏳ v3 |
| ★C15 | `experiments/template`、`figures` | → `experiments/_template`、`_figures` | git mv | ⏳ v3 |
| ★C16 | 根级／experiments 根的报告（`PHASE1B_REPORT.md` 等 6 个） | 归 `docs/`（或模板目录） | git mv | ⏳ v3 |
| ★C17 | `scripts/` 约 40 个无阶段前缀脚本 | 加 `phase<n><L>_` 前缀 | git mv + 引用同步 | ⏳ v3 |
| ★C18 | `configs/train_stageA_*.yaml` 等 29 个 | → `configs/phase1b_train_stage_a_*.yaml` | git mv + 引用同步 | ⏳ v3 |
| ★C19 | `configs/ours_*.yaml`、`phase2_stageA_0.235M.yaml` | → `phase2_ours_*.yaml`（小写化） | git mv | ⏳ v3 |
| ★C20 | 机器面驼峰（`singleKD`） | 小写化 | 文本/路径 | ⏳ v3 |

### 2.1 明确**不动**的项（附理由，非豁免）

| 项 | 理由 |
|---|---|
| `configs/phase2b_zsweep_z*.yaml` **vs** `phase2b_r1_zsweep_z*.yaml` | 经查为**两轮不同实验**（z32/z64/z96 在两轮中同时存在，强行合并会互相覆盖）。登记于别名表，不视为同名冲突。 |
| 各阶段 `phase*_results.csv` 的 **cell 名**（如 `danc_z16`、`dp2b_z16`） | 是 `clean_master_chain.sh` 幂等守卫 `row_exists` 的**键**；改名会破坏"停后衔接"语义。登记为"实验变体 tag"词表。 |
| `docs/TASKBOOK.md` 内的任务书原文引用 | 任务书是外部给定的**只读输入**，改动会使引用与原件不符。 |

---

## 3. 冻结清单（v3 最小集）

`scripts/naming_frozen.txt` 已收缩为**两类**，不再包含任何"历史阶段"：

1. **非文本产物**：`.git/**`、`__pycache__`、`data/**`、`datasets/**`、`weights/**`、
   `trained_models/**`、`*.pt|pth|onnx|engine|png|jpg|pdf|tgz|zip`。
2. **命名工具自指**：`apply_naming_migration.py`、`rename_paths.py`、`check_naming.sh`、
   `verify_naming.py`、`naming_frozen.txt`、`naming_path_map.csv`、`NAMING_CONVENTION.md`、
   `NAMING_ALIASES.csv`（这些文件含"旧形态字面量"，自改写会破坏工具本身）。
3. **运行中链脚本**（执行器临时追加，链结束后自动解除）。

> 冻结从 v2 的 **16 条 / 覆盖 8 类历史冲突**，收缩到 v3 的 **2 类 + 自指**。
> 历史阶段的"只登记不改名"策略已被用户否决并撤销。

---

## 4. 执行序列（单次守卫脚本，链结束后自动跑）

脚本：`scripts/finalize_naming_v3.sh`

```
0. 等待 —— 只要 phase6_round2_chain.sh 或 training/train.py 存活就 sleep 60
   （满足"不影响现在和未来的进程"）
1. 部署 v3 工具链（frozen / migration / rename_paths / path_map / checker）
2. 全树备份 -> /tmp/naming_bak/v3_<HHMM>/tree.tgz
3. 文本迁移   apply_naming_migration.py --apply   (C8–C11, C20 + 幂等重跑 C1–C6)
4. 语法门禁   py_compile / bash -n 所有改动文件   → 任一失败即整体回滚
5. 路径重命名 rename_paths.py --apply             (C12–C19，最长路径优先)
   └─ 同步改写全仓引用，并校验每条引用可解析（悬空引用导致退出码 2）
6. 二次语法门禁（改名可能破坏 import）
7. 校验器     check_naming.sh  → 期望 RESULT: naming UNIFIED
8. 汇总 git status，**不提交**
```

**顺序不可交换**：文本迁移先于路径重命名，这样改名后引用同步能覆盖迁移器新写入的注释。

---

## 5. 验证协议（机器可判定）

| 关卡 | 判据 |
|---|---|
| G1 幂等 | `apply_naming_migration.py` 第二次运行 = 0 changes |
| G2 语法 | 所有改动 `.py` 过 `py_compile`；所有改动 `.sh` 过 `bash -n`；失败数 = 0 |
| G3 引用 | `rename_paths.py` 的 resolution check 无悬空引用 |
| G4 规范 | `check_naming.sh --strict` → `RESULT: naming UNIFIED`，exit 0 |
| G5 可运行性 | 抽查 `scripts/phase6_run.sh --help` 类干跑 + `training/train.py --help` 能启动 |
| G6 未越界 | `git status` 变更中**无** `data/`、`weights/`、`*.pt` 等冻结产物 |

---

## 6. 诚实声明的限制

1. **git 历史不可改写**：已推送的 commit message 与旧文件名永久留存在历史中。本次统一只作用于**工作树与后续提交**。若你要求历史也统一，只能 `git filter-repo` 重写历史并强推 —— 本方案**不做**（风险远大于收益）。
2. **C12–C14 的目录归属**有置信度差异：`expA`/`exp_train_*`/`exp_001` 有文档明文（HIGH）；
   `expD`/`expF`/`expI` 为同期推断（MED）；`expB/expC/expE/expG` 为孤儿目录、无引用（LOW，可安全归位但归属属推断）。
   映射表 `naming_path_map.csv` 的 `confidence` 与 `evidence` 两列已逐条标注。
3. **`row_exists` 兼容**：Phase 5 的 CSV 文件名与 cell 名保持不动，故"停后衔接"守卫不受本次统一影响。
4. **不改语义 slug**：`r2u/r2d/r2p/r3tp/danc/l14f1` 等变体缩写按原样保留，仅登记词表。
   在缺乏文档定义的证据时重写它们会引入语义错误。

---

## 7. push 指引（用户执行）

远程：`git@github.com:baiyizhuoait-ui/PercepFlex.git`（SSH）。
**默认 key 未被 GitHub 接受，必须指定 `id_ed25519_trac`**（已实测认证成功）。

```bash
# 在 WSL 中：
cd /home/mycode/ai_study/trac

git status                       # 审阅本次统一产生的变更
git diff --stat | tail -20       # 规模确认
git add -A
git commit -m "naming: full unification v3 (historical phases) + phase6 round2 artifacts"

GIT_SSH_COMMAND="ssh -i ~/.ssh/id_ed25519_trac -o IdentitiesOnly=yes" \
  git push origin main
```

**提交前请确认**（三条即可）：
1. `bash scripts/check_naming.sh --strict` → `RESULT: naming UNIFIED`；
2. `git status` 里没有 `data/`、`*.pt`、`weights/` 之类的产物被误改；
3. 抽查一个被改名的脚本仍可运行（如 `bash scripts/phase1b_train_pipeline.sh --help` 或
   `python training/train.py --help`）。

回滚兜底：`tar xzf /tmp/naming_bak/v3_<HHMM>/tree.tgz -C /home/mycode/ai_study/trac`

---

## 8. 本轮产物索引

| 文件 | 作用 |
|---|---|
| `_phase6/NAMING_CONVENTION.md` | 本规范（部署到 `docs/NAMING_CONVENTION.md`） |
| `_phase6/NAMING_PATH_MAP.csv` | 路径重命名映射（kind/old/new/confidence/evidence/action） |
| `_phase6/naming_frozen_v3.txt` | 最小冻结清单 |
| `_phase6/apply_naming_migration_v3.py` | 文本迁移器（幂等，含 C8–C11/C20 规则） |
| `_phase6/rename_paths.py` | 路径重命名 + 引用同步 + 悬空引用校验 |
| `_phase6/check_naming_v3.sh` | 11 项校验器 |
| `_phase6/finalize_naming_v3.sh` | 等待链结束 → 一键执行（含回滚） |

---

## 9. 执行期定稿（2026-09-10 16:00–16:20，v3 final）

### 9.1 STEP 作用域裁定 —— 本轮最关键的一次修正

`STEP` 是**跨阶段重载词**（Phase 2 / 3C / 4A / 4B / 5 都在用）。原先的全局规则
`STEP<n> → P5-STEP<n>` 会把 `Phase 4A STEP 5` 错标成 `Phase 4A P5-STEP5`
（已在 dry-run 中实际观察到）。裁定为**两级规则**：

| 级别 | 规则 | 例 |
|---|---|---|
| 1. 阶段限定形（优先） | `Phase\|PHASE <n><L> [·\-_ ]* STEP <k>` → `P<n><L>-STEP<k>` | `P4A-STEP5`、`P3C-STEP1`、`P5-STEP7b`、`P4B-STEP4` |
| 2. 裸形（按文件所属阶段作用域） | `SCOPE_PREFIX`：`experiments/phase{2,3c,4a,5}/`、`docs/PHASE<n>*`、`scripts/phase<n>_*` → 各自阶段前缀 | `P2-STEPn`、`P4A-STEPn` |
| 2b. **4B→5 闭链**（danc/lane8/da14 全在此） | `clean_master_chain.sh`、`night_master_chain.sh`、`recover_step2_then_chain.sh`、`phase4b_run.sh`、`experiments/phase5/*` → `P4B5-STEP<n>` | 链的 STEP1–5 |
| 3. Phase 5 自身步号 6/7（lane probe） | 其他阶段无用 6/7 号 → `P5-STEP6/7b/7c` | `P5-STEP7c` |

**依据**：闭链 STEP4=lane8（4B-5）、STEP5=da14（4B-6）在脚本注释中有明文 4B 归属，
而其产物落在 `experiments/phase5/` → 标为 `P4B5`（4B→5）最忠实。

### 9.2 执行期修掉的 2 个工具 bug

1. **迁移器遍历未剪枝 `data/`**（412,026 文件）→ 扫描看似卡死并被回收（SIGTERM）。
   加 `PRUNE_DIRS` + 「冻结前缀目录」剪枝后，全仓扫描 **5.3 秒**完成。
2. **手术规则非幂等**：`\bSTEP([12])_(...)` 的 `\b` 也会匹配已迁移的 `P2-STEP2_...`
   → 二次前缀成 `P2-P2-STEP2_...`。加负向先行断言修正，并清理 5 个受影响文件。

### 9.3 校验器修掉的 3 个 bug

1. `is_frozen` **极性反转**（约定 0=冻结，代码却 `return 1`）→ 冻结清单静默失效，
   别名表与工具自指全被当成违规。
2. `find` 未排除 `data/` 等重目录 → CHK6 在 BDD100K 的 RLE base64 串里产生 **434 条假命中**。
3. CHK11 路径正则改为**必须带文件扩展名**，消除截断 token 假阳性（25 → 少数真悬空）。

### 9.4 最终实测计数

| 关卡 | 结果 |
|---|---|
| 文本迁移 | **585 处 / 95 文件**（主轮）+ **11 处 / 9 文件**（残差）= **596 处 / 约 100 文件** |
| G1 幂等 | 再运行 = **0 changes** ✓ |
| G2 语法 | 46 个 `.py` 过 `py_compile`、11 个 `.sh` 过 `bash -n`，失败 0 ✓ |
| G4 校验 | CHK1–CHK9 **全绿**；CHK10 仅 WARN（configs 大小写，交由路径重命名处理） |
| G3 引用 | CHK11 余 21 条**全部为历史遗留**（19 条已在 HEAD 中被引用；2 条为正则伪影）→ **本次迁移引入 0 条新悬空引用** |
| 代码安全 | 改动仅落于注释 / docstring / 显示字符串 / 文件内自建字典键，**无一行可执行逻辑被改** |

### 9.5 与前一版的差异（诚实记录）

- 本条重写了 C03：**闭链步的前缀由 `P5-STEP` 改为 `P4B5-STEP`**（理由见 9.1）。
- 上一版曾把 `TASKBOOK.md` 列为"外部只读、不动"，本轮按"全部统一"改为**一并迁移**（`H1→H-01` 共 5 处）；
  它是仓库内的 `docs/` 文档，改名自洽、可随时回滚。
- 上一版的冻结原则（Phase 0–5 全冻结）已作废：冻结集**只剩**非文本产物与命名工具自指。


---

## 10. 收尾定稿（2026-09-10 提交前最终核验）

### 10.1 新增 `scripts/refs_pass2.py`（第二遍引用同步，存在性验证）

`rename_paths.py` 之后需要第二遍，原因有二：

1. **映射表在工作树里曾被一次退化重写污染**：仓库内的 `scripts/naming_path_map.csv` 一度出现
   **95/98 条 RENAME 行 `old_path == new_path`** 的自指形态（`load_map()` 因此只返回 **3** 条）。
   该副本已在提交前**用正本回滚**，现为 **98/98 条有效映射**（`load_map()` = 98）。
   → 教训：**不要把生成物当权威源**；本轮改为「以 git 为准 + 磁盘存在性验证」。
2. 即便映射表正确，它**也只记录"计划内"的文件级改名**，不覆盖引用实际指向的**未跟踪子路径**
   （`checkpoint.pt`、`*.log`、`*.txt` 等**从未入库**的运行时产物）——而这些正是悬空引用的主体。
3. `git diff --cached -M` 是唯一权威改名记录（221 条），但它的 rename detection 会把
   **内容字节相同的文件**跨实验配对，产出**伪目录映射**
   （实测：`experiments/expA_equal_budget/dynA_s1 -> experiments/phase1b/exp_train_a`）。
   直接使用会产生**错误的批量改写**。

因此 refs_pass2 **从不"信任后改写"**：
只收集**当前解析不到**的引用 → 用四族规则生成候选 → **仅保留磁盘上真实存在的候选** → 候选唯一时才改写。
规则序：R1 精确文件对 · R2 保基名目录对（含末段同名筛除，用于杀掉跨实验配对噪声）· R3 阶段目录插入 · R4 阶段前缀基名。

实测：45 条悬空引用 → **26 条**经磁盘验证可解析（落盘改写 38 处），**19 条**无任何已验证目标 → 判为历史引用（§10.2）。
**幂等复查：0 残留。**

### 10.2 CHK11 判据收紧（**判据变更，明确留痕**）

原 CHK11 把"所有解析不到的路径 token"一律 FAIL，混入两类**非改名缺陷**：

| # | 类别 | 实例 | 处置 |
|---|---|---|---|
| 1 | 扩展名被截断的**模块符号** | `models/representation/det_from_z.DEFAULT_ANCHORS_3S` 被 `\.[A-Za-z0-9]{2,5}` 匹配成 `.DEFAU` | 扩展名后加边界 `(?![A-Za-z0-9_])` |
| 2 | 指向**从未进入 git** 的运行时产物 | `experiments/phase5/exp5_watcher.log`、`experiments/phase2d/expD_summary.txt` 等 | 归入 CHK11w（WARN），清单落盘 |

第 2 类**逐条经 `git ls-tree HEAD` 核验：19 条全部 HEAD hits = 0**，即**从未存在于任何提交**。

新判据：

| 项 | 级别 | 定义 | 本轮读数 |
|---|---|---|---|
| CHK11 | **FAIL** | 存在"改名后**仍可解析到真实文件**的陈旧引用"（refs_pass2 逐条做磁盘存在性验证） | **0** |
| CHK11w | WARN | 指向从未入库产物的历史引用，清单 `scripts/naming_historical_refs.txt`（可审计，**不改写成臆造路径**） | 18 |

**这不是"改判据让检查变绿"**：真实缺陷（改名遗留）已全部修掉并归零；剩余 18 条与改名无关，
改写它们等于**编造路径**。判据变更在此留痕，供后续质疑与复算。

### 10.3 追加修掉的机器面大写（§1.3 违规项）

`configs/phase3c_midL_z32.yaml`、`configs/phase3c_midM_z32.yaml` 违反 §1.3「机器面文件名禁止驼峰与多余大写」。

- **文件名**：`git mv` → `phase3c_midl_z32.yaml` / `phase3c_midm_z32.yaml`，引用同步 2 处。
- **变体 tag**：`docs/PHASE3C_REPORT.md` 内的 `midL` / `midM` **保持不变** —— 依 §2.1，变体 tag 属冻结词表，不得改写。

两条规则管辖对象不同（**文件名** vs **变体 tag**），故不冲突。CHK10 由 WARN 转 **[OK]**。

### 10.4 新增文件的自我合规（记录以免后人误以为存在豁免）

本轮新增的 `phase6_round2_decision.md`、`phase6_round2_statistics.csv`、`scripts/phase6_round2_stats.py`
首次落盘时触发 CHK1（prose 误用别名 `H-M`）与 CHK4（prose 误用 `Round 1/2`）。
处置：**在源头改回规范形**（`H-34` / `Phase 6A|6B`），**未加任何豁免**。

### 10.5 最终校验

```
$ bash scripts/check_naming.sh
scanned: 899 live file(s), 37 frozen pattern(s)
[OK]   CHK1 .. CHK11
[WARN] CHK11w historical citation(s) to never-committed artefacts: 18
RESULT: naming UNIFIED
```

### 10.6 产物清单（v3 最终）

| 文件 | 作用 |
|---|---|
| `scripts/refs_pass2.py` | 第二遍引用同步（存在性验证，幂等） |
| `scripts/phase6_round2_stats.py` | Phase 6B 统计（析因交互项 + 等预算 + B/C 严格对照 + Pareto） |
| `experiments/phase6/phase6_round2_statistics.csv` | 上述统计的 tidy 输出 |
| `experiments/phase6/phase6_round2_decision.md` | Phase 6B 决策报告（EXP-07 / EXP-08 判定） |
| `scripts/naming_historical_refs.txt` | CHK11w 历史引用清单（18 条，生成物） |

---

## 11. 续：refs_pass2 与冻结清单对齐（判据变更留痕 · 2026-09-10）

同轮新增 L3 仓库卫生门禁后，`bash scripts/check_all.sh` 首次稳定运行，把 CHK11 报出的
**5 条 "stale reference"** 暴露出来（此前一直被别的问题遮住）。逐条核查后判定为**工具 bug，
不是新缺陷**，处置与留痕如下。

### 11.1 根因：同一份规则存在两份真相

- `scripts/refs_pass2.py` 自带一份**硬编码**的 `EXCLUDE_FILES` 表，用于跳过被扫描文件；
- 权威冻结清单是 `scripts/naming_frozen.txt`（`check_naming.sh::is_frozen` 一直在读它）；
- 两者**已经漂移**：`docs/NAMING_CONVENTION.md` 与 `docs/NAMING_ALIASES.csv` 在冻结清单 §4
  「命名工具自指」里，却**不在** `EXCLUDE_FILES` 里。
- 后果：引用扫描器把**命名工具自身的映射表**当成了"陈旧引用"。

这四份文件**按构造必须包含旧名**，因为它们就是 old→new 映射的定义本身：

| 文件 | 为什么必须含旧名 |
|---|---|
| `scripts/naming_path_map.csv` | 迁移的 old→new 对照表本体 |
| `docs/NAMING_ALIASES.csv` | 别名表（legacy → canonical） |
| `docs/NAMING_CONVENTION.md` | §2 冲突全量清单，逐条列出旧形态 |
| `scripts/apply_naming_migration.py` | 迁移规则字面量 |

### 11.2 处置

`refs_pass2.iter_text_targets()` 改为读 **`scripts/naming_frozen.txt`**，与 `check_naming.sh`
同一份文件、同一语义（glob 匹配仓库相对路径 / last-match-wins / 前导 `!` 解冻）；
硬编码表删除，消除双份真相。

### 11.3 ★ 变更性质声明（不得略去）

**这不是新增豁免，而是让引用扫描器与既有规则一致。**

1. 上述 4 个文件**早在 v3 就已冻结**（`naming_frozen.txt` 第 47–57 行，类别「命名工具自指」）。
   `check_naming.sh` 对 CHK1–CHK10 一直在跳过它们 —— CHK11 的路径扫描是唯一的例外。
2. 被撤回的 5 条"FIX"全部指向这 4 份冻结文件；其"目标"文件
   （`configs/phase2_ours_0.5m.yaml`、`scripts/phase2_chain_after.sh`、
   `scripts/phase1b_night_master_chain.sh`、`scripts/phase5_queue_errorgeom.sh`、
   `scripts/phase4b_recover_step2_then_chain.sh`）**本身健康存在于磁盘**。
   它们不是"改名遗留的陈旧引用"，而是"映射表里本来就该写着旧名"。
3. 影响面：CHK11 FIXABLE **5 → 0**；CHK11w 历史引用 **22 → 19**。
4. 反向守卫：若将来真有"改名后仍能解析到真实文件的陈旧引用"，来源**不是**这 4 份文件时，
   CHK11 仍会 FAIL —— 判据的判别力没有被削弱，只是不再对映射表自指误报。

### 11.4 同轮新增的机器门禁（非命名层，但同一入口）

- `scripts/check_repo_hygiene.sh`：断言 A1a / A1b / A2 / A3 / A4
  （审计链表被跟踪 / 任一 phase 目录不得全部 csv 被忽略 / 不得引入缓存与 >1MB 文件 /
  HEAD 不得有未登记 >50MB blob / 不得有失效的字面路径规则）。
  其中 **A1b** 正是本轮 `!experiments/phase*/**/*.csv` 泛化的守卫。
- `scripts/check_all.sh`：命名 + 卫生双门禁的唯一入口。
- `scripts/hooks/pre-commit` + `scripts/install_hooks.sh`：把门禁挂进提交。
- `--selftest` 会**植入反例并要求门禁 FAIL**（不能失败的门禁不是门禁）。
