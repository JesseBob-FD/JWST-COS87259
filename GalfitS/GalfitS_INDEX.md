# GalfitS 工作区内部索引（COS-87259 MIRI+NIRCam 图像分解与 SED 拟合）

> 本文件是 `GalfitS\` 文件夹的内部索引，索引各拟合实验文件夹、拟合实验记录（TrialLog）、自研脚本与产出，供后续 agent 在 GalfitS 工作区内工作时快速定位。
> 项目总索引见 `..\INDEX.md`（Stage 3/4 概述）。路径均为相对 `GalfitS\`。
>
> **阅读顺序建议**：先读 §2 目录总览与 §3 命名约定（2026-08-17 改名），再按 §4 实验目录表定位具体拟合，最后看 §7 已知坑。
>
> 更新日期：2026-08-17（据 2026-08-17 更新的 `MIRI_NIRCam_result\TRIAL_LOG.md` 重写）。

---

## 1. 工作区概览

| 项      | 内容                                                                                                                                                                                                                                                                   |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 目的     | 回应用户任务（`Journal\prompts.md` 与主索引 Stage 3/4）：用 GalfitS 对 COS-87259 的 MIRI 9 波段（F560W–F2550W）+ NIRCam 3 波段（F115W/F200W/F410M）做图像分解与 SED 拟合，测量各 clump 流量与上限                                                                                                             |
| 对象     | COS-87259，z=6.83，中心 RA=149.74276239°、Dec=1.6555373°；C1 = Center = (+0.124, +0.139)″、C2 = West = (−0.116, +0.169)″（F410M 定位）                                                                                                                                          |
| 工具链    | `GalfitS\GalfitS\`（github.com/RuancunLi/GalfitS 克隆；GALFIT 的 SED 联合拟合版，JAX 自动微分；`Ia15=0` 纯测光模式每波段独立 logNorm，`Ia15=1` SED 模式用 BC03/CLOUDY/DL2007 模板）+ `GalfitS\libprofit\`（C++ 剖面库）+ `GalfitS\astroskills\`（Claude Code 技能）+ `GalfitS\tutorial\`——**均为第三方源码，勿当自研脚本修改** |
| 本地工具补丁 | `GalfitS\GalfitS\src\galfits\galfitS.py` 修复 numpy 2.0 `newbyteorder()` 兼容 bug；`gsutils.py` 手工补充 F560W/F1130W/F2550W 三个 MIRI filter 及转换因子。升级工具时需保留                                                                                                                    |
| 环境     | Python 3.11（严格）+ CUDA 12.4 + JAX；`GS_DATA_PATH` 指向 `GalfitS\data`（SED 模板 .npz 在 `data\templates\`）。使用说明见 `CLAUDE.md`                                                                                                                                                 |
| 科学解释口径 | 本索引只记录拟合实验与拟合质量（χ²/BIC），**不做强物理解释**。                                                                                                                                                                                                                                 |

## 2. 目录总览

| 路径 | 性质 | 内容 |
|---|---|---|
| `GalfitS\`、`libprofit\`、`astroskills\`、`tutorial\` | 第三方工具（克隆） | GalfitS 源码、C++ 剖面库、技能定义、教程 |
| `MIRI\`、`NIRCam\` | 原始数据（拷贝） | 9 波段 `cos87259_F*_60mas.fits.gz`（0.06″/px）；3 波段 `*_cos87259_sci.fits`（0.03″/px） |
| `MIRI_cutout\`（v1） | ⚠️ 弃用 | 坐标错误（不含源），仅保留 provenance |
| `MIRI_cutout_v2\` | 中间产物 | 修正坐标后 10″ cutout，仅 6 个有 filter 的波段；2 阶多项式天光减除；51×51 高斯 PSF |
| `MIRI_cutout_v3\` | **正式输入** | 2″ cutout（66×66 px）、全部 9 波段、σ=3 剪除 2 阶天光；`images\`（0=减天后、1=MASK、2=SIGMA）+ `psf\` |
| `NIRCam_cutout\` | **正式输入** | 3 波段 2″ cutout、σ-clipped 中位天光、MJy/sr、高斯 PSF |
| `data\` | 工具数据 | quickstart 示例（`result\` 为示例 J0056-0021 拟合产物）+ SED 模板（`templates\`） |
| `MIRI_result\` | 历史试错（v1） | run001–003，`TRIAL_LOG.md` 记录教训：**BC03 模板在 z≈6.8 的 MIRI 波段流量≈0 → SED 模式失效；坐标必须来自 IFU WCS** |
| `MIRI_result_v2\` | 历史试错（v2） | run001–004（各含 `result\` 与 `result_SED\`），6 波段；`TRIAL_LOG.md` |
| `MIRI_NIRCam_result\` | **正式结果（核心）** | 14 个拟合实验目录 + 1 个总结目录，见 §4；权威运行记录 `TRIAL_LOG.md` |
| `codes_for_analysis\` | 自研脚本 | 配置生成、批量运行、定位、排名等，见 §6 |
| `Journal\` | 工作记录 | `Journal.md`（6.19–7.31 问答日志）、`prompts.md`（用户任务指令原文）、`check_result_files.py`、`debug_agn.py` |
| `.file\` | 空占位 | 无内容 |
| 根下文档 | 说明 | `CLAUDE.md`（工具使用）、`GalfitS_SED_analysis.md`（源码分析）、`Knowledge.md`（χ²/BIC 模型选择笔记）、本索引 |

## 3. 拟合实验命名约定（重要，2026-08-17 改名）

- **`*_C1C2` 后缀 = 相关点源成分的中心固定在 C1/C2**（`vary=0`）；**无后缀 = 中心自由**（`vary=1`）。
- 2026-08-17 把 7-31/8-02 的自由中心结果目录改名（去后缀），并在原 `*_C1C2` 名下重跑"仅固定 AGN 中心"的新拟合：
  - `host_dualAGN_C1C2` → **`host_dualAGN`**（自由中心，R012）；新 `host_dualAGN_C1C2`（AGN 中心固定）
  - `dual_AGN_C1C2` → **`dual_AGN`**（自由中心，R011）；新 `dual_AGN_C1C2`（中心固定）
  - `two_sersic_two_AGN_C1C2` → **`two_sersic_two_AGN`**（Sérsic 固定 + AGN 自由，R013）；新 `two_sersic_two_AGN_C1C2`（全部固定）
  - `two_sersic_C1C2` 未改名（无 AGN 成分，Sérsic 中心本就固定）
- ⚠️ 改名目录内 `.lyric` 配置文件名仍保留旧 `_C1C2` 名（如 `host_dualAGN\noSED\host_dualAGN_C1C2_noSED.lyric`），与新固定中心目录内的同名文件区分要靠**目录**，不要只看文件名。
- 早期 run（R001–R008）已从 `run001\`–`run010\` 迁移到描述性目录；其 `.gssummary` 头部仍记录旧 `runXXX\` 路径。

## 4. 拟合实验索引（`MIRI_NIRCam_result\`）

**统一拟合设置**：optimizer、3000 步、lr=0.001（CPU JAX）；12 波段 atlas（`jwst_miri` + `jwst_nircam`）；两种模式：noSED（Ia15=0）与 SED（Ia15=1）。**NP AGN**（Na18=4）：AGN 流量由 12 个逐波段自由 `logL` 决定，初值取 Elvis+94 射电宁静类星体模板插值；其余 AGN 参数（logM、logLedd、spin、Av、logL5100、torus、热尘埃）在拟合中不移动。幂律 AGN（Na18=0）以 logL5100 + 幂律指数描述。

### 4.1 实验目录总表（χ²/BIC 摘自 2026-08-17 版 TRIAL_LOG.md，noSED 为主）

| 目录 | 模型构成 | 配置（noSED） | χ² | BIC | 备注 |
|---|---|---|---|---|---|
| `single_sersic\` | 单 Sérsic host | `single_sersic\noSED\COS87259_R001_noSED.lyric` | 53,508.65 | 53,713.42 | R001；基线（另有 SED 版 R002，χ²=55,748） |
| `host_agn_powerlaw\` | 单 Sérsic + 幂律 AGN | `host_agn_powerlaw\noSED\COS87259_R002_noSED.lyric` | 50,862.51 | 51,067.28 | R003；SED 版 R004 χ²=51,909 |
| `bulge_disk\` | bulge + disk 双 Sérsic | `bulge_disk\noSED\COS87259_R003_noSED.lyric` | 53,508.66 | 53,918.20 | R005；双成分无改善；SED 版 R006 χ²=55,653 |
| `free_sky_single_sersic\` | 单 Sérsic + 自由天光 | `free_sky_single_sersic\noSED\COS87259_R004_noSED.lyric` | 50,611.37 | 50,952.65 | R007；SED 版 R008 χ²=53,996 |
| `host_freeAGN\` | 单 Sérsic + 自由 NP AGN（同中心） | `host_freeAGN\noSED\COS87259_R010_NPnoSED.lyric` | 53,442.40 | 53,783.68 | R010；SED 版 R009 χ²=99,008（唯一 red.χ²≈1 的 run）；`SED\` 内有 leftover 配置无输出 |
| `fixed_center_C1host_C2AGN\` | host@C1 + NP AGN@C2（固定） | `fixed_center_C1host_C2AGN\noSED\C1host_C2AGN_noSED.lyric` | 49,113.79 | 49,432.31 | SED 版 χ²=74,585；`noSED\` 内有 `fit_component_sed.py`（对分离流量做幂律 AGN + BC03 host SED 拟合 → `component_sed_fit.png`） |
| `fixed_center_C2host_C1AGN\` | host@C2 + NP AGN@C1（固定，另一种分配） | `fixed_center_C2host_C1AGN\noSED\C2host_C1AGN_noSED.lyric` | 52,951.86 | 53,270.38 | SED 版 χ²=57,639；明显差于 C1host_C2AGN（Δχ²≈3,838） |
| `two_sersic_C1C2\` | 双 Sérsic 固定 C1/C2（无 AGN） | `two_sersic_C1C2\noSED\C1C2_noSED.lyric` | 48,345.35 | 48,709.38 | 双 AGN run 之前的最佳；SED 版 χ²=56,785；`noSED\separate_components.py` v1 有 bug（见 §5） |
| `dual_AGN\`（R011） | 双 NP AGN，中心自由，无 host | `dual_AGN\noSED\dual_AGN_C1C2_noSED.lyric` | 57,982.44 | 58,460.23 | 无延展成分时拟合最差 |
| `dual_AGN_C1C2\` | 双 NP AGN，中心固定 | `dual_AGN_C1C2\noSED\dual_AGN_C1C2_noSED.lyric` | 62,550.41 | 62,982.70 | 2026-08-17 新跑（`run_fixed_center_fits.sh`）；分离：`separate_two_AGN.py` → `two_component_fluxes.csv` |
| `host_dualAGN\`（R012） | 自由 Sérsic host + 双 NP AGN，**全部中心自由** | `host_dualAGN\noSED\host_dualAGN_C1C2_noSED.lyric` | **47,341.38** | **48,023.91** | **全波段 noSED BIC 最优**（含 MIRI-only 排名）；分离：`separate_three_components.py` → `three_component_fluxes.csv`、`three_component_SED.png`、`{Host,AGN_C1,AGN_C2}_model_{band}.fits` |
| `host_dualAGN_C1C2\` | 同上，但 AGN 中心固定（host 仍自由） | `host_dualAGN_C1C2\noSED\host_dualAGN_C1C2_noSED.lyric` | 48,492.45 | 49,129.49 | 2026-08-17 新跑；比 R012 差 Δχ²=+1,151；分离产物同结构 |
| `two_sersic_two_AGN\`（R013） | 双 Sérsic 固定 C1/C2 + 双 NP AGN 中心自由 | `two_sersic_two_AGN\noSED\two_sersic_two_AGN_C1C2_noSED.lyric` | 47,567.80 | 48,409.59 | 全波段 BIC 第 2；分离：`separate_and_plot.py` → `four_component_fluxes.csv`、`two_group_fluxes.csv` |
| `two_sersic_two_AGN_C1C2\` | 同上，但 AGN 中心也固定 | `two_sersic_two_AGN_C1C2\noSED\two_sersic_two_AGN_C1C2_noSED.lyric` | 49,109.95 | 49,906.24 | 2026-08-17 新跑；比 R013 差 Δχ²=+1,542；分离产物同结构 |
| `MIRI_clump_departure\` | **clump 测光总结（非拟合目录）** | — | — | — | 见 §5 |

**各拟合目录内产物规律**：`{tag}.lyric`（配置）、`.params`（最佳参数）、`.constrain`、`.gssummary`（总 χ²/约化 χ²/BIC/逐波段 χ²/dof/自由参数表）、`{tag}image_fit.png`（12 波段 数据|模型|残差 三联图）、`{tag}SED_model.png`（仅 SED 模式）、12 个 `{tag}_{band}_result.fits`、`.galfitS.log`；分离产物见各目录备注。

### 4.2 排名摘要（拟合质量指标，摘自 TRIAL_LOG.md「Rankings」）

**全波段 noSED（按 BIC，前 6）**：R012 host+dualAGN（自由中心，48,024）＞ R013 2Sérsic+2AGN（AGN 自由，48,410）＞ two_sersic_C1C2（48,709）＞ host_dualAGN_C1C2（8-17 固定中心，49,129）＞ C1host_C2AGN（49,432）＞ two_sersic_two_AGN_C1C2（8-17 固定中心，49,906）。

**MIRI-only（按 BIC_MIRI，前 3）**：R012（19,715）＞ C1host_C2AGN（19,821）＞ host_dualAGN_C1C2（19,850）。（由 `MIRI_NIRCam_result\compute_miri_stats.py` 从 `.gssummary` 反解 k 重算。）

**SED 模式单独排序**（参数空间不同，不与 noSED 混排）：全部差于同模型 noSED；最优 R004 host+幂律 AGN（χ²=51,909），最差 R009 host+NP AGN（χ²=99,008）。

**TRIAL_LOG 中记录的拟合行为要点**（供后续 agent 参考，非科学结论）：
- 固定 AGN 中心相对自由中心：χ² 增加 +1,151（host+dualAGN）、+1,542（2Sérsic+2AGN）、+4,568（纯双 AGN）。
- 自由中心拟合中 AGN_C1 漂移 ≈0.045″、AGN_C2 ≈0.03″（相对 C1/C2）。
- NP AGN 只有 12 个逐波段 logL 参与优化，其余 AGN 参数停在初值。
- 多数 run 的约化 χ² ≈ 0.54–0.72（误差可能被高估）；F2550W 是 MIRI 中逐波段约化 χ² 最高的波段。

## 5. MIRI_clump_departure（Stage 3 的直接产出：clump 测光总结）

**核心文档**：`MIRI_clump_departure\MIRI_clump_departure.md`（方法论 + 结果 + 错误方法存档，2026-07-28）。基于 `two_sersic_C1C2/noSED` 分解，最终采用 `cal_model_image()` + logNorm 置零法（`plot_component_models.py` 最终版）做正确分量分离。

**关键文件**：
- 分量模型图：`component_model_images_{two_sersic,C1host_C2AGN,C2host_C1AGN}.png`
- 流量表：`clump_fluxes_{two_sersic,C1host_C2AGN,C2host_C1AGN}.csv`
- SED 图：`clump_seds*.png`（含 5 clump 总览与各方案版本）
- 5σ 上限（North/South/Southeast）：`clump_upper_limits.csv`（由 `upper_limits_clumps.py`，10×10 px 孔径 σ-clipped RMS；F560W 处 ~0.015–0.028 μJy，F2550W 处 ~0.6–0.9 μJy）
- 脚本：`extract_all_components.py`、`plot_component_models.py`、`plot_host_agn.py`、`recompute_fluxes.py`、`verify_separation.py`

**⚠️ 注意**：该 md（7.28）称 two_sersic 为 "best overall"，早于 8-17 版 TRIAL_LOG 的排名更新（现最优为 R012 host+dualAGN）——引用"最优模型"时以 **`MIRI_NIRCam_result\TRIAL_LOG.md`** 为准。另外 md 中的 `clump_fluxes_corrected.csv`/`clump_seds_corrected.png` 现已改名/拆分为 `clump_fluxes_two_sersic.csv`/`clump_seds_two_sersic.png`。前两版错误分离方法（漏 PSF 卷积、卷积顺序错误）及对应无效产物作为 provenance 存档于 md §3。

## 6. 自研脚本索引（`codes_for_analysis\` 与根下）

| 脚本 | 功能 | 对应产出 |
|---|---|---|
| `preprocess_miri.py` / `_v2.py` / `_v3.py`、`preprocess_nircam.py`（根下） | MIRI/NIRCam 预处理三代 | `MIRI_cutout\`（弃用）/`_v2\`/`_v3\`、`NIRCam_cutout\` |
| `generate_configs.py` | 模板化生成 12 波段 .lyric（4 setups × 2 modes） | 早期 run 配置 |
| `batch_all.sh` / `batch_noSED.sh` / `batch_agn.sh` | 批量跑 galfits | run001–008 等 |
| `gen_two_sersic.py` / `gen_fixed_center.py` | 生成双 Sérsic / 固定中心配置 | `two_sersic_C1C2\`、`fixed_center_*\` |
| `batch_two_sersic.sh` / `batch_fixed_center.sh` | 批量运行上述配置 | 同目录 `.gssummary` 等 |
| `run_fixed_center_fits.sh`（在 `MIRI_NIRCam_result\` 根下） | **2026-08-17 三个固定中心重跑**（顺序跑 host_dualAGN_C1C2、dual_AGN_C1C2、two_sersic_two_AGN_C1C2） | 三个新 `*_C1C2` 目录 |
| `compute_ia9_rcli.py` | 算 12 波段 Ia9 转换因子（RC-Li 法） | .lyric 中 Ia9 取值 |
| `compute_clump_sky.py`、`test_f410m_clumps.py` / `test_f410m_pixels.py` | F410M 像素团 → 天球坐标 → C1/C2 偏移 | 所有固定中心配置的坐标依据 |
| `compute_miri_stats.py`（在 `MIRI_NIRCam_result\` 根下） | 从 `.gssummary` 反解 k_free 重算 MIRI-only χ²/BIC | TRIAL_LOG「MIRI-Only Ranking」 |
| `collect_results.sh` / `collect_two_sersic.sh` / `collect_fixed.sh` | 汇总各 `.gssummary` 头部 | 终端排名表 |
| `upper_limits_clumps.py`、`plot_clump_seds.py`（在 `Journal\`） | 5σ 上限、clump SED 总览图 | `MIRI_clump_departure\` 对应 csv/png |
| `verify_conversion_factor.py` / `Code_ConversionFactor*.py` | Ia9 新旧方法对比验证 | 方法选定 |
| `check_nircam.py` / `check_wcs.py` / `check_result_files.py` / `debug_agn.py` / `debug_two_sersic.py` / `_temp_analyze_images.py` | 数据与拟合检查/调试（含临时工具） | — |
| `plot_resu.ipynb`、`zijian.ipynb`（根下） | 由工具示例（GNZ7Q）改编的模型/分量绘制 notebook；`cal_model_image()` 分离方法出处 | 方法论来源 |

## 7. 已知问题与坑（GalfitS 内部）

1. **目录改名**（2026-08-17）：自由中心结果在无后缀目录、固定中心重跑在 `*_C1C2` 目录；配置文件名两边相同，必须靠目录区分（见 §3）。
2. **MIRI_cutout（v1）坐标错误**，产物已弃用；引用 cutout 一律用 `MIRI_cutout_v3\` + `NIRCam_cutout\`。
3. **SED 模式（Ia15=1）整体劣于 noSED**：BC03 模板在 z≈6.8 的 MIRI 波段流量≈0，且误差校准问题；引用 SED 模式参数需谨慎（MIRI-only 阶段详见 `MIRI_result(_v2)\TRIAL_LOG.md`）。
4. **文档不一致**：`MIRI_clump_departure\MIRI_clump_departure.md`（7.28）的 "best overall" 表述早于最新 TRIAL_LOG；`.gssummary` 头部仍记录旧 runXXX 路径。
5. **NP AGN 参数**：除 12 个逐波段 logL 外，其余 AGN 物理参数停在初值，不能解读为物理参数测量。
6. **工具链**：Python 3.11 严格；filter/转换因子为手工补丁；`GalfitS_test\` 为早期试验区（numpy 2.0 bug 修复记录在 `ChangeLog.md`/`FixingLog.md`），无 COS-87259 数据。
7. **分量分离脚本历史**：`two_sersic_C1C2\noSED\separate_components.py` v1 与早期 `plot_component_models.py` 有 bug（漏/错 PSF 卷积），其产物无效；正确方法见 `MIRI_clump_departure.md` §3c。

## 8. 工作记录索引

| 记录 | 内容 |
|---|---|
| `MIRI_NIRCam_result\TRIAL_LOG.md` | **权威运行记录**（2026-08-17 更新）：全部 run 的模型/模式/χ²/BIC/配置路径、逐阶段细节、排名、改名说明 |
| `MIRI_result\TRIAL_LOG.md`、`MIRI_result_v2\TRIAL_LOG.md` | MIRI-only 阶段教训与 4+4 轮记录 |
| `Journal\Journal.md` | 6.19–7.31 工作日志（filter 缺失、Na27 bug、与作者 RC Li 的讨论、7.31 反思：free AGN 相对幂律无实质收益等） |
| `Journal\prompts.md` | 用户任务指令原文（含 7.18 联合拟合要求） |
| `CLAUDE.md` | 工具安装/使用/配置格式说明 |
| `GalfitS_SED_analysis.md`、`Knowledge.md` | 源码分析报告；χ²/BIC 模型选择笔记 |
