# COS-87259 JWST 研究项目 —— 文件索引与任务导航

> 本文件是工作目录 `D:\Fudan_University\Research\JWST` 的总索引，用于把**阶段性任务 → 工作代码 → 代码结果**串成可追溯的链路，供后续 AI agent 与研究者快速定位。
>
> **阅读顺序建议**：先读 §1 任务主线（含每阶段的代码与结果对照表），再按 §2 数据流链路图定位具体文件，最后看 §7 已知问题清单避免踩坑。
>
> 索引生成日期：2026-08-17（同日更新：拆分出 `GalfitS\GalfitS_INDEX.md`，主索引大幅精简 GalfitS 部分）。路径均为相对本目录。后续工作目录更新时，同步修改应更新的内容并补充更新日期。

---

## 0. 项目概况

| 项               | 内容                                                                                                                                        |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| 目标源             | COS-87259，z ≈ 6.85 的极亮、尘埃遮蔽、射电噪 AGN（RA 09:58:58.3, Dec +01:39:20.2），宿主为巨质量星系（M_* ~ 10^10 M_sun，M_BH ~ 10^9 M_sun）                         |
| 科学目标            | 刻画宿主星系、尘埃辐射与周围环境；理解早期超大质量黑洞与尘埃如何形成增长、AGN 与宿主如何相互作用                                                                                        |
| 观测数据            | JWST/NIRSpec IFU（PRISM + G395H/F290LP）、JWST/MIRI（9 宽带 F560W–F2550W）、JWST/NIRCam（F115W/F200W/F410M），均已 pipeline 减约（Proposal 6576，另参考 4877） |
| 目标结构            | 5 个 clump：**Center、West、North、South、Southeast**（另有第 6 个"North of Southeast"候选亮斑）                                                          |
| 当前科学图景（2026-08） | **双 AGN 并合系统**；GalfitS 全局最优模型为 Sérsic host + AGN@C1&C2（C1 处AGN点源较弱）；West 有宽 Hα（FWHM ~3600 km/s）                                           |
| 环境              | Python 3.11 虚拟环境 `.venv\`；GalfitS 需 CUDA 12.4 + JAX                                                                                       |

**术语约定**：clump 名称 Center/West/North/South/Southeast 在 GalfitS 语境下对应 C1=Center、C2=West。`s3d` = NIRSpec IFU 3D 立方体。

---

## 1. 阶段性任务主线（任务 → 代码 → 结果）

### Stage 1 —— G395H 光谱提取 + MIRI 九波段测光（2026-03，已完成）

**任务定义**：`MissionLine\First stage mission.md`（docx 版同目录）。两个要求：①从 G395H 提取 5 个 clump 的 1D 光谱；②提取各 clump 的 MIRI 9 波段光度。

| 子任务 | 工作代码 | 输入 | 结果/产出 | 备注 |
|---|---|---|---|---|
| 背景扣除、clump 识别、PRISM 椭圆孔径提取（全流程原始版） | `COS87259.ipynb`（导出版 `COS87259.py`，背景扣除段已注释） | `NIRSpec_IFU\COS-87259_NOBG_{prism-clear,g395h-f290lp}_s3d.fits` | `NIRSpec_IFU\COS-87259_BGSUB_{prism-clear,g395h-f290lp}_s3d.fits`；clump 椭圆孔径参数（后续被硬编码复用） | 方法说明见 `Journal\Code_Explanation_and_FITS_Guide.md` |
| G395H 五 clump 1D 光谱提取（椭圆/Kron 孔径） | `extract_spectra.ipynb` | BGSUB g395h s3d | `G395H_analysis\{Clump}_G395H_spectrum.txt` ×5（3 列：Wavelength_um, Flux, Err）+ `G395H_analysis\all_clumps_G395H.png`、`photo_G395H.png` | 原输出目录 `extracted_spectra\` 已不存在，产物归档于 `G395H_analysis\` |
| 双窗口多高斯谱线拟合（Hβ+[OIII] 与 Hα+[NII]；窄+宽双组分；[OIII] 1:2.98、[NII] 1:2.95 固定比；速度空间 FWHM 绑定；z ∈ [6.80, 6.90]） | `fit_spectra.py` | `G395H_analysis\{Clump}_G395H_spectrum.txt` | **`fitted_lines\`**：`{Clump}_{Hb_OIII,Ha_NII}_fit.png` ×10、`hboiii_fit_results.csv`、`ha_fit_results.csv` | 拟合方案设计文档：`Journal\Implementation_plan\implementation_plan-fit.md`；结果解读：`Journal\Workthough\walkthrough-fit.md`。⚠️ CSV 流量为 ×1e19 缩放单位，非物理值 |
| 快速单高斯测线（早期探索） | `quantify_lines.py` | 同上 5 个 txt | `G395H_analysis\extract_lines.txt`（UTF-16，控制台输出转存） | — |
| G395H 汇总图 | `plot_spectra.py` | 同上 5 个 txt | `G395H_analysis\all_clumps_G395H.png` | — |
| MIRI 9 波段测光（IFU WCS → MIRI 像素坐标，缩放椭圆孔径） | `extract_miri.ipynb`（末 cell = `plot_miri_seds.py`） | `MIRI\cos87259_F*_60mas.fits.gz` + PRISM s3d（仅取 WCS） | **`MIRI_analysis\miri_photometry.csv`**（5 clump × 9 波段，列：Clump ID, Band, Wavelength_um, Flux_uJy, Flux_Err_uJy）+ `MIRI_analysis\miri_seds.png` | 原输出目录 `extracted_photometry\` 现为空。⚠️ Center 与 West 流量完全相同（未真正分离）；South/Southeast 部分波段负流量 |

**相关质检/探索**：`Interactive_2D_Spectra.ipynb`（逐波长 2D 切片交互查看，留截图 3 张于 `MIRI_analysis\`）；`Journal\Journal.md` 2026.3.10–4.28 记录本阶段问答与结论（宽线仅 Hα 无 Hβ → 类 Seyfert 1.9 尘埃消光解释；West [OIII]/Hβ 下限等）。

### Stage 2 —— 新孔径范式试验：等值线法与分水岭法（2026-04，已完成）

**任务定义**：`MissionLine\Second stage mission.md`。椭圆孔径包含大量非 clump 像素，要求用①通量等值线孔径（老师方案）、②自拟新范式（实现为分水岭分割）并行试验，提取 1D 光谱并用 `fit_spectra.py` 同款方法拟合，全部结果存入 `NewApertureTrials\`。

| 子任务 | 工作代码 | 输入 | 结果/产出 | 备注 |
|---|---|---|---|---|
| 生成等值线 notebook（峰值 80%/50%/20% 等值线掩膜 + `scipy.ndimage.label` 防 clump 合并 + 3σ 底限） | `make_contour_ipynb.py`（生成器）→ `NewApertureTrials\Contour_Aperture.ipynb` | BGSUB g395h s3d | `NewApertureTrials\Contour_Output\{Clump}_Lvl{0.2,0.5,0.8}_spectrum.txt`（实际 12 个，**Southeast 缺失**：掩膜为空被跳过） | ⚠️ 未批量拟合，仅 West_Lvl0.5 一个示例拟合（内嵌图不落盘） |
| 生成分水岭 notebook（`skimage.segmentation.watershed`，5 个 marker） | `make_watershed_ipynb.py`（生成器）→ `NewApertureTrials\Watershed_Aperture.ipynb` | 同上 | `NewApertureTrials\Watershed_Output\{Clump}_spectrum.txt` ×4（**Southeast 缺失**） | ⚠️ Center/West 共用峰值，实际用手工矩形切分（y=27, x=29）补救，非纯算法分界 |
| 分水岭光谱拟合（`fit_spectra.py` 变体，只改路径与 clump 列表） | `NewApertureTrials\fit_spectra.py` | `Watershed_Output\*.txt` | `NewApertureTrials\fitted_lines_watershed\{Clump}_{Ha_NII,Hb_OIII}_fit.png` ×8 + `ha_fit_results.csv`、`hboiii_fit_results.csv`（4 clump） | ⚠️ 多个宽成分 FWHM 钉在拟合边界（23550 km/s 等），解读需注意 |
| 总结文档 | `Journal\Workthough\walkthrough-aperture.md`（另拷贝于 `NewApertureTrials\`）、`NewApertureTrials\WatershedAlgo.md` | — | — | walkthrough 声称两个方案均"自动拟合"，与磁盘实际不符（contour 无拟合产物） |

**结论记录**（`Journal\Journal.md` 4.25）：分水岭法所得谱线流量比椭圆法小约 3–5 倍；Center 的 Hα 峰值在两种方法下与 West 的大小关系相反。此后主流程仍沿用椭圆孔径（`fitted_lines\`）。

### Stage 3 —— MIRI 九波段图像分解（2026-07-27 下达，进行中，与 GalfitS 阶段合流）

**任务定义**：`MissionLine\Third stage mission.md`。以 NIRCam F200W 或 F410M 为 baseline，提取五个 clump 的 MIRI 图像并测每波段流量；F560W 预计可分出 C（Center）与 W（West），更长波段预计只剩 W（单一 PSF 轮廓）；其余 clump 给 3–5σ 上限；暂时不做每 clump 的 galaxy+AGN 成分分离。参考 arXiv:2602.12325（= `Papers\RuancunLi_2602.12325v2.pdf`）。

**实际响应落在 `GalfitS\` 工作区**：clump 流量与 5σ 上限的直接产出在 `GalfitS\MIRI_NIRCam_result\MIRI_clump_departure\`（`MIRI_clump_departure.md`、`clump_fluxes_*.csv`、`clump_seds*.png`、`clump_upper_limits.csv`）。各拟合实验与产物的完整索引见 **`GalfitS\GalfitS_INDEX.md`**。

### Stage 4（并行/后续）—— GalfitS 图像分解与 SED 拟合（2026-06 ~ 08，阶段性完成）

**任务来源**：`GalfitS\Journal\prompts.md`（含 7.18 联合拟合要求）与 `GalfitS\Journal\Journal.md`。用 GalfitS（RuancunLi/GalfitS 克隆 + libprofit，JAX）对 MIRI 9 波段 + NIRCam 3 波段做形态-SED 联合拟合，确定各成分性质。

> **GalfitS 工作区的完整索引见 `GalfitS\GalfitS_INDEX.md`**（工具链、预处理链、14 个拟合实验目录与 χ²/BIC 排名、TrialLog、分量分离产出、已知坑）。此处只留骨架：
>
> - 数据：`GalfitS\{MIRI,NIRCam}\` → `preprocess_miri_v3.py` / `preprocess_nircam.py` → `MIRI_cutout_v3\`、`NIRCam_cutout\`（v1 弃用、v2 仅 6 波段）
> - 拟合：`codes_for_analysis\generate_configs.py` + `batch_*.sh` 与手写 .lyric → `GalfitS\MIRI_NIRCam_result\{14 个方案目录}\`；权威运行记录 `MIRI_NIRCam_result\TRIAL_LOG.md`（2026-08-17 更新，含 8-17 固定中心重跑与目录改名说明）
> - clump 测光总结：`MIRI_NIRCam_result\MIRI_clump_departure\`
> - 历史试错：`MIRI_result\`（v1）、`MIRI_result_v2\`（MIRI-only 阶段）；工具链试验区：`GalfitS_test\`
>
> 拟合质量排名（noSED 全波段 BIC）以 TRIAL_LOG.md 为准；对应物理解释**"双 AGN 星系并合"猜想**（见 §6）。

---

## 2. 数据流链路图（快速导航）

```
链路 A（G395H 光谱主线）
  NIRSpec_IFU\NOBG s3d
    → COS87259.ipynb（背景扣除+clump识别）→ NIRSpec_IFU\BGSUB s3d
    → extract_spectra.ipynb（椭圆孔径提取）→ G395H_analysis\{Clump}_G395H_spectrum.txt
        ├→ fit_spectra.py → fitted_lines\（10 png + 2 csv，主拟合结果）
        ├→ plot_spectra.py → G395H_analysis\all_clumps_G395H.png
        └→ quantify_lines.py → G395H_analysis\extract_lines.txt

链路 B（第 6 clump "North of Southeast"）
  find_extra_spot.py → temp_spots.json（x=18.75, y=24.92）
    → extract_north_southeast.py → G395H_analysis\NorthTo_Southeast\{spectrum.txt, plot.png}

链路 C（MIRI 测光）
  MIRI\*.fits.gz（9 波段）+ IFU WCS
    → extract_miri.ipynb → MIRI_analysis\miri_photometry.csv
    → plot_miri_seds.py（= notebook 末 cell）→ MIRI_analysis\miri_seds.png
  （原 extracted_photometry\ 已清空）

链路 D（新孔径试验，对照实验，未进主流程）
  make_contour_ipynb.py → NewApertureTrials\Contour_Aperture.ipynb → Contour_Output\*.txt（12）
  make_watershed_ipynb.py → NewApertureTrials\Watershed_Aperture.ipynb → Watershed_Output\*.txt（4）
    → NewApertureTrials\fit_spectra.py → fitted_lines_watershed\（8 png + 2 csv）

链路 E（GalfitS 图像分解与 SED 拟合，Stage 3/4 —— 详见 GalfitS\GalfitS_INDEX.md）
  原始数据 → preprocess_*.py → cutout（MIRI_cutout_v3 + NIRCam_cutout）
    → generate_configs.py + batch_*.sh（galfits optimizer 3000 步）→ MIRI_NIRCam_result\{方案}\（TRIAL_LOG.md 为权威运行记录）
    → separate_*.py → 分量流量 csv；MIRI_clump_departure\（clump 流量、SED 图、5σ 上限）
```

---

## 3. 目录速查表

| 目录 | 性质 | 关键内容 |
|---|---|---|
| `MissionLine\` | **任务定义** | First/Second/Third stage mission.md（+docx）、clumps.png、Endsley_2023-1b.png |
| `Journal\` | **工作日志与知识文档** | Journal.md（2026.3–8 逐条问答与结论）、Code_Explanation_and_FITS_Guide.md、Spectra_BasicKnowledge.md、Implementation_plan\implementation_plan-fit.md、Workthough\walkthrough-fit.md、walkthrough-aperture.md、MIRI_bands.png、MIRI_sensitivities.png |
| `NIRSpec_IFU\` | 原始数据 + 中间产物 | NOBG / BGSUB 的 prism-clear 与 g395h-f290lp s3d fits（共 5 个，约 354 MB；`._*` 为 macOS 垃圾文件） |
| `MIRI\` | 原始数据 | `cos87259_F{560W..2550W}_60mas.fits.gz` ×9（约 578 MB） |
| `NIRCam\` | 原始数据 | `{F115W,F200W,F410M}_cos87259_sci.fits`（约 1.36 GB） |
| `postage_stamp\` | 展示切片（无 WCS，不能做天文测量） | 12 个 2D 裁剪 fits（NIRCam 200×200、MIRI 100×100） |
| `G395H_analysis\` | Stage 1 光谱提取成果 | 5 clump 光谱 txt、all_clumps_G395H.png、photo_G395H.png、extract_lines.txt、`NorthTo_Southeast\` |
| `fitted_lines\` | Stage 1 主拟合结果 | 10 张拟合图 + 2 个 csv |
| `MIRI_analysis\` | Stage 1 MIRI 测光结果 | miri_photometry.csv、miri_seds.png、3 张 2D 切片截图 |
| `extracted_photometry\` | ⚠️ 空目录（历史输出目录） | 产物已移至 `MIRI_analysis\` |
| `NewApertureTrials\` | Stage 2 全部成果（自包含） | Contour_Aperture.ipynb、Watershed_Aperture.ipynb、fit_spectra.py、Contour_Output\、Watershed_Output\、fitted_lines_watershed\、walkthrough-aperture.md、WatershedAlgo.md |
| `GalfitS_test\` | GalfitS 早期试验区（2026-06 初） | 工具克隆 + quickstart 数据；ChangeLog/FixingLog.md（numpy 2.0 bug 修复记录）；无 COS-87259 数据 |
| `GalfitS\` | **GalfitS 正式工作区**（Stage 3/4） | 内部结构与拟合实验索引见 **`GalfitS\GalfitS_INDEX.md`**；关键入口：CLAUDE.md、Journal\Journal.md、MIRI_NIRCam_result\TRIAL_LOG.md |
| `Papers\` | 参考文献与提案 | 见 §5 |
| `.venv\`、`.vscode\` | 环境 | 勿动 |

## 4. 根目录脚本速查表

| 脚本 | 一句话作用 | 产物 |
|---|---|---|
| `COS87259.ipynb` / `.py` | PRISM 全流程：背景扣除 + clump 识别 + 椭圆孔径提取（参数被后续脚本硬编码复用） | BGSUB s3d fits |
| `extract_spectra.ipynb` | G395H 五 clump 椭圆孔径正式提取 | G395H_analysis\*.txt |
| `fit_spectra.py` | 双窗口窄+宽双组分多高斯拟合（lmfit 风格约束） | fitted_lines\ |
| `quantify_lines.py` | 快速单高斯测线 | extract_lines.txt |
| `plot_spectra.py` | 五 clump 光谱汇总图 | all_clumps_G395H.png |
| `Interactive_2D_Spectra.ipynb` | 逐波长 2D 切片交互查看（质检） | 无文件（截图见 MIRI_analysis\） |
| `find_extra_spot.py` | SEP 找线窗内额外亮斑 | temp_spots.json |
| `extract_north_southeast.py` | 提取第 6 clump 光谱 | G395H_analysis\NorthTo_Southeast\ |
| `make_contour_ipynb.py` | 生成等值线孔径 notebook | NewApertureTrials\Contour_Aperture.ipynb |
| `make_watershed_ipynb.py` | 生成分水岭孔径 notebook | NewApertureTrials\Watershed_Aperture.ipynb |
| `extract_miri.ipynb` | MIRI 9 波段测光（WCS 坐标变换） | MIRI_analysis\miri_photometry.csv |
| `plot_miri_seds.py` | MIRI SED 图 | MIRI_analysis\miri_seds.png |
| `COS87259_JWST.tar.gz` | 原始下载数据包（1.2 GB） | — |
| `光谱拟合对比.pptx` | 拟合结果汇报幻灯片（2026-04-28） | — |

## 5. 文档与文献索引

**文献（`Papers\`）**
- `COS87259\Endsley_2022.pdf`（MNRAS 512, 4248）——目标源发现论文（射电噪 AGN 候选）
- `COS87259\Endsley_2023.pdf`（MNRAS 520, 4609）——ALMA 确认 z=6.853 超亮尘埃遮蔽射电噪 AGN
- `COS87259\Endsley_2021a/b.pdf`——高红移星族/再电离背景
- `COS87259\JWST Proposal 4877.pdf`、`JWST Proposal 6576.pdf`——本项目观测提案（6576 为 COS-87259 数据来源）
- `RuancunLi-2505.12867v2.pdf`——高红移类星体核区与宿主二分性（课题组）
- `RuancunLi_2602.12325v2.pdf`——z=7.2 红超爱丁顿类星体（Fujimoto 等合作；**第三阶段任务指定参考**）
- `Chen_2025_ApJL_989_L12.pdf`、`Chen_2025_ApJ_983_60.pdf`——Little Red Dots 宿主星系研究（课题组）
- `s41586-026-10579-4.pdf`（Nature 2026）——LRD 黑洞质量直接测量基准
- `reading_notes.pdf`——个人阅读笔记

**方法/流程文档（`Journal\`、`GalfitS\Journal\`）**：见 §3 对应行；GalfitS 内部文档（`Journal\prompts.md`、各结果目录 `TRIAL_LOG.md`、`MIRI_clump_departure\MIRI_clump_departure.md` 等）完整清单见 `GalfitS\GalfitS_INDEX.md` §8。

---

## 6. 关键科学结论备忘（截至 2026-08）

- **West = 宽线 AGN**：G395H 拟合 Hα 宽成分 FWHM ≈ 3580–3650 km/s（窄线 ≈ 400 km/s），宽成分约占 Hα 总流量 40%；Hβ 宽线缺失（尘埃消光，类 Seyfert 1.9）；West 各线宽均大于 Center，[OIII]/Hβ 只能给出下限。
- **GalfitS 拟合的模型解释（仅为猜想）**：拟合效果最好的模型为自由中心 Sérsic host + 自由中心 AGN@C1（Center）@C2（West），且 C1（Center）处 AGN 点源可能可以看做被 Sérsic 吸收——即 Center 更像恒星/host 主导、West 为尘埃遮蔽 AGN；C1 偏蓝、C2 偏红且主导 F2100W–F2550W；North/South/Southeast 仅得 5σ 上限。而同时，双固定中心 Sérsic+双固定中心AGN 的拟合被看做是最 general 的模型，代表对该源物理模型的猜想——**双 AGN 星系并合系统**。（各拟合的 χ²/BIC 与细节见 `GalfitS\GalfitS_INDEX.md`）
- 红移测量在 6.82–6.86 间略有分歧（不同 clump/谱线拟合所得），尚未统一钉死。

## 7. 已知问题与坑（后续 agent 必读）

1. **目录迁移**：`extracted_spectra\`、`extracted_photometry\` 已不存在/清空，产物分别在 `G395H_analysis\`、`MIRI_analysis\`；旧代码（如 `plot_miri_seds.py`）仍写旧路径，直接运行会失败或产出到旧目录。
2. **MIRI 测光未分离 C/W**：`miri_photometry.csv` 中 Center 与 West 各行完全相同；South/Southeast 有负流量且无 3–5σ 上限列（第三阶段要求的上限未在该表中体现，GalfitS 路径的 `clump_upper_limits.csv` 已补）。
3. **单位陷阱**：`fitted_lines\*.csv` 与 `fitted_lines_watershed\*.csv` 的 Flux 列为拟合时 ×1e19 的缩放单位；`postage_stamp\` 无 WCS。
4. **NewApertureTrials 缺口**：Southeast 在 contour 与 watershed 两方案下均无输出；contour 方案无批量拟合结果（walkthrough 文档表述与实际不符）；watershed 的 Center/West 边界为手工矩形切分；部分 FWHM 钉在拟合边界值（23550 / 1884 km/s）不可直接引用。
5. **GalfitS 环境与数据**：需 Python 3.11 + CUDA 12.4；`GS_DATA_PATH` 指向 `GalfitS\data`；工具源码已有本地补丁（numpy 2.0 bug、三个 MIRI filter 转换因子），升级需保留；其余 GalfitS 内部注意事项（v1 cutout 弃用、SED 模式劣于 noSED、2026-08-17 目录改名、文档与排名不一致等）见 `GalfitS\GalfitS_INDEX.md` §7。
6. **原始数据残留**：`NIRSpec_IFU\`、`MIRI\`、`NIRCam\` 中的 `._*` 文件为 macOS AppleDouble 垃圾，可忽略。

## 8. 未完成/待办方向

- **Stage 3 收尾**：以 F200W/F410M 为 baseline 的 MIRI 分解已由 GalfitS 路径大体完成（`MIRI_clump_departure\`），可核对是否满足任务验收标准（每 clump 各波段流量或上限、与 arXiv:2602.12325 Figure 4 的对照）。
- **双 AGN 科学故事**（Journal 8.2，Mengtao 提出）：对 Center/West 做 AGN + host SED 模型拟合确定物理参数（如黑洞质量、宿主星族）；处理每 clump 的 SED。
- **遗留技术问题**：统一红移（钉死 Hα 红移）；West Hα 宽线的严格统计认证（Journal 4.28 提出的 1000 条随机光谱模拟法）；`miri_photometry.csv` 的 C/W 分离与负流量上限重做。
