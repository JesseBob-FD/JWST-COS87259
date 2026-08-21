# GalfitS 多波段 SED 拟合：源代码分析报告

> 分析日期：2026-07-16  
> 源代码路径：`./GalfitS/src/galfits/`

---

## 目录

1. [核心问题：形态拟合与 SED 拟合的关系](#1-核心问题形态拟合与-sed-拟合的关系)
2. [整体架构概览](#2-整体架构概览)
3. [数据流全景](#3-数据流全景)
4. [数据容器层次（images.py）](#4-数据容器层次imagespy)
5. [SED 模板与物理模型（sed_interp.py）](#5-sed-模板与物理模型sed_interppy)
6. [星系模型：空间 + SED 的结合（galaxy.py）](#6-星系模型空间--sed-的结合galaxypy)
7. [拟合引擎（gsfit.py）](#7-拟合引擎gsfitpy)
8. [二维面亮度轮廓（profiles.py）](#8-二维面亮度轮廓profilespy)
9. [配置系统（gsutils.py）](#9-配置系统gsutilspy)
10. [关键实现文件速查表](#10-关键实现文件速查表)

---

## 1. 核心问题：形态拟合与 SED 拟合的关系

### 1.1 简短回答

**形态拟合与 SED 拟合是同时进行的（联合拟合），而非先后串行。** GalfitS 将形态参数和 SED 物理参数放在同一个参数向量中，通过统一的 χ² 损失函数同时优化。

### 1.2 传统两步法 vs GalfitS 联合法

| 方面 | 传统（GALFIT + CIGALE/MAGPHYS） | GalfitS（`images - SED` 模式） |
|---|---|---|
| 步骤 | ① 每波段独立拟合形态 → 得到流量表 ② 用流量表拟合 SED | 所有波段像素级数据 + SED 模板同时拟合 |
| 形态参数 | 每波段独立，不同波段可能得到不同的 Re/n/PA | 所有波段共享同一套形态参数 |
| SED 参数 | 仅从积分流量约束（信息量有限） | 从每个像素的 SED 约束，可利用空间颜色梯度 |
| 误差传播 | 两步传播，测光误差再叠加 SED 拟合误差 | 像素误差直接约束物理参数，传播更准确 |
| 空间 SED 梯度 | 无法处理（每个波段只有总流量一个数） | 天然支持（age/Z/Av 的径向梯度） |
| 最大优势 | 快速、成熟 | 信息利用充分、自洽性强 |

### 1.3 代码证据

**证据一：所有参数在同一个 θ 向量中**

`gsfit.py:347` — `gsfitter.cal_residual(theta)` 方法中，θ 同时包含：
- 形态参数：x, y, Re, n, axrat, PA...
- SED 参数：age, Z, Av, logM, sSFR, logU...

优化器（梯度下降/CMA-ES/嵌套采样）同时更新所有参数。

**证据二：模型图像由 SED × 形态联合计算**

`galaxy.py:1667` — `Galaxy.generate_image(band, ...)`:
```python
# 伪代码表示
for each subcomponent:
    # SED 部分：计算该波段的恒星族光度
    rest_sed = sed_interp.get_host_SED(logM, f_cont, age, Z, Av)
    obs_sed = sed_to_obse(rest_sed, redshift, Av_gal)
    flux_per_band = integrate(obs_sed * filter_response)  # 物理流量

    # 形态部分：将流量空间分布到像素
    model_image += mass_map(Re, n, axrat, PA, x, y) * flux_per_band

# 最终与 PSF 卷积
model_image = convolve(model_image, PSF)
```

SED 决定"每个波段的总亮度"，形态决定"这些亮度在空间上怎么分布"，两者在一次 `generate_image` 调用中同时计算。

**证据三：统一 χ² 损失函数**

`gsfit.py:347-399`:
```
总 χ² = Σ_图集 Σ_波段 Σ_像素 [(data_band - model_band(θ_all)) / σ]²
       + w_spec × Σ_光谱 Σ_λ [(data_spec - model_spec(θ_all)) / σ]²
```

优化器看到的是一个统一的损失函数。要降低 χ²，形态参数和 SED 参数必须联合调整——单独调整任何一组都无法使整体最优。

**证据四：纯测光模式是独立的例外**

GalfitS 也提供了传统的两步法路径——`images - photometry` 模式（`imagefitter_phot` 类，`gsfit.py:3512`），其中每个波段有独立的归一化参数 `logNorm_{comp}_{band}`，不使用 SED。这相当于传统 GALFIT。但默认推荐模式是 `images - SED`。

### 1.4 关键结论

> GalfitS 的创新之处恰恰在于**不做两步分离**，而是将 SED 物理建模直接嵌入像素级图像拟合中。这样做的代价是参数空间更大（每个子成分 ~15-20 个参数）、计算更昂贵，但好处是信息利用更充分、结果更自洽、能自然地处理空间 SED 梯度。

---

## 2. 整体架构概览

源代码位于 `GalfitS/src/galfits/`，核心模块如下：

```
src/galfits/
├── galfitS.py          # CLI 入口，参数解析，拟合分发
├── gsfit.py            # 核心拟合引擎（~4700 行），含 11 个 fitter 类
├── gsutils.py          # 配置解析、参数初始化、结果可视化（~2600 行）
├── galaxy.py           # 星系模型：空间结构 + SED 的结合（~2100 行）
├── sed_interp.py       # SED 模板插值引擎（~1835 行）
├── images.py           # 图像/数据容器（~3995 行）
├── profiles.py         # 二维面亮度轮廓（~935 行）
├── disperser.py        # 无缝光谱色散建模（~1198 行）
├── mathfunc.py         # 数学工具（~700 行）
├── emission_lines.py   # 发射线模型
├── external.py         # 外部接口（SExtractor 配置生成）
├── Constant.py         # 物理常数（10 行）
└── version.py          # 版本号
```

**核心设计理念**：将 SED 物理建模（恒星族 + 尘埃 + 星云）与二维空间轮廓建模统一在同一个前向模型和 χ² 框架中。对每个波段计算完整模型图像，将所有波段的像素级残差合并为一个向量，使用 JAX 加速优化。

---

## 3. 数据流全景

```
配置文件 (.lyric)
    │
    ▼
┌─ gsutils.read_config_file() ───────────────────────────────────┐
│  1. 解析图像配置（波段名、滤光片响应、PSF、单位转换）           │
│  2. 解析星系模型参数（Sérsic 结构 + SED 物理参数）              │
│  3. 构建 image → image_atlas → galfitS_data 数据层次            │
│  4. 自动检测 fitMode，选择合适的 fitter 子类                    │
└────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─ gsfit.gsfitter（或子类）.cal_residual(θ) ─────────────────────┐
│  对每个波段:                                                     │
│    1. Galaxy.generate_image(band) → 模型图像（计数率）           │
│       ├─ sed_interp 计算静止系 SED（恒星+星云+尘埃+AGN）        │
│       ├─ 红移 + 消光 + 滤光片积分 → 该波段的流量                 │
│       ├─ 乘以二维质量图（profiles） → 空间分布                   │
│       └─ 与 PSF 卷积 + 天光背景                                  │
│    2. (data - model) / sigma → 像素级残差                        │
│  对每条光谱:                                                     │
│    3. Galaxy.fiducial_sed() → 一维光谱模型                       │
│    4. (data - model) * w_spec / sigma → 光谱残差                 │
│  返回: 展平的总残差向量                                          │
└────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─ 优化/采样 ────────────────────────────────────────────────────┐
│  - optimizer():        Optax 梯度下降（Adam/AdamW/Lion/SGD）     │
│  - optimizer_multi():  多起点 + 多 GPU 并行                      │
│  - evolution_strategies(): CMA-ES 无梯度全局优化                 │
│  - nested_sampling():  Dynesty 嵌套采样（贝叶斯证据）            │
│  - jnesty_nested_sampling(): JAX 原生 GPU 嵌套采样               │
│  - mcmc():             NumPyro NUTS/HMC 采样                     │
│  - flowmc():           归一化流 + MCMC 混合                      │
└────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─ 输出 ─────────────────────────────────────────────────────────┐
│  .gsresu:     pickle 完整拟合结果                                │
│  .gssummary:  文本摘要（每波段 χ²、最佳参数、误差）             │
│  *image_fit.png: 每波段 数据|模型|残差 三图拼接                  │
│  *SED_model.png: 分解 SED 图（恒星/星云/尘埃/AGN 分量）         │
└────────────────────────────────────────────────────────────────┘
```

---

## 4. 数据容器层次（images.py）

多波段数据通过三层结构组织：

### 4.1 `image` 类（行 1123）

单个波段的图像容器。关键属性：

| 属性 | 含义 |
|---|---|
| `band` | 波段名字符串，如 `'F150W'`, `'g'`, `'H'` |
| `data / cut_image` | 图像数据（numpy 或 CCDData） |
| `cut_sigma_image` | 像素级误差图 |
| `cut_mask_image` | 像素掩模（坏像素/污染源） |
| `PSF` | 二维 PSF 图像 |
| `resp` | 滤光片透过率曲线 `[波长(Å), 透过率]` |
| `respnorm` | 滤光片归一化因子 `∫T(λ)dλ` |
| `phys_to_counts_rate` | 物理流量 (erg/s/cm²/Å) → 计数率转换因子 |
| `pixel_scales` | 像元比例尺 (arcsec/pixel) |
| `fitSED` | 布尔值，是否参与 SED 拟合 |
| `coordinates_transfer_para` | WCS 坐标变换参数字典 |
| `wcs_rotation` | 图像方位角 (rad) |
| `magzp` | 星等零点（默认 20.0） |

关键方法：
- `img_cut()` (行 1277)：在 RA/Dec 周围提取 cutout，计算 WCS 变换参数
- `img_cut_align()` (行 1535)：重采样到标准 North-Up 网格的 cutout

### 4.2 `image_atlas` 类（行 3128）

多个同视场波段图像的集合。关键属性：

| 属性 | 含义 |
|---|---|
| `image_list` | `image` 对象列表 |
| `band_list` | 对应波段名列表 |
| `spectra` | 可选光谱数据附件 |
| `same_grid` | 所有图像是否共享同一像素网格 |
| `common_catalog` | 多波段交叉匹配的源表 |
| `img_residual / img_dof` | 拟合后逐波段 χ² 残差统计 |
| `grism_image_list` | 无缝光谱数据（可选） |

关键方法：
- `init_sed()` (行 3474)：收集所有波段的滤光片响应函数和积分流量
- `source_detection()` (行 3489)：逐波段源检测
- `make_common_catalog()` (行 3504)：多波段源表交叉匹配
- `generate_PSFs()` (行 3586)：利用共有的恒星构建 ePSF
- `add_spectrum()` (行 3311)：添加光谱及孔径定义

### 4.3 `galfitS_data` 类（行 3677）

顶层容器，管理多仪器、多目标。关键方法：

- `calculate_tranformation()` (行 3904)：将图像按方位角分组（容差 3°），每组选最高分辨率图像作为参考，构建统一的滤光片波长网格
- `init_sed()` (行 3980)：汇集所有 atlas 的 SED 数据为平坦数组

---

## 5. SED 模板与物理模型（sed_interp.py）

### 5.1 模板库

模块在导入时从 `GS_DATA_PATH/templates/` 加载以下预计算模板网格：

| 模板文件 | 物理含义 | 插值维度 |
|---|---|---|
| `host_conti.npz` | 恒星连续谱（连续 SFH） | Z × age × wavelength |
| `host_inst.npz` | 星暴瞬时谱 | Z × age × wavelength |
| `host_conti_hires.npz` | 恒星连续谱（高分辨率） | Z × age × wavelength |
| `host_inst_hires.npz` | 星暴瞬时谱（高分辨率） | Z × age × wavelength |
| `nebular_DC.npz` | 星云连续谱（CLOUDY） | Z × logU × wavelength |
| `nebular_line.npz` | 星云发射线（CLOUDY） | Z × logU × wavelength |
| `dl2014Uconst.npz` | DL2014 尘埃（恒定 Umin） | qPAH × Umin × wavelength |
| `dl2014Upower.npz` | DL2014 尘埃（幂律 U） | qPAH × Umin × alpha × wavelength |
| `stellar.npz` | 恒星光谱库 | Teff × logg × logZ |
| `thin_disk_log.npz` | AGN 薄吸积盘 | spin × logMdot × logM × wavelength |
| `starburst_disk.npz` | AGN 星暴盘 | amax × logMdot × logM × wavelength |
| `BLRDC_ev.npz` | 宽线区自由-自由发射 | spin × logM × logMdot × wavelength |
| `eaZY.npz` | EAZY PCA 模板（18 分量） | template_index × wavelength |
| `cat3D.template` | CAT3D 环星周尘埃 torus | a × h × N0 × i |

所有插值器均为 JAX `RegularGridInterpolator`，支持 GPU 加速和自动微分。

### 5.2 关键 SED 生成函数

#### `get_host_SED(logM, f_cont, age, Z, Av)` (行 1203)

**恒星族 SED 的核心函数**。流程：

1. 根据 SFH 类型分支：
   - **`'conti'`**：连续恒星形成 + 瞬时星暴
     - 连续谱：`intp_host_cont(Z, age)` → 乘以 `f_cont`
     - 星暴谱：`intp_host_inst(Z, burst_age)` → 乘以 `(1 - f_cont)`
   - **`'burst'`**：纯星暴
   - **`'bins'`**：自定义分段 SFH，`sum(weight_i × intp_host_inst(Z, age_i))`
2. 合并连续谱和星暴谱，乘以 `10^logM / C_unit`
3. 应用尘埃消光：`mycigale(sed, Av, bump)` — Calzetti+2000 曲线 + 可选 2175Å 鼓包
4. 返回静止系 SED（单位：`10^38 erg/s/Å`）

#### `get_nebular(Lha, logU, Z, Av)` (行 1136)

星云连续谱 + 发射线。归一化到 Hα 光度（CLOUDY 模型拟合因子 1.87531）。

#### `get_cold_dust_sed(logMdust, qPAH, Umin, alpha, gamma)` (行 837)

Draine & Li (2007) 尘埃热辐射：
- 恒定辐射场分量：`Umin` 加热的尘埃
- 幂律辐射场分量：`alpha` 斜率的高能尾（星暴区域加热）
- `gamma` 控制两部分的比例

#### `get_AGN_SED(logM, logMdot, spin)` (行 1413)

薄吸积盘 SED。有效波长范围：125–24700 Å。

#### `sed_to_obse(sed, wave, z, Av_gal, Rv)` (行 1473)

将静止系 SED 转换到观测系：红移 → IGM 吸收（Madau 1995）→ 宇宙学亮度距离衰减 → 银河系前景消光（CCM89）。

### 5.3 辅助函数

| 函数 | 行号 | 功能 |
|---|---|---|
| `mycigale()` | 1769 | Calzetti+2000 消光 + 2175Å 鼓包 |
| `ccm89()` | 609 | Cardelli-Clayton-Mathis 银河系消光曲线 |
| `conv_spec()` | 1010 | 速度弥散卷积（log-λ 空间高斯核） |
| `conv_spec_lsf_R()` | 1089 | 波长相关 LSF 卷积 |
| `igm_transmission()` | 1587 | IGM 衰减（Madau 1995） |
| `luminosity_distance()` | 493 | 光度距离（ΛCDM 查询表） |
| `kpc_per_arcsec()` | 516 | 角尺度转换 |
| `cosmo_age()` | 539 | 宇宙年龄（红移处） |
| `FeII()` | 631 | 光学/UV FeII 发射模板 |
| `BaC()` | 701 | Balmer 连续谱（宽线区） |
| `get_eazy_sed()` | 390 | EAZY 18 分量线性组合 |
| `TableModel` 类 | 277 | 自定义表格 SED 模型加载器 |

---

## 6. 星系模型：空间 + SED 的结合（galaxy.py）

### 6.1 `Galaxy` 类（行 469）

核心设计：将星系分解为多个**子成分**（如核球 + 盘 + 棒 + AGN），每个子成分有独立的结构轮廓和 SED 参数。

关键属性：

| 属性 | 含义 |
|---|---|
| `subCs` | 每个子成分的结构参数（Re, n, axrat, PA...） |
| `ageparams / Zparams / Avparams / f_cont` | SED 参数及其**径向梯度**定义 |
| `logMlist` | 每个子成分的 log₁₀(恒星质量 / M☉) |
| `SFH_list` | 恒星形成历史类型：`'conti'` / `'burst'` / `'bins'` |
| `sedmodelist` | SED 模式标志：0=恒星+星云, 1=仅恒星, 2=仅星云, 3=仅尘埃, 4=全包 |
| `mass_map` | 二维质量分布图字典（由轮廓 + logM 决定） |
| `allconstant` | 布尔值：是否所有子成分的 SED 参数无径向梯度（决定优化路径） |
| `useEZ` | 是否使用 EAZY 模板模式 |
| `redshift / rnorm` | 红移 / 距离归一化因子 |
| `Nebular` | ISM 对象（星云发射建模） |
| `energy_balance` | 是否强制执行能量平衡（吸收 = 发射） |

### 6.2 关键方法

#### `add_subC()` (行 690)

添加一个子成分，参数包括：
- `Pro_names`：成分名（如 `'bulge'`, `'disk'`）
- `logM`：恒星质量
- `Sparams`：结构轮廓类型和参数
- `ageparam`, `Zparam`, `f_cont`, `Avparam`：SED 参数的径向梯度定义
- `SEDmode`：0-4 整数或 `'T'` (TableModel) / `'Z'` (EAZY)

#### `generate_mass_map()` (行 1047)

生成所有子成分的二维质量分布图。支持 15+ 种轮廓类型（Sérsic、Ferrer、边缘盘、Fourier 螺旋模式等）。返回 `mass_map[comp]` 字典。

#### `generate_image(band, impsf, resp, noSED, nebularpar)` (行 1667)

**多波段成像的最核心方法**。每个波段、每个图集调用一次：

**noSED 模式**（仅测光，传统 GALFIT 模式）：
```
model_image = Σ_comp (10^logNorm_{comp}_{band} × mass_map[comp])
```

**SED 模式**（联合物理建模）：
```
对每个子成分:
  1. 取 r=0 处的 SED 参数 (age₀, Z₀, Av₀, f_cont₀)
  2. sed_interp.get_host_SED(logM, f_cont, age, Z, Av)  → 恒星族静止系 SED
  3. + get_cold_dust_sed(...)                              → 尘埃热辐射
  4. + get_nebular(Lha, logU, Z, Av)                       → 星云连续谱+发射线
  5. sed_to_obse(sed, z, Av_gal)                           → 观测系 SED
  6. flux_per_band = ∫ obs_SED(λ) × filter_response(λ) dλ / ∫ filter dλ
  7. model_image += mass_map[comp] × flux_per_band × 距离归一化
与 PSF 卷积 → 最终模型图像
```

**关键效率假设**：SED 在 r=0 处计算，假设空间均匀。空间变化仅来自质量图。若需要径向 SED 梯度，需使用 `imagefitter_SED` 子类（构建 SED 数据立方体）。

#### `fiducial_sed(wavelength, apertures)` (行 1335)

生成集成一维 SED（非逐波段），用于光谱拟合和 SED 可视化。

#### `generate_group_image()` (行 1946)

当所有成分 `allconstant=True` 时，一次性计算所有 N 个波段的图像（`(Nbands, ny, nx)` 数组），大幅提升效率。由 `imagefitter_SED_allconst` 调用。

---

## 7. 拟合引擎（gsfit.py）

### 7.1 基类 `gsfitter`（行 55）

#### `cal_residual(θ)` (行 347) — 核心残差函数

```
总 χ² = Σ_图集 Σ_波段 Σ_像素 [(data - model) / σ]²
       + w_spec × Σ_光谱 Σ_λ [(data - model) / σ]²
```

执行流程：
1. 将 θ 解包为参数字典 `pardict`
2. 运行用户定义的约束函数
3. 更新 Galaxy 模型的子成分参数和红移信息
4. **逐波段循环**：
   - 调用 `gmodel.generate_image(band, PSF, resp)` → 物理流量模型图像
   - `im.phys_to_counts_rate` 转换为计数率单位
   - 加天光背景
   - 像素残差：`(data - model) / sigma`
   - 掩模坏像素、NaN 处理
5. **光谱残差**（如果有）：
   - 调用 `gmodel.fiducial_sed()` → 与 LSF 卷积
   - `(model - data) * weight_spec / σ`
6. 所有残差展平为单一向量返回

#### `loglike(θ)` (行 602)

JIT 编译：`-0.5 × Σ(residual²)`。假设独立高斯像素误差。

#### `lnprior(θ)` (行 622)

均匀先验（sigmoid 平滑边界）+ 可选高斯先验。

#### `lnprob(θ)` (行 576)

`logprior + loglike` — 贝叶斯后验（用于嵌套采样和 MCMC）。

### 7.2 Fitter 子类 — fitMode 路由

`gsutils.py:1827-1853` 根据数据特征自动选择 fitter 子类：

| fitMode | Fitter 类 (行号) | 使用场景 |
|---|---|---|
| `images - photometry` | `imagefitter_phot` (3512) | 纯形态拟合，每波段独立归一化（类 GALFIT） |
| `images - SED` + vectorize + allconstant | `imagefitter_SED_allconst` (3630) | 多波段 SED + 空间均匀 SED（**最快**） |
| `images - SED` + vectorize + ~allconstant | `imagefitter_SED` (4010) | 多波段 SED + 径向 SED 梯度（**最通用**） |
| `images - SED` + ~vectorize + allconstant | `imagefitter_SED_allconstNV` (3817) | 非向量化空间均匀 |
| `images + spectra - SED` | `gsfitter` 基类 (55) | 成像+光谱联合拟合 |
| `photometric - SED` | `sedfitter` (4309) | 仅测光点 SED 拟合（无空间信息） |
| `photometric + spectrum - SED` | `sedfitter` (4309) | 测光+光谱 SED 拟合 |
| `grism - SED` | `mbifitter` (3272) | 无缝光谱 (grism) + 成像 |
| `high resolution spectrum - SED` | `vdfitter` (4193) | 仅高分辨率光谱（含速度弥散） |

### 7.3 优化算法一览

| 方法 | 实现函数 | 行号 | 特点 |
|---|---|---|---|
| 梯度下降 | `optimizer()` | 930 | Optax (Adam/AdamW/SGD/Lion)，logit 空间无约束 |
| 多起点梯度 | `optimizer_multi()` | 1239 | pmap+vmap 多 GPU 并行，warmup+cosine 学习率 |
| 红移网格搜索 | `optimizer_zgrid()` | 1155 | 固定 z 网格点，每个点优化其他参数 |
| CMA-ES | `evolution_strategies()` | 1518 | evosax 实现，无梯度全局优化 |
| 多起点 ES | `evolution_strategies_multi()` | 2122 | Sep_CMA_ES + pmap 多 GPU |
| L-BFGS/SciPy | `jaxopt()` / `minimize()` | 2382 / 2554 | jaxopt 包装器 |
| 嵌套采样 | `nested_sampling()` | 2676 | Dynesty，完整贝叶斯后验+证据 |
| JAX 嵌套采样 | `jnesty_nested_sampling()` | 2845 | GPU 加速 JAX 原生实现 |
| MCMC | `mcmc()` | 3178 | NumPyro NUTS/HMC/HMCECS/SA |
| FlowMC | `flowmc()` | 3041 | RealNVP 归一化流 + HMC/GRW/MALA |

---

## 8. 二维面亮度轮廓（profiles.py）

所有轮廓函数均为 JAX JIT 编译，支持亚像素过采样（中心像素 2500 子像素）。

### 8.1 可用轮廓

| 轮廓类型 | 函数 | 行号 | 参数 |
|---|---|---|---|
| Sérsic | `sersic2D` | 187 | x, y, axrat, PA, Re, n, L |
| 截断 Sérsic | `sersic2D_broken` | 314 | + Rout |
| 环状 Sérsic | `sersic2D_ring` | 368 | + Rin, width |
| Ferrer | `Ferrer` | 423 | R_out, a, beta, L |
| 高斯环 | `GaussianRing` | 504 | r0, Sig, L |
| 边缘盘 | `EdgeonDisk` | 557 | rs, hs, L (Bessel K1 + sech²) |
| Sérsic + 螺旋 | `sersic2D_fourier` | 636 | + m, am, theta_m, i_arm |
| 环 + 螺旋 | `sersic2D_ringfourier` | 728 | 同上 |
| 截断 + 螺旋 | `sersic2D_brokenfourier` | 772 | 同上 |
| Ferrer + 螺旋 | `Ferrer_fourier` | 817 | 同上 |
| 高斯环 + 螺旋 | `GaussianRing_fourier` | 853 | 同上 |

### 8.2 轮廓与波长的关系

轮廓本身是纯形态学的——它们接受总光度 `L` 作为参数并将其空间分布。波长依赖性完全由 SED 模块处理：**SED 模板经过滤光片积分决定了每个波段每个成分的总光度 L，轮廓负责将 L 分布到像素上**。

### 8.3 PSF 函数

- `add_PSF()` (行 888)：在亚像素位置放置 PSF，使用 `jax.image.scale_and_translate` 三次插值
- `add_PSF_3D()` (行 904)：`vmap` 矢量化，支持波长依赖 PSF

### 8.4 几何工具

- `R_2d()` (行 103)：投影半径
- `R_3d()` (行 106)：倾斜解投影半径
- `R_fourier()` (行 130)：Fourier 螺旋调制半径
- `xpyp()` (行 126)：位置角旋转坐标
- `xy_3d()` (行 120)：倾斜平面坐标解投影

---

## 9. 配置系统（gsutils.py）

### 9.1 `read_config_file()` (行 498)

多波段 SED 拟合的关键配置解析流程：

**图像配置 `Ix1)-Ix15)`**：
- `Ia1)` 波段名
- `Ia2)-Ia7)` 图像文件、PSF、误差、掩模
- `Ia9)` 物理→计数率转换因子（若为 `-1` 则自动从 `phys_to_image` 字典查表）
- `Ia15)` SED 开关（1=参与 SED，0=仅形态）

**光谱配置 `Sx1)-Sx4)`**（可选）：
- `Sa1)` 光谱文件
- `Sa2)-Sa4)` 孔径定义、权重、LSF

**星系模型 `Px1)-Px32)`**：
| 参数 | 含义 | 物理范畴 |
|---|---|---|
| Px1-Px8 | x, y, mag, Re, n, axrat, PA... | 结构/形态 |
| Px9 | 特定恒星形成率 sSFR (Gyr⁻¹) | SED |
| Px10 | 星暴年龄 (Gyr) | SED |
| Px11 | 金属丰度 Z（Z☉=0.02） | SED |
| Px12 | 尘埃消光 Av (mag) | SED |
| Px13 | 恒星速度弥散 σ (km/s) | SED/运动学 |
| Px14 | log₁₀ 恒星质量 (M☉) | SED |
| Px15 | SFH 类型（conti/burst/bins） | SED |
| Px16 | 星云电离参数 logU | SED |
| Px26 | 2175Å 鼓包振幅 | SED（消光细节） |
| Px27 | SED 模式（0-4 或 T/Z） | SED |
| Px28-Px32 | logMdust, Umin, qPAH, alpha, gamma | 尘埃模型 |

**仪器单位转换** (`phys_to_image` 字典，行 47-150)：

涵盖 **29 个仪器**的滤光片到物理单位转换：GALEX (FUV/NUV)、Pan-STARRS (grizy)、SDSS (ugriz)、DESI、CGS、2MASS (JHKs)、WISE (ch1-4)、HSC (grizy)、DECam (ugrizy)、**JWST NIRCam (f070w–f480m，22 个滤光片)**、**JWST MIRI (f770w)**、Swift UVOT (uvw2/uvm2/uvw1)、XMM-OM、HST WFC3 (F275W–F160W)、Spitzer IRAC (ch1-4)。

**天体物理先验** (行 1559-1691)：
- **质量-尺寸关系 (MSR)**：van der Wel et al. 2014
- **质量-金属丰度关系 (MZR)**：Kewley & Ellison 2008 / Kewley & Dopita 2002
- **质量-消光关系**：Garn & Best 2010
- **能量平衡**：恒星 UV/光学吸收光功率 = 尘埃红外辐射功率（与 CIGALE 一致）
- **AGN 约束**：M_BH-M_bulge 关系、Hα/Hβ-L5100 相关性
- **SFH 形式约束**：指数衰减、延迟、双指数

### 9.2 `standard_display()` (行 2248)

生成多波段拟合的可视化输出：
- 逐波段数据/模型/残差三图拼接（行 2251-2401）
- SED 分解图（恒星、星云、尘埃、AGN 分量 + 宽带测光点）（行 2443-2490）
- 光谱拟合图（行 2491-2536）

---

## 10. 关键实现文件速查表

| 文件 | 行号 | 关键内容 |
|---|---|---|
| **gsfit.py** | 55 | `gsfitter` 基类定义 |
| | 177 | `__init__` — 参数初始化、自由度统计 |
| | 347 | `cal_residual()` — ★ 多波段 χ² 残差向量构造 |
| | 426 | `cal_model_image()` — 拟合后模型图像和 SED 生成 |
| | 576 | `lnprob()` — 对数后验 |
| | 602 | `loglike()` — JIT 对数似然 |
| | 622 | `lnprior()` — 均匀+高斯先验 |
| | 930 | `optimizer()` — 梯度下降优化 |
| | 1239 | `optimizer_multi()` — 多起点多 GPU 优化 |
| | 1518 | `evolution_strategies()` — CMA-ES |
| | 2676 | `nested_sampling()` — Dynesty 嵌套采样 |
| | 3041 | `flowmc()` — 归一化流 MCMC |
| | 3272 | `mbifitter` — 无缝光谱+成像 fitter |
| | 3512 | `imagefitter_phot` — 纯测光 fitter（无 SED） |
| | 3630 | `imagefitter_SED_allconst` — 均匀 SED fitter |
| | 3817 | `imagefitter_SED_allconstNV` — 非向量化均匀 SED |
| | 4010 | `imagefitter_SED` — 完整空间 SED fitter |
| | 4193 | `vdfitter` — 仅光谱 fitter |
| | 4309 | `sedfitter` — 仅测光 SED fitter |
| | 4471 | `mesfitter` — 多成分发射线 fitter |
| **galaxy.py** | 469 | `Galaxy` 类定义 |
| | 690 | `add_subC()` — ★ 添加子成分（结构+SED 参数） |
| | 1047 | `generate_mass_map()` — 二维质量分布图 |
| | 1667 | `generate_image()` — ★ 逐波段模型图像生成 |
| | 1946 | `generate_group_image()` — ★ 多波段批处理 |
| | 1335 | `fiducial_sed()` — 集成一维 SED |
| | 1478 | `generate_SED()` — 多波段测光 SED |
| | 1518 | `get_component_SED()` — 分量分解 SED |
| **sed_interp.py** | 1203 | `get_host_SED()` — ★ 恒星族 SED 生成 |
| | 1136 | `get_nebular()` — 星云连续谱+发射线 |
| | 837 | `get_cold_dust_sed()` — DL2014 尘埃热辐射 |
| | 1413 | `get_AGN_SED()` — AGN 薄吸积盘 |
| | 1473 | `sed_to_obse()` — 静止系→观测系转换 |
| | 1769 | `mycigale()` — Calzetti+2000 消光曲线 |
| | 609 | `ccm89()` — CCM89 银河系消光 |
| | 1010 | `conv_spec()` — 速度弥散卷积 |
| | 390 | `get_eazy_sed()` — EAZY PCA 模板组合 |
| | 277 | `TableModel` 类 — 自定义表格 SED |
| **images.py** | 1123 | `image` 类 — 单波段图像容器 |
| | 3128 | `image_atlas` 类 — 多波段图集 |
| | 3677 | `galfitS_data` 类 — 顶层数据容器 |
| | 3904 | `calculate_tranformation()` — 图像对齐和分组 |
| | 1277 | `img_cut()` — 图像 cutout 提取 |
| **gsutils.py** | 498 | `read_config_file()` — ★ 配置解析和 fitter 初始化 |
| | 47 | `phys_to_image` — 29 个仪器的单位转换 |
| | 957-985 | fitMode 自动检测 |
| | 1829-1853 | Fitter 子类路由 |
| | 2248 | `standard_display()` — 结果可视化 |
| | 1559-1691 | 天体物理先验施加 |
| **profiles.py** | 187 | `sersic2D()` — 标准 Sérsic |
| | 314 | `sersic2D_broken()` — 截断 Sérsic |
| | 423 | `Ferrer()` — 截断幂律（棒） |
| | 557 | `EdgeonDisk()` — 边缘盘 |
| | 636 | `sersic2D_fourier()` — Sérsic + 螺旋臂 |
| | 888 | `add_PSF()` — PSF 卷积 |
| **disperser.py** | 440 | `dispersion_ifu_with_flux_jax()` — 色散 IFU 建模 |
| | 597 | `GSDisperser` 类 — 无缝光谱色散器 |
| **Constant.py** | 1-10 | 物理常数（c, ckm, ms, π, FWHM 转换等） |
