# Video2PersonVideo

用 Ultralytics YOLO 检测视频中的人（COCO `person`，class 0），找到**画面中的主要人物**，
再按用户指定的长宽比裁剪输出，得到近似「电视节目 / 播客主播」的半身构图视频。
命令行与向导式图形界面两种用法，可用 PyInstaller 打包成免安装的 Windows exe。

## 功能特性

- **人像构图裁剪**：输出比例完全由用户决定（`9:16` / `16:9` / `1:1` / `4:5` / `21:9`…），
  裁剪框自动跟随画面中的主要人物，输出每一帧的宽高比恒等于所选比例
- **主角选择**：多人物场景下按「面积 + 靠近中心 + 时序连续（IoU）+ 置信度」打分，
  主角不会每帧乱跳；人物过小（bbox 高不足画面高的 `min_person_height_ratio`，默认 `0.73`）视为无人物
- **说话人跟随（多人物时对准正在说话的人）**：YOLO 本身判断不了"谁在说话"（纯图像模型，
  关键点里也没有嘴部），所以补上音频这一半 —— 抽音轨算**逐帧语音能量包络**，
  再看每个人的**嘴部运动**，谁的声音与嘴型最同步就跟谁（音频-视觉主动说话人检测）；
  带切换滞回，不会在两人之间来回跳；单人画面 / 无音轨 / 没有 ffmpeg 时自动退回上面的规则
- **多人分屏（多人各占一个上半身小窗口）**：画面里出现两个及以上主要人物时，
  每人给一个上半身小窗口**同时显示**，而不是整段视频只跟一个人；
  窗口怎么排由「输出长宽比 + 人数」查 `configs/multi_person_layout.yaml` 决定
  （竖屏 3 人 = 上通栏大窗 + 下排两个，横屏 3 人 = 左侧通高大窗 + 右侧上下两个…），
  每个窗口一律**只切该人物的上半身**（头顶 / 腰取姿态关键点，没有关键点时按人体比例估算），
  并按**自己的宽高比**取景，人物在画面里占比很小也不会把腿部与左右大片场景一起框进来，
  窄窗口切出来同样是半身构图、不会被拉伸变形；
  说话人（若启用了说话人跟随）默认坐 1 号大窗，其余人按画面位置依次入座 ——
  **左边的人就在左边的窗口**；默认开启，可在界面第 2 步取消勾选或用
  `--no-multi-person` 关掉（关掉 = 整段视频同时只显示一个人）；
  背景里的小人 / 路人（低于 `min_person_height_ratio`）不算主要人物，不进窗口
- **自动构图分档**：按人物占比自动判定头像 / 半身 / 全身，并处理人体被画面边缘截断的情况
- **姿态关键点精修（可选）**：模型换成 `yolo11n-pose.pt` 后，"半身"下边界取腰/髋、
  头顶用面部关键点纠正，抬手/戴帽子也不会把画面顶偏
- **镜头稳定（不易看晕）**：人物在画面内走动时取景框**完全不动**（自由活动区），
  越过边界后镜头仍受速度 / 加速度限幅（偏离越大越宽松，但只到 4 倍就平滑封顶，
  不会"人跑多快镜头就甩多快"），缩放比平移更慢更少 ——
  针对"人一走动镜头就跟着摇"的晕动感做了专门处理；
  镜头参数按**每秒感受**标定，并用源视频真实帧率换算，24 / 30 / 60fps 素材的
  镜头真实速度一致（60fps 不会比 30fps 快一倍）；
  另提供「锁定 / 舒缓 / 标准 / 跟手」四档镜头跟随预设（`--camera-follow`，默认「跟手」）
- **两级提速**：`--detect-interval N` 抽帧检测 + 线性插值，`--infer-batch N` 批量帧推理，
  两者结果与逐帧推理逐帧等价（有专门测试兜住）
- **无人物时的显示方式（默认「全画面适配」，让观众看到完整的"主屏幕"）**：短暂丢失目标
  时保持上一帧取景框；仍然无人（超出 `hold_frames`）时按 `no_person_mode` 显示：
  - `fit`（默认）**全画面适配**：整幅画面等比缩小**完整放进**输出画布，上下的留白用
    **同一帧的模糊放大版**填满（不是黑边、不引入画面之外的信息）—— 一帧看全、镜头
    **完全不移动**，从"人物取景框"到"整幅画面"用 `no_person_seconds` 秒平缓拉远；
  - `tiles` **全景 + 特写**：整幅画面收进上方通栏，下方纵向排列几个**次要小人物**
    （高度占比低于 `min_person_height_ratio`、又高于 `no_person_secondary_ratio` 的远景人物）
    的**上半身特写**，过渡后半段淡入 —— 每格承载不同信息（环境 + 被过滤掉的人），
    既保住全貌又看得清人；没有够大的小人物、或竖屏里排不下时自动退回全画面适配；
    纵向排列的是**不同内容**而不是同一画面的复制（同一时刻只有一幅画面）；
  - `scan` **空镜巡视**：取景框沿"源画面比取景框多出来的那一侧"缓慢往返，几个来回把被
    居中裁剪丢掉的左右 / 上下内容带到成片里（单程时长 `smoothing_scan_seconds`）；
  - `center` **画面居中**：直接取画面正中，最保守的一档
- **分辨率恒定**：输出尺寸只初始化一次（且宽高均为偶数），全程不出现丢帧、黑边或变形
- **体积守护**：OpenCV 只能写 `mp4v`，成品体积常是原视频的 3~10 倍；
  一旦「输出体积 > 原视频 × `max_size_ratio`」就自动按体积上限反推码率重压成 H.264
- **批量处理**：输入可以是文件夹（可递归），按扩展名白名单遍历、自动忽略 `*_person.*` 产物，
  单文件失败只记录错误并继续，绝不中断整批
- **向导式 GUI**：选择输入（含拖拽）→ 选择比例（网格卡片 + 自定义）→ 执行进度 → 结果汇总，
  处理前还能抽样几帧**预览构图效果**并调参数
- **时间轴归一化**：处理前先探测源视频的帧率 / 时长信息，可变帧率（VFR）源先用 ffmpeg
  把画面归一到恒定帧率再逐帧处理，根治「开头画面很慢、随后忽然变快」（声音正常）的音画不同步
- **首次运行显卡自检 + 依赖下载**：第一次打开界面先看清这台机器是什么卡
  （CPU 型号 / 内存 / 每张显卡的型号 · 显存 · 驱动 · 算力），据此推荐该装哪一套
  PyTorch（CUDA 11.8 / 12.1 / 12.4 / 12.6 / 12.8 / 12.9 / 13.0、ROCm、XPU、CPU），
  再带**断点续传**地把 wheel 与 YOLO 权重下下来（暂停 / 断网 / 关机都不白下）；
  **已有的内容不会重复下载**（本机装过 PyTorch / 已有权重时页面会写明并默认跳过），
  已经自检过、或本机环境本来就齐全时，这一步直接跳过
- **可视化安装包（Setup.exe）**：把项目当成普通 Windows 软件安装 —— 图形向导里选目录、
  按显卡挑 PyTorch、边装边下载（断点续传）；安装进度逐步落盘，任何一步中断都能
  「继续安装 / 暂停并退出 / 回滚并退出」，已经下好的部分绝不白下；
  装完自动建快捷方式、登记卸载信息，**全程不需要管理员权限**
- 音轨保留（ffmpeg 合并）、可选 H.264 重编码、含人帧截图、环境自检 `v2pv --check`
- 输出体积与「相对原视频的倍数」在结果汇总里如实报出，压不动也会明说

## 构图算法（一图流）

```
源帧 ──► YOLO 检测 ──► 选主角 ──► 构图分档 ──► 时序平滑 ──► 按比例取景框 ──► 切片 ──► 缩放到固定输出尺寸 ──► 写入
                       M1-2      M1-3         M1-4          不变量 3          不变量 4        不变量 2

音轨 ──────────► 语音能量包络 ┐
                             ├─► 谁在说话 ─► 主角打分的偏置 ─► （叠加到上面的"选主角"，M7）
画面中每人的嘴部运动 ────────┘
```

画面里**没有主要人物**时，"取景框"这条线换成"无人物显示"（见上一节的 `no_person_mode`）：

```
无人帧 ──► 保持 hold_frames 帧 ──► 仍无人 ──► 过渡进度 0→1（no_person_seconds 秒，缓入缓出）
                                            ├─ fit    整幅画面 → 全画面适配矩形（画布仍是 9:16）
                                            ├─ tiles  整幅画面 → 上方通栏 + 下方特写（淡入）
                                            ├─ scan   取景框沿多出来的那一侧缓慢往返
                                            └─ center 取景框回中
```

过渡的几何由 `core/layout.py:transition_frame` 统一给出：**源侧**从"人物取景框"连续变成
"整幅画面"，**画布侧**从"整块画布"连续变成"目标矩形"，两边的宽高比在每一帧都相等
（比例本身也在插值），因此中间任何一帧都**不拉伸、不越界、不出现空洞**。

多人物场景的"选主角"有两条通路：默认按面积 / 居中 / 时序 / 置信度打分；
开启**说话人跟随**后，会先判定"现在谁在说话"（音频-视觉主动说话人检测），
把结果作为额外偏置叠加到打分上 —— 于是"正在说话的人"能压过"个子更大 / 更居中"的人。
两路信号都不需要额外模型：语音来自 ffmpeg 抽出的音轨，嘴部运动来自画面本身。
细节见 `core/audio.py` / `core/mouth.py` / `core/speaker.py`。

开启**多人分屏**（默认）后，判定的对象从"选一个人"变成"排一屋子人"：

```
多个人物 ──► 过滤过小人物 ──► 各自打分 ──► 取前 N 个 ──► 排窗口 ──► 逐人取景 ──► 拼接
            M1-2             M1-2         max_persons   布局表       各自的      不变量 2
                                                          + 说话人     平滑器
```

* **取谁**：过滤掉过小人物的，按主角打分取前 `multi_person_max` 个（默认 4）；
* **排哪**：布局表按「输出比例 + 人数」给出网格（`1` 号窗口给说话人 / 主角，
  其余按空间位置入座）；表里查不到的（人数超出、自定义比例）按宽高比自动生成网格；
* **怎么稳**：每个人有自己的"座位"（跨帧轨迹 + 独立平滑器），不会因为旁边的人动了而晃；
  人数偶尔抖动会滞后几帧再换布局，窗口不会闪；
* **切多紧**：每个窗口都按"**恰好装下该人物的上半身**"取景（
  `core/framing.py:compute_upper_body_box`）—— 不管人物在画面里是大是小，都先定出
  "头顶 → 腰"的上半身范围，再分别按"上半身高度"与"人物宽度"算两个候选框、取较大的那个，
  于是两个方向都只留一层薄薄的留白。远景小人物会被放大成半身，而不是连着腿和左右场景一起框；
* **怎么不出丑**：格子小到看不清（短边 < 96 像素）时宁可少显示几个人；
  每格按自己的比例取景，所以窄窗口里的半身也不会变形。

细节见 `core/layout.py`（布局表与像素划分）与 `core/multi.py`（时序合成）。

一句话总结：**输出尺寸是常量，取景框是变量。** 用户选的是「输出长什么样」，
算法调的是「从原片里切哪一块」。

几条贯穿全项目的硬性约定（实现与测试都以此验收）：

| 不变量 | 含义 | 保障方式 |
|---|---|---|
| 1 | 输出每一帧的 `宽/高` 恒等于所选比例 | 输出尺寸由 `AspectRatio` 固定给出（比例 + 目标像素） |
| 2 | 输出分辨率全程恒定，宽高均为偶数 | 取景框切片后**统一缩放**到固定目标尺寸；`VideoWriter` 只初始化一次 |
| 3 | 取景框自身比例恒定 | 先平滑中心点，再按比例反推宽高，不分别独立平滑宽高 |
| 4 | 不出现黑边 / 变形 / 越界 | 先 `clamp`/`fit` 再切片，不做 letterbox、不做非等比缩放 |

> 不变量 3 有一条**明确标注的例外**：没有主要人物且显示方式为 `fit` / `tiles` 时不存在
> "取景框"，`CropBox` 被借用来表示"从源画面取出的那一块"（比例从目标比例连续过渡到源
> 比例），真正的合成由 `core/crop.py` 完成。不变量 1 / 2 / 4 在这一档同样成立：
> 画布仍是 9:16、尺寸仍是偶数且恒定、画面完整放入并居中（留白由模糊底填充，不是黑边）。

模块划分（`core/` 里的几何与打分都是纯函数，可脱离视频单测）：

| 模块 | 职责 |
|---|---|
| `core/ratio.py` | 比例定义、解析、校验、目标像素推导 |
| `core/framing.py` | `CropBox`、构图分档、`compute_target_box` / `compute_upper_body_box`、插值 |
| `core/pose.py` | COCO-17 关键点 → 头顶 / 腰部锚点（可选精修半身构图） |
| `core/subject.py` | 主角打分与选择、过小阈值过滤、说话人偏置融合 |
| `core/audio.py` | 音轨 → 逐帧语音能量包络 + 语音活动判定（说话人跟随的音频侧） |
| `core/mouth.py` | 嘴部 ROI 定位与嘴动强度测量（说话人跟随的视觉侧） |
| `core/speaker.py` | 主动说话人判定：轨迹关联 + 相关性 + 切换滞回 |
| `core/layout.py` | 分屏布局表与几何：字符网格解析、按比例 / 人数查表、像素划分、全画面适配矩形、无人物过渡几何、全景 + 特写网格 |
| `core/multi.py` | 多人分屏合成：选人排座、逐人取景、轨迹关联与逐人平滑 |
| `core/noperson.py` | 无人物帧的显示：全画面适配的外观参数、「全景 + 特写」的选人与排布 |
| `core/smoothing.py` | 取景框时序平滑、自由活动区、速度 / 加速度限幅、帧率归一化、镜头档位预设、保持 N 帧 |
| `core/crop.py` | `frame + CropBox → 固定尺寸输出帧` |
| `core/pipeline.py` | 编排：检测 → 选主角 → 构图 → 平滑 → 裁剪，含抽帧插值 |
| `core/batch.py` | 扫描路径、任务队列、失败隔离、两级进度 |
| `core/detector.py` | YOLO 人像检测器：模型加载、推理、画框（结构化 bbox / 关键点） |
| `core/video_io.py` | 视频底层读写：读帧预读 + 写帧异步（后台线程）、截图 |
| `core/processor.py` | 单视频处理流水线：检测 → 选主角 → 构图裁剪 → 写视频 → 补回音轨 + 统计结果 |
| `utils/gpu_probe.py` | 硬件自检：CPU / 显卡型号 → 厂商 · 架构 · 算力（型号库覆盖主流型号） |
| `utils/torch_backends.py` | PyTorch 构建选择表：算力 + 驱动 + 系统 → 该装那一套（纯函数） |
| `utils/downloader.py` | 断点续传下载：`Range` 续传 + `sha256` 校验 + 重试 + 可取消 |
| `utils/torch_install.py` | 解析官方索引 → 挑 wheel → 生成下载计划 → `pip` 安装 |
| `utils/env_check.py` | 本机环境速查：PyTorch / YOLO 权重是否已在本地（不 `import`、不联网，用于决定要不要弹下载页） |
| `utils/app_paths.py` | 安装布局与随包资源定位（安装根识别、模型 / 随包 ffmpeg 目录、随包工具挂到 `PATH`） |
| `utils/device.py` | 推理设备探测（CUDA / MPS / CPU）与环境自检报告（`v2pv --check` 的内容） |
| `utils/ffmpeg_tools.py` | ffmpeg / ffprobe 调用：音轨探测与合并、H.264 重编码、媒体信息（带结果缓存） |
| `utils/logger.py` | 统一日志配置（命令行与 GUI 共用，GUI 可把日志转发到界面） |
| `gui/framing_sketch.py` | 自绘示意图：比例页右侧按当前档位画出「无人时 / 多人分屏」输出长什么样 |

## 目录结构

```
Video2PersonVideo/
├── pyproject.toml              # 项目元数据、依赖、入口脚本、工具配置
├── requirements*.txt           # 运行期 / GUI / 开发依赖
├── configs/default.yaml        # 全部可调参数（含裁剪构图参数与注释）
├── configs/multi_person_layout.yaml  # 多人分屏布局表（按比例 + 人数查表）
├── build/Video2PersonVideo.spec        # 主程序 PyInstaller 打包配置
├── build/Video2PersonVideo-Setup.spec  # 安装器（Setup.exe）打包配置
├── data/input|output/          # 素材与（可选）统一输出目录
├── assets/models|samples/      # 权重与演示素材（默认不提交）
├── scripts/                    # 一键建环境 / 打包 / 打包入口
├── src/video2personvideo/
│   ├── cli.py                  # 命令行参数解析与主流程（文件 / 文件夹）
│   ├── config.py               # AppConfig：默认值 / YAML 加载 / 参数校验 / M0 输出决策
│   ├── core/
│   │   ├── ratio.py            # 比例与目标像素
│   │   ├── framing.py          # 取景框与构图策略
│   │   ├── pose.py             # 姿态关键点 → 构图锚点（可选）
│   │   ├── subject.py          # 主角选择
│   │   ├── audio.py            # 音轨 → 语音能量包络与语音活动
│   │   ├── mouth.py            # 嘴部 ROI 与嘴动强度
│   │   ├── speaker.py          # 主动说话人判定（轨迹 + 相关性 + 滞回）
│   │   ├── layout.py           # 分屏布局与几何（网格解析 / 查表 / 像素划分 / 过渡几何）
│   │   ├── multi.py            # 多人分屏合成（选人排座 / 逐人取景 / 平滑）
│   │   ├── noperson.py         # 没有主要人物时的显示（全画面适配 / 全景 + 特写）
│   │   ├── smoothing.py        # 时序平滑
│   │   ├── crop.py             # 切片 + 缩放
│   │   ├── pipeline.py         # 单视频裁剪流水线
│   │   ├── batch.py            # 批量任务编排
│   │   ├── detector.py         # YOLO 检测（结构化 bbox / 画框 / 计数）
│   │   ├── video_io.py         # 视频读取、写入、截图
│   │   └── processor.py        # 处理流水线 + 统计结果
│   ├── gui/
│   │   ├── app.py              # Qt 应用启动（高 DPI + 主题 + 首次运行自检）
│   │   ├── wizard.py           # 四步向导主窗口
│   │   ├── main_window.py      # 主窗口别名（兼容既有导入与打包配置）
│   │   ├── controller.py       # GUI ↔ 后台任务桥接（批处理 / 环境自检下载）
│   │   ├── setup_dialog.py     # 首次运行的环境自检对话框（含跳过条件，也用于主界面按钮）
│   │   ├── theme.py            # 统一配色 / 样式 / 尺寸
│   │   ├── ratio_grid.py       # 比例网格卡片
│   │   ├── ratio_dialog.py     # 比例选择对话框
│   │   ├── preview_dialog.py   # 处理前构图预览
│   │   ├── framing_sketch.py   # 构图参数自绘示意图（比例页右侧"无人时输出长什么样"）
│   │   └── pages/              # 输入 / 比例 / 执行 / 结果 / 环境自检页
│   ├── installer/              # 可视化安装器（打包成 Setup.exe，与主程序分开）
│   │   ├── app.py              # 安装器入口（GUI / 命令行分发、日志落盘）
│   │   ├── cli.py              # 参数解析：--silent / --uninstall / --list-backends…
│   │   ├── stages.py           # 安装阶段模型（权重 / 可否续传 / 中断说明）
│   │   ├── journal.py          # 安装状态日志 install.json（原子写 + 镜像 + 续装判定）
│   │   ├── engine.py           # 安装引擎：部署运行时 → pip → PyTorch → 依赖 → 本体 → YOLO → ffmpeg → 收尾
│   │   ├── runtime.py          # 内嵌 Python（embeddable）下载 / 解压 / 修 ._pth
│   │   ├── payload.py          # 随安装器分发的程序本体与依赖清单
│   │   ├── options.py          # 组件选择（PyTorch 构建 / 权重 / ffmpeg / 快捷方式）
│   │   ├── windows.py          # 快捷方式、卸载注册表、目录自删、启动入口
│   │   ├── paths.py            # 安装目录布局与系统位置
│   │   └── gui/                # 安装向导（欢迎 / 位置 / 组件 / 进度 / 完成 + 卸载对话框）
│   └── utils/
│       ├── logger.py           # 统一日志配置
│       ├── device.py           # 推理设备探测（CUDA / MPS / CPU）与环境自检报告
│       ├── app_paths.py        # 安装布局 / 资源定位（安装根、模型目录、随包 ffmpeg）
│       ├── env_check.py        # 本机环境速查（PyTorch / 权重是否已在本地，不联网）
│       ├── gpu_probe.py        # 硬件自检：CPU / 显卡型号 → 厂商 · 架构 · 算力
│       ├── torch_backends.py   # PyTorch 构建选择表（算力 + 驱动 + 系统）
│       ├── torch_install.py    # 官方索引解析 / wheel 挑选 / 下载计划 / pip 安装
│       ├── downloader.py       # 断点续传下载（Range + sha256 + 重试 + 可取消）
│       └── ffmpeg_tools.py     # ffmpeg / ffprobe 调用（音轨、转码、媒体信息、体积）
└── tests/                      # pytest：单元 + 不变量 + 端到端 + GUI 冒烟 + 安装器
```

## 环境要求

- Python 3.11 ~ 3.13（本机实测 3.13）
- **用安装包（Setup.exe，仅 Windows）安装时不需要上面这些**：安装器自带一份独立的 Python 运行时
- Windows / Linux / macOS 均可；打包 exe 面向 Windows
- 可选：NVIDIA 显卡 + 驱动（开启 CUDA 推理，速度提升明显）
- 可选：`ffmpeg` 并加入 PATH（用于保留音轨、可变帧率时间轴归一化、H.264 重编码、
  体积守护自动压缩、元数据标记，以及**说话人跟随的音轨分析**）
  下载地址 <https://www.gyan.dev/ffmpeg/builds/> 或 `winget install Gyan.FFmpeg`

## 快速开始

三种装法，按需要挑一种（装完都是同一个东西：可用的 `v2pv` 命令 + 可选图形界面）：

| 方式 | 适合谁 | 需要预装什么 | 见 |
|---|---|---|---|
| **安装包 `Video2PersonVideo-Setup.exe`** | 最终用户（Windows），只想用 | 什么都不用（自带 Python、免管理员权限） | 第 0 步 /「安装成 Windows 软件」 |
| **一键脚本 `scripts\setup_env.ps1`** | 开发者（Windows） | Python 3.11~3.13 | 第 1 步 |
| **手动 pip 安装** | 开发者（Windows / Linux / macOS） | Python 3.11~3.13 | 第 2 步 |

### 0. 像装软件一样装（最终用户，推荐）

下载 `Video2PersonVideo-Setup.exe` 双击，按向导走完即可
（不需要预装 Python、不需要管理员权限）。细节见「安装成 Windows 软件」一节。

下面 1~5 是开发者 / 高级用户从源码跑的用法。

### 1. 一键安装（推荐）

```powershell
# 无独显 / 只用 CPU
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1

# NVIDIA 显卡 + CUDA（脚本会从 PyTorch 官方源装 CUDA 版 torch）
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -Backend cuda
```

脚本会把「建虚拟环境 → 装 PyTorch → 装其余依赖 → `pip install -e .` → 跑一次 `--check`」
一条龙做完。可用参数：

| 参数 | 作用 | 默认值 |
|---|---|---|
| `-Backend cpu\|cuda` | 装 CPU 版还是 CUDA 版 PyTorch | `cpu` |
| `-TorchIndexUrl <url>` | CUDA 版 torch 的索引地址（要换 CUDA 版本时改它） | `https://download.pytorch.org/whl/cu129` |
| `-VenvDir <dir>` | 虚拟环境目录（已存在则直接复用） | `.venv` |
| `-Dev` | 额外装开发 / 打包依赖（ruff、pyinstaller、auto-py-to-exe） | 关 |

### 2. 手动安装

Windows（PowerShell）：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

# ① 先装 PyTorch（CPU：pip install torch torchvision；
#    CUDA：pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129）
pip install torch torchvision

# ② 装其余依赖 + 本项目
pip install -r requirements.txt
pip install -r requirements-gui.txt   # 需要图形界面时
pip install -e .
```

Linux / macOS（bash / zsh）：

```bash
python3 -m venv .venv
source .venv/bin/activate

# Apple 芯片装 CPU 版即可（自带 Metal / MPS 加速）；NVIDIA 卡换官方 CUDA 索引地址
pip install torch torchvision

pip install -r requirements.txt
pip install -r requirements-gui.txt   # 需要图形界面时
pip install -e .
```

> `scripts\*.ps1`（一键安装 / 打包）是 Windows 专用；Linux / macOS 走上面的手动步骤，
> 或改用 `pip install -e ".[gui]"`（等价于 `requirements-gui.txt`）。
> **先装 PyTorch 再装其余依赖**：`ultralytics` 也会拉 torch，顺序反了会装成 CPU 版。

### 3. 跑一次

```powershell
# 横屏视频裁成竖屏 9:16（默认输出 <原名>_person.<原后缀>，与原视频同目录）
v2pv -i data\input\demo.mp4

# 指定比例与目标像素
v2pv -i demo.mp4 --ratio 4:5 --target-size 1080x1350

# 整目录批量处理（递归），已存在的输出自动跳过
v2pv -i data\input --recursive --ratio 16:9

# 只看环境是否就绪 / 打开图形界面
v2pv --check
v2pv --gui
```

首次运行会自动下载 `yolo11n.pt`（约 5MB），日志里会有提示；也可以手动下载后放入
`assets/models/`，再用 `-m assets/models/yolo11n.pt` 指定。

想把半身构图切得更准，把模型换成姿态模型即可（会自动使用关键点，无需其他配置）：

```powershell
v2pv -i demo.mp4 -m yolo11n-pose.pt    # 6MB，自动下载
```

### 4. 验证安装

```powershell
v2pv --version      # 版本号
v2pv --check        # 环境自检：设备 / torch / CUDA 是否可用 / ffmpeg / 默认权重
```

`--check` 是判断"装对没有"的权威口径，重点看两行：

- `CUDA 可用: 是` —— 才说明装的是 GPU 版 torch；显示 `否` 说明装成了 CPU 版，
  按第 1 / 2 步换 CUDA 索引地址重装（也可以打开界面点左下角「环境自检」让它自动判断）；
- `ffmpeg` —— 显示路径即已就绪；显示「未找到…」说明 PATH 里没有 ffmpeg，
  音轨保留 / 时间轴归一化 / 体积守护 / 说话人跟随都会退化成"跳过"，出片功能不受影响。

图形界面：`v2pv --gui`（首次打开可能先进「环境自检」，见后面一节）。

### 5. 升级 / 卸载

源码安装（`pip install -e .`）：

```powershell
git pull
pip install -r requirements.txt   # 依赖有变动时
pip install -e .                  # 重新登记入口点（升级后必做）

pip uninstall video2personvideo   # 卸载；接着删掉 .venv 或整个项目目录即可
```

用安装包装的（Windows）见「卸载」一节：开始菜单 / 设置 → 应用 / 安装目录里的
`Uninstall.cmd` 三个入口等效，卸载器还会顺带清掉快捷方式与注册表项。

## 命令行参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `-i, --source` | 输入视频**或**视频文件夹 | 必填（`--check`/`--gui` 除外） |
| `-o, --output` | 输出视频路径 | `<原视频同目录>/<原名>_person<原后缀>` |
| `-c, --config` | YAML 配置文件 | `configs/default.yaml` |
| `-m, --model` | 模型名或 `.pt` 路径 | `yolo11n.pt` |
| `--classes` | 保留的类别索引，`-1` 表示全部 | `0`（person） |
| `--conf` / `--iou` | 置信度 / NMS 的 IoU 阈值 | `0.25` / `0.45` |
| `--imgsz` / `--max-det` | 推理分辨率 / 单帧目标上限 | `640` / `300` |
| `--device` | `auto` / `cpu` / `cuda:0` / `mps` | `auto` |
| `--ratio` | 目标长宽比：`9:16`、`4:5`、`1080x1920`… | `9:16` |
| `--target-size` | 目标像素 `宽x高`（宽高需为偶数） | 按短边 1080 推导 |
| `--[no-]crop` | 启用裁剪构图（`--no-crop` = 旧的逐帧画框标注） | 启用 |
| `--annotate` | 在输出帧上叠加检测框 | 关 |
| `--detect-interval` | 每 N 帧推理一次，中间帧线性插值 | `1` |
| `--infer-batch` | 一次推理塞几帧（批量帧推理提速） | `1` |
| `--no-keypoints` | 即使模型自带姿态关键点也不用于精修构图 | 关（自动启用） |
| `--camera-follow` | 镜头跟随档位：`lock` / `calm` / `standard` / `active`（一次设置一组平滑参数） | `active` |
| `--deadzone` | 自由活动区（相对框长）：人物在框内走动小于它时镜头**完全不动** | `0.06` |
| `--smoothing-alpha` | 平移平滑系数（越大越跟手） | `0.45` |
| `--pan-speed` | 平移速度上限（相对框长/帧，防甩镜；`0` = 不限） | `0.05` |
| `--zoom-alpha` | 缩放（推拉）平滑系数，越小越不易晕 | `0.15` |
| `--zoom-deadzone` | 缩放死区（相对框高） | `0.05` |
| `--hold-frames` | 丢失目标后保持取景框的帧数 | `30` |
| `--no-person-mode` | 没有主要人物时怎么显示：`fit`（全画面适配）/ `tiles`（全景 + 特写）/ `scan`（空镜巡视）/ `center`（画面居中） | `fit` |
| `--no-person-seconds` | 从"人物取景框"过渡到无人物显示的时长（秒）；`0` = 立即切换 | `0.8` |
| `--[no-]blur-background` | 无人物画面的留白用"同一帧的模糊放大版"填充（关闭 = 用纯色底色） | 开 |
| `--no-person-tiles-max` | 「全景 + 特写」最多给几个远景小人物开特写窗口（1~4） | `2` |
| `--no-person-secondary-ratio` | 「全景 + 特写」里小人物的高度下限（占画面比例），低于它不给特写 | `0.12` |
| `--scan-seconds` | 「空镜巡视」的单程时长（秒），只在 `--no-person-mode scan` 时用得上；`0` = 关闭、退回居中 | `12.0` |
| `--min-person-ratio` | 人物过小阈值（bbox 高 / 画面高）：低于它的人算背景路人 | `0.73` |
| `--headroom` | 半身构图的头顶留白比例 | `0.08` |
| `--[no-]speaker-tracking` | 多人物时优先对准**正在说话的人**（需要 ffmpeg 抽音轨） | 启用 |
| `--speaker-weight` | 说话人偏置大小（与归一化后的主角打分同量纲，≥0；≥1.0 时"正在说话的人"必定胜出，<1.0 则是"软优先"） | `1.8` |
| `--speaker-switch-margin` | 换人所需的最低领先分差 | `0.15` |
| `--speaker-switch-hold` | 换人前挑战者需连续领先的判定回合数（越大镜头越稳） | `8` |
| `--speaker-window` | 语音-嘴动的相关性滑窗长度（帧），约 1~1.5 秒 | `45` |
| `--[no-]multi-person` | 多人分屏：多人时每人一个上半身小窗口（关掉 = 同时只显示一个人） | 开 |
| `--multi-person-max` | 分屏最多同时显示几个人（2~9） | `4` |
| `--multi-person-order` | 分屏排列：`spatial`（先说话人再按位置）/ `score`（按主角打分） | `spatial` |
| `--layout-config` | 多人分屏的布局表配置 | `configs/multi_person_layout.yaml` |
| `--no-audio` | 不保留原视频音轨 | 关 |
| `--reencode` / `--crf` | H.264 重编码 / 质量 | 关 / `23` |
| `--[no-]normalize-vfr` | 可变帧率源先归一时间轴（避免画面忽快忽慢） | 开 |
| `--[no-]size-guard` | 输出体积超过原视频时自动压缩 | 开 |
| `--max-size-ratio` | 输出体积上限（相对原视频的倍数） | `1.0` |
| `--min-bitrate` | 自动压缩时的视频码率下限（kbps） | `250` |
| `--output-suffix` | 强制输出后缀（如 `.mp4`） | 沿用原后缀 |
| `--[no-]same-dir` | 输出到原视频同目录 | 是 |
| `--overwrite` | 同名输出存在时覆盖（默认跳过） | 关 |
| `--metadata` | 写入比例 / 尺寸 / 模型 / 时间元数据 | 关 |
| `--write-preview` | 额外输出首帧构图预览图 | 关 |
| `--save-frames` | 含人帧的截图输出目录 | 关 |
| `--[no-]recursive` | 文件夹输入时递归遍历 | 是 |
| `--show` | 处理时弹窗实时预览（按 `q` 结束） | 关 |
| `--log-level` | `DEBUG`/`INFO`/`WARNING`/`ERROR` | `INFO` |
| `--check` / `--gui` / `--version` | 环境自检 / 图形界面 / 版本 | - |

调参小抄：

- 漏检多人/小目标 → 调低 `--conf`、调大 `--imgsz`（如 960）
- 误检多 → 调高 `--conf`（如 0.4~0.5）；人挤人框重叠 → 调低 `--iou`（如 0.3）
- 人物一走动镜头就跟着摇、看着头晕 → `--camera-follow lock`；只想稍稳一点用 `calm`，
  想更跟手用 `active`。也可以逐项调：**调大 `--deadzone`**（人物更自由地在画面内走动、
  镜头不动）、**调小 `--pan-speed`**（镜头移动更慢）、调小 `--smoothing-alpha`
- 背景里的路人 / 小人也被当成主角 → 调高 `--min-person-ratio`（默认 `0.73`，只认画面里占比够大的人）
- 没有人物时空镜"看着空"或"切得太狠" → 用 `--no-person-mode` 换档：
  想看到完整画面（推荐）保持默认 `fit`；想让远景小人物也看得清用 `tiles`
  （配 `--no-person-tiles-max` / `--no-person-secondary-ratio` 调数量与门槛）；
  喜欢"镜头慢慢扫过去"的观感用 `scan`（再配 `--scan-seconds` 调快慢）；
  只想稳稳取画面中就 `center`。过渡生硬 / 突兀 → 调大 `--no-person-seconds`
- 全画面适配时觉得上下留白太抢眼 → `--no-blur-background` 换成纯色底，或
  `--no-person-mode tiles` 把纵向空间用起来
- 人物头顶总是被切 → 调大 `--headroom`
- 多人对话时镜头跟错了人 / 切换太频繁 → 调大 `--speaker-switch-hold`（换人更慎重）、
  调大 `--speaker-switch-margin`；判定太"强势"（偶尔被噪声带偏）→ 调小
  `--speaker-weight`（如 `0.4`，变成"软优先"）；想彻底关掉用 `--no-speaker-tracking`
- 多人画面里小窗口太密、看不清脸 → 调小 `--multi-person-max`（如 `--multi-person-max 2`），
  或提高 `--target-size`（分屏后每格的有效像素取决于它）；程序本身在格子短边
  不足 96 像素时也会自动少显示几个人
- 分屏的窗口排得不好看 → 改 `configs/multi_person_layout.yaml` 里对应比例 / 人数的
  网格（如竖屏 3 人想让主角窗更大就调 `rows` 权重），改完用「预览构图效果」直接看
- 只想跟一个人，不要分屏 → `--no-multi-person`（或界面第 2 步取消勾选「多人分屏」）
- 长视频太慢 → 上 GPU，或 `--detect-interval 3`（抽帧 + 插值）、`--infer-batch 4`（批量帧推理）
- 成品体积想更小 → 调小 `--target-size`，或 `--max-size-ratio 0.5`（体积守护会算好码率）

## 配置文件

`configs/default.yaml` 覆盖**全部**可调参数（模型与推理、裁剪构图、说话人跟随、多人分屏、
镜头平滑、输出策略、批量……），每个字段都带中文注释；下面只摘录最常调的一部分，
命令行优先级更高：

```yaml
model: yolo11n.pt
aspect_ratio: "9:16"        # 目标长宽比
target_width: null          # 自定义目标像素（两者同时给出才生效）
target_height: null
crop: true                  # 关闭则回到旧的逐帧画框标注模式
overwrite: false            # 同名输出存在时跳过
smoothing_deadzone: 0.06    # 自由活动区：人物在框内走动小于它时镜头完全不动
smoothing_alpha: 0.45       # 镜头跟进快慢
smoothing_max_speed: 0.05   # 平移速度上限（相对框长/帧），防止甩镜
smoothing_accel: 0.5        # 起步/反向的柔和度（越小越柔和）
smoothing_zoom_alpha: 0.15  # 缩放（推拉）比平移更慢
smoothing_zoom_deadzone: 0.05
hold_frames: 30             # 短暂丢人时保持取景框的帧数
no_person_mode: fit         # 没有主要人物时怎么显示：fit / tiles / scan / center
no_person_seconds: 0.8      # 从人物取景框过渡到无人物显示的时长（秒），0 = 立即切换
no_person_blur: true        # 无人物画面的留白用同一帧的模糊放大版填充
no_person_tiles_max: 2      # 「全景 + 特写」最多给几个远景小人物开特写窗口（1~4）
no_person_secondary_ratio: 0.12  # 「全景 + 特写」里小人物的高度下限（占画面比例）
smoothing_scan_seconds: 12.0  # 「空镜巡视」单程时长（秒），只在前一项为 scan 时用得上
detect_interval: 1          # 每 N 帧推理一次
infer_batch: 1              # 一次推理塞几帧（批量帧推理提速）
speaker_tracking: true      # 多人物时优先对准正在说话的人（需 ffmpeg 抽音轨）
speaker_weight: 1.8         # 说话人偏置在主角打分里的权重
speaker_window_frames: 45   # 语音-嘴动的相关性滑窗（帧）
speaker_switch_margin: 0.15 # 换人所需的最低领先分差
speaker_switch_hold: 8      # 换人前挑战者需连续领先的判定回合数
output_same_dir: true       # 输出到原视频同目录
size_guard: true            # 输出体积超过原视频时自动压缩
max_size_ratio: 1.0         # 体积上限 = 原视频 × 该倍数
min_video_bitrate_kbps: 250 # 自动压缩的码率下限，防止压得过糊
multi_person: true          # 多人分屏：多人时每人一个上半身小窗口（关掉 = 只显示一个人）
multi_person_max: 4         # 分屏最多同时显示几个人（2~9）
multi_person_order: spatial # spatial = 先说话人再按位置；score = 按主角打分
multi_person_layout_file: null   # 布局表路径；null = configs/multi_person_layout.yaml
```

用别的配置：`v2pv -c configs/my.yaml -i data\input\demo.mp4`

### 多人分屏的布局表

`configs/multi_person_layout.yaml` 只描述一件事：**「输出长宽比 + 人数」→ 窗口怎么排**。
它是可选的 —— 文件不存在时用内置默认（内容与仓库里的表一致），所以删掉它也不会出问题。

布局用字符网格描述，一个字符串 = 网格的一行，数字 = 窗口编号，
同一个数字连续占多格就是合并单元格（大窗口）：

```yaml
layouts:
  portrait:                 # 竖屏（9:16 / 4:5 / 3:4 / 2:3）
    2: ["1", "2"]           # 上下两格
    3: ["11", "23"]         # 上通栏大窗 + 下排两个
    4: ["12", "34"]         # 2×2
  landscape:                # 横屏（16:9 / 21:9 / 2:1 …）
    2: ["12"]               # 左右两格
    3: ["12", "13"]         # 左侧通高大窗 + 右侧上下两个
```

* **编号就是优先级**：`1` 号窗口给"正在说话的人"（没开说话人跟随时给主角），
  其余人按画面位置（先上后下、先左后右）依次入座 —— 左边的人就在左边的窗口；
* **键可以是族名或具体比例名**，`"9:16"` 会覆盖 `portrait`，方便给某个比例单独定制；
* 每个档位也可以写完整形式给行列加权：
  `3: {grid: ["12", "13"], cols: [1.4, 1.0]}`（左列更宽）；
* 表里查不到的（人数超出、自定义比例如 `7:8`）会按宽高比自动生成一个尽量方的网格，
  所以任何输入都不会崩，只是排得不如查表精致；
* 全局项还有 `gap_ratio`（窗口缝隙）、`background`（缝隙底色）、
  `switch_hold`（人数变化滞后几帧才换布局），以及 `groups`（把某个比例归到
  `portrait` / `square` / `landscape` 哪个族；没写的比例按宽高比自动归类）。

## 图形界面

```powershell
pip install -r requirements-gui.txt
v2pv --gui
```

首次打开可能先进入「环境自检 · 显卡自检与依赖下载」（见下一节；已经自检过、
或本机依赖本来就齐全的用户会直接跳过），之后主界面左下角的「环境自检」按钮可以随时重做。
向导本身分四步：

1. **选择输入**：视频文件或文件夹（二选一），支持拖拽；选文件夹后显示「共发现 N 个视频」
   预览列表，并自动忽略 `*_person.*` 产物
2. **选择比例**：网格卡片直观展示比例形状，也可输入自定义 `W:H`（带实时校验与分辨率预估）；
   可微调**镜头跟随档位**（锁定 / 舒缓 / 标准 / 跟手）、人物最小占比、头顶留白，
   可勾选「多人分屏 · 每人一个上半身窗口」（**默认勾选**；取消勾选 = 整段视频同时只显示一个人）
   与「优先对准正在说话的人」（说话人跟随，需 ffmpeg），
   并用「没有人物时」下拉选择空镜那一段怎么显示（**默认「全画面适配」**），
   并点「预览构图效果」抽样几帧看裁剪前后对照（预览里也能直接试这几个开关）；
   底部「输出体积 · 自动控制输出体积」勾选项即体积守护开关
   —— 示意图右侧会按当前档位画出"无人时输出长什么样"，不用跑一遍也能看出来
3. **执行处理**：整体 + 当前文件两级进度条，显示帧进度、裁剪模式、FPS 与剩余时间；
   「取消处理」在当前帧处理完后生效，输出要么完整要么不存在
4. **结果汇总**：成功 / 跳过 / 失败列表与原因，一键打开输出文件夹

界面会记住上次的输入路径与比例（`QSettings`）。所有耗时操作都在 `QThread` 里跑，界面不卡。
深浅色主题默认跟随系统，也可以用环境变量强制指定：

```powershell
$env:V2PV_THEME = "dark"   # light / dark / auto（默认）
v2pv --gui
```

## 首次运行 · 显卡自检与依赖下载

第一次打开界面可能会先弹出这一步（之后可在主界面左下角「环境自检」随时重做）。
它解决一个很实际的问题：**PyTorch 官方源里同一个版本有好几个构建，装错了 GPU 就用不上**，
而 wheel 动辄 1~2 GB，下错了非常难受。

**这一步不一定会弹** —— 只有"真的需要装点什么"时才出现：

- 已经做过一次自检（`QSettings` 里的 `setup/completed` 标记）→ 直接进主界面；
- 本机环境本来就齐全（PyTorch 已装 **且** 默认权重 `yolo11n.pt` 已在本地）→ 直接跳过，
  且**不写标记**（以后哪天把依赖卸了，下次启动还会重新弹出来提醒）；
- 主界面左下角的「环境自检」按钮走强制路径，任何时候都能打开来核对或换版本。

页面是「① 检查电脑 → ② 确认方案 → ③ 下载安装」三步向导，全程只要点「下一步」，
硬件明细 / 文件清单 / 安装版本这些细节都收在「查看硬件详情」「查看文件清单」
「我要自己选安装版本」里，默认不挡路：

1. **① 检查电脑**：一句话结论（能不能用显卡加速）+ 可展开的硬件明细 ——
   操作系统 / Python / CPU 型号与核心数 / 内存，以及每一张显卡的
   **型号 · 显存 · 驱动版本 · 算力（`sm_xx`）**；底部「重新自检」可在换过显卡 /
   更新过驱动后重来。虚拟显示器与远程桌面注入的"假显卡"会被自动忽略。
   探测来源按可靠性排序：已装的 `torch` → `nvidia-smi` → 系统自带的设备列举
   （Windows CIM / Linux `lspci` / macOS `system_profiler`），任何一步失败都不影响其它步骤。
2. **② 确认方案**：给出推荐结论 + 理由（例："检测到 NVIDIA GeForce RTX 4070 Ti，算力 sm_89，
   驱动支持 CUDA 12.6，选择 CUDA 12.6"），并列出全部可选项；
   不可用的档位**仍然列出但标注原因**（如 "驱动支持的 CUDA 只到 12.6，装不了 CUDA 13.0"），
   想自己升级驱动再装的人能看到目标。
   同时给出每个文件的状态（已下载 / 可续传 / 待下载 / 大小不符）、总量、
   "需下载"与磁盘剩余空间对比，并可更换下载目录、展开文件清单。
3. **③ 下载安装**：当前文件 + 整体两个进度条、大小、速度、剩余时间与日志，
   以及"本次从断点续传了多少"；主按钮就是「开始下载并安装」。

**本地已经有的东西不会被重下一遍**（`utils/env_check.py` 只做秒回、不联网的判断）：
本机装过 PyTorch 时勾选项会写明已装版本并默认不勾（想换 CUDA 版本时自己勾上），
YOLO 权重已存在的文件"只做校验不重复下载"；方案区还会用一行
「本机环境：已安装 PyTorch x.y.z · 本地已有 YOLO 权重 …（已有的内容不会重复下载）」
把现状说清楚。PyTorch 与 YOLO 都跳过时，主按钮变成「确认已就绪（无需下载）」
—— 这是一次确认，不是失败。

**断点续传在界面上是能看见、能操作的**，不是藏在代码里的行为：

| 界面元素 | 含义 |
|---|---|
| 文件清单的「状态」列 | 每个 wheel 是 `已下载` / `可续传（已有 1.0 GB，42%）` / `待下载` / `大小不符，将重新下载`；下载中实时显示 `下载中 50% · 11 MB/s（续传自 1.0 GB）` |
| 汇总行 | `已下载 6.7 MB · 可续传 1.0 GB · 需下载 1.4 GB ｜ 磁盘剩余 118 GB`，空间不足会变红并建议清理或换目录 |
| 本机环境行 | `本机环境：已安装 PyTorch 2.14.0 · 本地已有 YOLO 权重 yolo11n.pt（已有的内容不会重复下载）` |
| 主按钮 | 检测到断点就自动变成「**继续下载**」（鼠标悬停提示能复用多少字节）；PyTorch 与 YOLO 都没什么可下时变成「确认已就绪（无需下载）」；否则是「开始下载并安装」 |
| 「重新下载（丢弃断点）」 | 只想从头下时用；有二次确认，且**只删断点、不动已经校验通过的成品** |
| 「更换目录」 | C 盘不够用时把安装包换个盘放，界面会重新解析下载地址 |
| 「暂停（保留断点）」 | 随时暂停，断点与已下好的 wheel 都留在缓存目录，下次接着下 |

### 怎么决定装哪一套

型号库覆盖市面主流型号，匹配出的**算力 + 驱动能力 + 操作系统**三者共同决定唯一的答案：

| 显卡 | 算力 | 结论 |
|---|---|---|
| RTX 50 系 / B100 · B200（Blackwell） | `sm_120` | 需要 CUDA ≥ 12.8 的构建（`cu128` 及以后） |
| RTX 40 系 / L40S（Ada） | `sm_89` | `cu118` ~ 最新都行，按驱动能力取最新 |
| RTX 30 系 / A100（Ampere） | `sm_86` / `sm_80` | 同上 |
| RTX 20 / GTX 16 / T4（Turing） | `sm_75` | 同上 |
| GTX 10 系（Pascal） | `sm_61` | `cu128` 起已移除该算力，落在 `cu126` / `cu118` |
| GTX 9 / 7 系（Maxwell / Kepler） | `sm_52` / `sm_37` | Kepler 已无官方构建，建议 CPU |
| AMD Radeon RX 5000~9000 / Vega / Instinct | — | Linux 用 ROCm，Windows 无官方构建 → 回退 CPU |
| Intel Arc A / B 系 | — | 可选 XPU（实验性），失败可退回 CPU |
| Apple M1~M4 | — | 装 CPU 版 wheel 即可，自带 Metal(MPS) 加速 |
| 无独显 / 核显 / 其它 | — | CPU（稳定，速度慢） |

驱动能力取自 `nvidia-smi` 报出的 `CUDA Version`（驱动支持的最高运行时）；
拿不到时退回比驱动号（`cu118` ≥ 452.39、`cu126` ≥ 560.76、`cu128` ≥ 570.65…）。

### 断点续传怎么做的

wheel 太大，断网重来代价太高，所以下载层专门做了三件事：

- 先写 `<文件名>.part`，**下完再原子改名** —— 中途失败不会留下"看起来能用"的残缺文件；
- 重新开始时读 `.part` 的大小并发 `Range: bytes=<已有>-` 续传；
  服务器不支持 `Range`（返回 200）或返回的续传起点与本地不符时，
  自动丢掉断点**整文件重下**，绝不把两段不同的字节拼在一起；
- 每个 wheel 都用官方索引里的 `sha256` 校验，不匹配就删掉重下；网络抖动自动重试。

已实测：对官方 CDN 连续中断两次（每次保留 `.part`）、第三次续传下完，
最终 `sha256` 与索引一致 —— 证明续传拼出来的文件与完整下载逐字节相同。

暂停 / 关窗口 / 断网 / 关机都不会白下：断点文件与 wheel 都留在缓存目录
（默认 `~/.video2personvideo/wheels`，界面上有「打开下载目录」），
下次点「继续下载」会从断点接着下。下载完可选自动 `pip` 安装
（先 `pip install --no-deps` 装本地 wheel，再按 `requirements.txt` 补齐其余依赖）；
运行在打包好的 exe 里时不会自动安装，只把 wheel 放好并给出 pip 命令。

> 版本默认跟随 `requirements.txt` 的锁定值（保证装完与仓库一致），读不到锁定值时才取索引最新版。
> 装完需要**重启程序**才会用上 GPU（Python 进程内的 torch 无法热替换）。

## 安装成 Windows 软件（Setup.exe）

前面几节都是"开发者视角"（venv + pip）。如果要把程序**当成普通 Windows 软件**发给别人，
用安装器：对方双击一个 exe，一路点"下一步"就能装好并直接用，**不需要预装 Python**。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -Dev    # 需要 pyinstaller
powershell -ExecutionPolicy Bypass -File scripts\build_installer.ps1   # 产出 dist\Video2PersonVideo-Setup.exe
```

发布时把 `dist\Video2PersonVideo-Setup.exe`（约 60~90 MB）单独发给对方即可，
PyTorch / YOLO / OpenCV 都在对方机器上现下。

### 装出来的是什么

安装器**自己不带** PyTorch / YOLO / OpenCV（否则 Setup.exe 就是近 1 GB），
它只带一份"安装逻辑 + 程序源码"；真正的重家伙在安装过程中按你的机器现下：

```
<安装目录>（默认 %LOCALAPPDATA%\Programs\Video2PersonVideo）
├── install.json          # 安装状态（阶段进度 / 断点 / 用户选择；也是"这是安装目录"的标记）
├── python/               # 内嵌的独立 Python 运行时（官方 embeddable 包）
│   └── Lib/site-packages # 程序本体、PyTorch、ultralytics、PySide6 都装在这里
├── app/                  # 程序本体（wheel / 源码）
├── configs/              # 默认配置（可直接改的 YAML）
├── assets/models/        # YOLO 权重（装完断网也能用）
├── ffmpeg/bin/           # 可选：随包 ffmpeg（保留音轨 / H.264 / 说话人跟随）
├── Video2PersonVideo.cmd # 双击开界面
├── v2pv.cmd              # 命令行入口
└── Uninstall.cmd         # 卸载入口
```

好处是**安装过程与用户的机器彻底解耦**：不碰系统 Python、不污染 conda，
版本冲突无从发生；卸载 = 删一个目录。

装完之后怎么用：

- 桌面 / 开始菜单的「Video2PersonVideo」双击即开图形界面（等价于安装目录里的
  `Video2PersonVideo.cmd`；没建快捷方式时直接双击这个 `.cmd` 也一样）；
- 命令行：安装目录里的 `v2pv.cmd`（开始菜单文件夹里是「Video2PersonVideo」开界面入口
  与「卸载 Video2PersonVideo」），`v2pv.cmd --check` 可核对环境；
- 参数与源码安装完全一致（同一个 `v2pv`）；卸载走开始菜单 / 设置 → 应用 / `Uninstall.cmd`。

### 向导里的四步

1. **欢迎**：说明一次会做什么 + 许可条款；如果这台机器上**已经装过**或**上次装到一半**，
   这里会出现状态卡片和「继续上次的安装 / 重新安装 / 卸载」三个按钮。
2. **安装位置**：默认装在当前用户目录（**不需要管理员权限，不弹 UAC**），
   实时显示目标磁盘剩余空间与预计占用；可勾选桌面 / 开始菜单快捷方式、
   装完立即启动，以及"装完删除安装包"（默认保留，方便修复安装）。
3. **组件**：先做一次硬件自检（CPU / 内存 / 每张显卡的型号 · 显存 · 驱动 · 算力），
   给出**推荐的一套 PyTorch** 并说明理由；下拉里能改（不可用的档位会标注原因，
   如"驱动支持的 CUDA 只到 12.6，装不了 CUDA 13.0"）；再选 YOLO 权重与随包 ffmpeg。
   底部实时显示"要下载多少 / 装完占多少"。
4. **安装**：阶段列表 + 当前文件进度 + 总进度 + 速度 / 剩余时间 / 续传字节 + 实时日志。

### 安装过程中退出：在哪一步、怎么退出

下载动辄 1~3 GB，中途关窗口是常态，所以**每一步都能安全退出**，而且界面会讲清楚后果：

| 阶段 | 会写盘吗 | 在这一步退出的后果 | 下次 |
|---|---|---|---|
| 准备安装目录 | 只写 `install.json` | 只留一个状态文件 | 继续 / 直接回滚删掉空目录 |
| 部署 Python 运行时 | 是 | `.part` 断点保留 | 从断点续传（已解压的部分不重来） |
| 引导 pip | 是 | 同上 | 重跑一次引导（几秒） |
| 下载安装 PyTorch | 是 | **断点完整保留** | 续传，已下好的 wheel 只校验不重下 |
| 安装依赖 / 程序本体 | 是 | pip 可能停在半成品状态 | 重跑该步（pip 幂等，已装好的包不重复下载） |
| 下载 YOLO 权重 | 是 | 断点保留（只有几 MB） | 续传 |
| 部署 ffmpeg（可选） | 是 | 失败 / 退出都不影响安装成功 | 续传或跳过 |
| 快捷方式 + 卸载登记 | 是 | 下次补做 | — |

点「取消安装…」或直接关窗口时，会弹出**三选一**：

- **继续安装**（默认）；
- **暂停并退出**：先等当前数据块写完、断点落盘，再退出；
  界面明确写出"已经下载 X GB，不会白下"；
- **回滚并退出**：删掉这次已写入的文件、快捷方式与注册表项；
  **已经下载的安装包默认保留**（在 `%LOCALAPPDATA%\Video2PersonVideo\cache`），
  下次安装可以直接复用，不用重下几个 GB。

再次运行安装器时，它会读 `install.json`：

- 上次没装完 → 欢迎页给出「继续上次的安装」，从记录的那一步接着走；
- 已经装好了 → 给出「重新安装」（重置阶段，但复用下载缓存）与「卸载」；
- 安装目录被手工删了 → 提示"清理残留后重新安装"。

命令行的等价能力（排错 / 批量部署用）：

```powershell
Video2PersonVideo-Setup.exe --list-backends                  # 这台机器能装哪几套 PyTorch
Video2PersonVideo-Setup.exe --silent --backend cu126         # 静默安装（进度打到日志）
Video2PersonVideo-Setup.exe --uninstall --quiet              # 静默卸载
Video2PersonVideo-Setup.exe --repair                         # 重新安装（保留安装包）
```

### 卸载

三种入口都指向同一个卸载器，**不依赖当初那个 Setup.exe 还在不在**：

- 开始菜单 → `Video2PersonVideo` 文件夹 → 「卸载 Video2PersonVideo」；
- 设置 → 应用 → 已安装的应用 → 「Video2PersonVideo」→ 卸载
  （安装时写入了 `HKCU\...\Uninstall\Video2PersonVideo`）；
- 安装目录里的 `Uninstall.cmd`。

卸载会删除安装目录、快捷方式与注册表项；是否同时删掉已下载的安装包由界面上的勾选项决定
（默认删，因为用户是主动卸载）。若安装目录里的解释器正在运行（卸载器自己就跑在里面），
Windows 不允许删正在使用的 exe，卸载器会**安排一个后台脚本在退出后清理**并在界面上说明，
而不是留一个删了一半的目录。

### 打包说明

| 开关 | 作用 |
|---|---|
| `scripts\build_installer.ps1` | 默认产出**单文件** `dist\Video2PersonVideo-Setup.exe`（约 60~90 MB） |
| `-Dir` | 产出目录版（启动快、便于排错） |
| `-Console` | 保留控制台窗口（调试 `--silent` 输出时用） |
| `-SkipWheel` | 跳过 wheel 构建，复用 `build\installer_payload` |
| `-VenvDir <dir>` | 用哪个虚拟环境打包（不存在会直接报错并提示先跑 `setup_env.ps1 -Dev`） |

构建脚本先 `pip wheel` 把项目打成 wheel 塞进安装器的 payload（安装时直接装它，
不需要目标机器有 setuptools），再交给 `build\Video2PersonVideo-Setup.spec` 打单文件 exe。
spec 里显式排除了 torch / ultralytics / opencv / numpy，所以安装器体积极小。

> 安装器与主程序是两个独立程序：主程序仍然是"可直接跑 / 可 PyInstaller 打包"的，
> 安装器只是给最终用户多提供一条"像装软件一样装"的路。

## 打包成 Windows exe

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -Dev   # 需要 pyinstaller
powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1        # 目录版（推荐，启动快）
powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1 -OneFile   # 单文件版
```

两个脚本都接受 `-VenvDir <dir>` 指定用于打包的虚拟环境（默认 `.venv`）；没装好环境时
脚本会直接报错并提示先跑 `setup_env.ps1 -Dev`。用 PyInstaller 手工打包则用
`build\Video2PersonVideo.spec`（`V2PV_ONEFILE=1` 切单文件）。

产物：

| 模式 | 产物 | 体积 | 打包耗时 | 备注 |
| --- | --- | --- | --- | --- |
| 目录版（默认） | `dist/Video2PersonVideo/` | 约 860 MB | 约 3~7 分钟 | 启动快，整个文件夹拷给对方即可 |
| 单文件版 | `dist/Video2PersonVideo.exe` | 约 316 MB | 约 4 分钟 | 首次启动要解压到临时目录，慢十几秒 |

两种模式都已在 Windows 11 + CPU 环境实测：`--version` / `--check` / 真实视频裁剪 /
`--gui` 图形界面均正常。

> spec 里额外做了一件事：torchvision ≥0.21 的扩展改名为 `_C_stable.pyd`，
> PyInstaller 自带 hook 只找 `_C.pyd`，漏掉会导致运行时
> `operator torchvision::nms does not exist`（NMS 用不了）。spec 已显式收集这两个扩展。

如需无控制台窗口的纯界面版，把 `build/Video2PersonVideo.spec` 里的 `console` 改成 `False`。

## 常见问题

**Q：输出视频没有声音？**
OpenCV 写出的视频本身不带音轨。程序默认调用 ffmpeg 合并原音轨；请确认 `ffmpeg -version` 可用
（`v2pv --check` 会提示），否则加 `--no-audio` 明确输出无声视频。

**Q：画面一会儿很慢、随后忽然变快，但声音是正常的？**
源视频是**可变帧率（VFR）**——手机录像、录屏、剪辑导出里很常见：每帧的真实显示时长
并不相等。而 OpenCV 读帧不带时间戳、写帧只能用单一 fps，逐帧原样写回会把画面「均匀
摊平」：帧率高的片段被拉长（慢放）、帧率低的片段被压缩（快进），声音却仍按真实时间轴走，
于是听感上就是「开头很慢、后面突然变快」。

程序默认先探测源视频信息（标称帧率 / 平均帧率 / 帧数 / 时长），一旦判定为可变帧率，
就先用 ffmpeg 把画面归一化成恒定帧率再逐帧处理，输出画面的节奏即与原片一致。
这一步需要 `ffmpeg`（`v2pv --check` 会提示），缺失时日志会告警、行为退回旧版；
想手动关掉可用 `--no-normalize-vfr`（配置项 `normalize_vfr: false`）。

**Q：为什么输出文件的尺寸和原视频不一样？**
这是设计目标：输出尺寸由你选的比例决定（如 `9:16 → 1080×1920`），
算法只调整「从原片里切哪一块」。想换清晰度用 `--target-size` 指定目标像素。

**Q：输出视频会不会比原视频大很多？**
会，而且很容易。两个原因叠加：(1) OpenCV 只能写 `mp4v`（MPEG-4 Part 2），
同等画质下它需要的码率是 H.264 的好几倍，而原视频通常已经是 H.264/H.265 且码率很低；
(2) 输出被规整到目标像素（默认短边 1080），低分辨率源视频被放大后像素数暴涨，
而 `mp4v` 没有任何码率控制。常见结果是成品的 3~10 倍。

默认开启的**体积守护**用来兜住这件事：处理完成后先比对体积，
一旦 `输出 > 原视频 × max_size_ratio`（默认 1.0），就用 ffmpeg 按
「体积上限 ÷ 时长」反推码率压成 H.264，超发就自动降码率重试，直到达标；
压不到（已到最低码率）会明确告警并在汇总里标注，不会假装成功。
结果汇总里会如实打印「输出体积（原视频 X MB，N.NN×）」。

想进一步控制体积：

- `--max-size-ratio 0.5`：允许输出只有原视频的一半大小
- `--target-size 720x1280`：降一档分辨率，比任何码率调整都有效
- `--min-bitrate 150`：放宽码率下限，让它敢压得更狠
- `--no-size-guard`：彻底关掉自动压缩（自己用 `--reencode --crf` 接管）
- 注意体积守护需要 `ffmpeg`；没有它时只会在日志里告警，输出仍是 `mp4v` 版本

**Q：人物被裁到只剩半张脸 / 头顶被切？**
调大 `--headroom`（如 `0.15`），或在界面第 2 步用「预览构图效果」调参数后再全量处理。

**Q：为什么 YOLO 不能直接识别"谁在说话"？**
YOLO 是纯图像检测 / 姿态模型：① 没有音频输入；② COCO-17 关键点里没有嘴部关键点
（只有鼻 / 眼 / 耳 / 肩 / 髋…）；③ Ultralytics 也没有官方的人脸 / 主动说话人模型。
所以本工具自己补了音频这一半：ffmpeg 抽音轨 → 逐帧语音能量包络 → 与每个人的
**嘴部运动**做滑窗相关，谁的声音与嘴型最同步就跟谁（音频-视觉主动说话人检测）。
嘴部区域优先用姿态关键点定位（模型换成 `*-pose.pt` 时最准），没有关键点就用
OpenCV 自带的人脸检测兜底，再退到按人框比例估算；三级都拿不到就放弃该候选，
不参与判定（宁可不动，也不瞎猜）。

**Q：多人对话的镜头没跟着说话的人走？**
按顺序排查：

1. `v2pv --check` 看有没有 ffmpeg —— 抽音轨依赖它，缺失时会**安静跳过**（结果汇总里会写明）
2. 结果汇总里的「说话人跟随」一行会给出原因（未启用 / 无音轨 / 未检测到语音 / 已启用）
3. 人物在画面里太小（远景全身）：嘴动幅度低于阈值时会放弃判定，此时可调低
   `speaker_min_mouth_motion`，或把模型换成 `*-pose.pt` 让嘴部定位更准
4. 换人太频繁 → 调大 `--speaker-switch-hold` / `--speaker-switch-margin`；
   该跟谁没跟上（总跟着更"抢镜"的那个人）→ 确认 `--speaker-weight` ≥ `1.0`
   （默认 `1.8`：判定出说话人后他必定胜出；小于 `1.0` 只是"软优先"）
5. 单声道混音下"谁在说话"只能靠视觉嘴型区分：两人都背对镜头或被遮挡时无法判定，
   程序会退回原来的主角打分规则（此时可用 `--camera-follow lock` 让镜头更稳）

**Q：画面里好几个人，为什么只显示了其中几个？**
多人分屏默认开启，窗口布局按「输出比例 + 人数」查表（`configs/multi_person_layout.yaml`），
同时显示的人数有三个上限在起作用：

1. `--multi-person-max`（默认 4）：超出的按"正在说话的人 > 主角打分"取舍；
2. 格子太小：某档布局里最窄的格子短边不足 96 像素时，程序会主动少显示几个人
   （与其切成看不清的小方块，不如把留下的几个人显示清楚）；提高 `--target-size` 可以缓解；
3. `--min-person-ratio`（默认 0.73）：低于该高度占比的人物算"背景里的路人"，不算主要人物、
   不进窗口。想让画面里占比很小的人也参与，把它调小（如 `0.15`）。

结果汇总里的「多人分屏」「分屏布局」两行会说明本片用了哪些布局、各多少帧。
想改排版就编辑布局表；想让画面里永远只有一个人，用 `--no-multi-person`
（或界面第 2 步取消勾选「多人分屏」）。

**Q：人物一走动，镜头就跟着摇，看着头晕？**
这是"镜头跟随"的副作用：整个背景在视野里持续流动，视觉输入与身体平衡感冲突，
看久了就会不适（晕动症）。程序默认已经做了三层保护：① 人物在画面内走动时**镜头完全不动**
（自由活动区）；② 平移有速度与加速度上限，不会甩镜、不会急起急停；
③ 缩放（推拉）比平移更慢更少。如果还是晕，按下面顺序加码：

1. `--camera-follow lock`（界面第 2 步「镜头跟随 · 锁定」）：最稳的一档
2. 仍不够就逐项调：`--deadzone 0.30`（人物可在画面内自由走动的范围更大）、
   `--pan-speed 0.01`（镜头移动更慢）、`--smoothing-alpha 0.12`
3. 取舍要清楚：**死区越大、速度越慢，人物偏离画面中心就越多** ——
   镜头只把人"推回边界"，不会拽回正中。本工具认为"看着不晕"比"人物绝对居中"更重要；
   反过来，想让人物更居中就用 `--camera-follow active`

**Q：镜头抖动厉害（画面在抖，但人物没怎么动）？**
调低 `--smoothing-alpha`（如 `0.12`）；如果是对讲/直播素材，也可以把
`--hold-frames` 调小，让无人时更快切到**无人物显示**（默认「全画面适配」，
镜头此时**完全不移动**，是最不容易看晕的一档）。

**Q：画面里的人走开之后，画面变小了 / 上下有模糊的边？**
这是默认的「全画面适配」：没有主要人物时把**整幅画面完整放进**输出画布，一帧看全、
镜头不动；横屏素材转竖屏时上下必然空出一块，用同一帧的模糊放大版填充（不是黑边）。
不想要这种留白可以换档：

- `--no-person-mode tiles`：把纵向空间用起来 —— 上格放整幅画面，下格给画面里的
  远景小人物开半身特写（`--no-person-tiles-max` 控制开几个）；
- `--no-person-mode scan`：回到旧的空镜巡视（取景框缓慢扫过整幅画面）；
- `--no-person-mode center`：不解释，直接取画面正中（最保守）；
- `--no-blur-background`：留白改成纯色底色；
- `--no-person-seconds 2`：过渡更慢更舒缓（0 = 直接切过去）；
- 反过来，如果这类空镜很少、你更在意"别切少了"，把 `--hold-frames` 调大
  （如 `90`）能让取景框多撑一会儿再切。

**Q：提示 `无法创建输出视频`？**
换 `.mp4` 后缀（编码器 `mp4v` 兼容性最好），或换成带编码器的 `opencv-python` 完整版
（不要用 `opencv-python-headless` 的裁剪构建）。

**Q：`--check` 显示 `CUDA 可用: 否`，但我是 N 卡？**
说明装的是 CPU 版 torch。打开界面点左下角「环境自检」，它会自动判断该装哪一套并下载安装
（也可以自己按第 1 步的 CUDA 命令重装）；先 `nvidia-smi` 确认驱动正常。

**Q：第一次打开程序弹出的「环境自检」是什么？下错了东西怎么办？**
它是显卡自检 + 依赖下载，见「首次运行 · 显卡自检与依赖下载」一节。几点说明：

- 只是想知道自己该装什么 → 点「稍后再说」就能进主界面，什么都不会下载；
- 想换一套构建 → 上方的「安装」下拉里自己选（不可用的档位会标注原因），
  选了之后会重新解析地址，再点「开始下载」即可；
- 下到一半关了 / 断网了 → 断点都留着，下次点「继续下载」接着下，不会重复下载；
- 本机已经有 PyTorch / 权重 → 已有的文件"只做校验不重复下载"，页面会用一行
  「本机环境：…」说明已经有什么，PyTorch 的勾选项也会写明已装版本并默认不勾；
- 已经装好了 GPU 版 torch → 这一步可以整个跳过（`--check` 里会显示 `CUDA 可用: 是`）；
  若 PyTorch 与默认权重都齐全，连自检页都不会弹出来（想核对就点主界面的「环境自检」）。

**Q：处理速度慢？**
优先上 GPU（`--device cuda:0`），其次换 `yolo11n.pt`、降低 `--imgsz`、加大
`--detect-interval`（抽帧检测），最后关掉 `--reencode`。

读帧与写盘已经默认与推理 / 裁剪重叠（后台线程），不需要额外设置；GPU 上再叠一个
`--infer-batch 4~8` 收益明显（每帧的调用开销被摊薄）。另外两个容易白付的代价：
整片始终只有一个人时可以用 `--no-speaker-tracking` 关掉说话人分析（单人画面本来
就不会产生说话人偏置，结果一致）；输出体积超标时体积守护会整片重编码一遍
（`--no-size-guard` 可关，代价是成品可能比原片大几倍）。

## 开发

```powershell
pip install -r requirements-dev.txt
pip install -e .
pytest                          # 快速用例（不需要模型）
pytest tests/test_invariants.py # 不变量验收用例
$env:RUN_SLOW=1; pytest -m slow # 端到端用例（会下载模型）
ruff check src tests
```

测试分层：

- `tests/test_ratio.py` / `test_framing.py` / `test_smoothing.py` / `test_crop.py` / `test_pose.py`：纯函数单元测试
- `tests/test_audio.py` / `test_mouth.py` / `test_speaker.py`：说话人跟随的三块（音频 / 视觉 / 判定）
- `tests/test_speaker_pipeline.py`：多人物时主角真的跟着说话人走 + 各级降级路径
- `tests/test_layout.py`：多人布局表（网格解析 / 权重划分 / 查表 / 配置加载，纯函数）
- `tests/test_multi_person.py`：多人分屏流水线（每个窗口是谁、开关与回退、人数抖动、
  格子过小时自动少显示、抽帧插值、说话人占 1 号窗口、远景小人只切上半身）
- `tests/test_no_person.py`：无人物显示（适配矩形 / 过渡几何不拉伸不越界 / 全景 + 特写
  网格、完整放下与模糊底合成、不透明度、兜底时机与渐进过渡、自动退回全画面适配、
  `scan` 与 `center` 两档的旧行为不变、配置规整与校验、端到端汇总）
- `tests/test_pipeline.py`：抽帧插值、批量推理与逐帧结果的逐帧等价性
- `tests/test_invariants.py`：**不变量验收门槛** + 端到端场景（合成视频 + 假检测器，无需模型）
- `tests/test_batch.py` / `test_output.py`：批量遍历、失败隔离、音轨、元数据、临时文件清理
- `tests/test_gpu_probe.py`：型号库覆盖（主流 NVIDIA / AMD / Intel / Apple 型号、
  虚拟显卡过滤）+ 算力 / 驱动 → 构建推荐的组合校验
- `tests/test_torch_install.py`：索引页解析、wheel 筛选、torch↔torchvision 配对、
  版本锁定、下载计划生成（索引用内联 HTML 模拟，不联网）
- `tests/test_downloader.py`：断点续传（本地 HTTP 服务器模拟 `Range` 行为）、
  服务器不支持续传 / 续传位置不符的降级、`sha256` 校验、重试、取消保留断点
- `tests/test_gui_smoke.py`：向导构建、参数收集、比例对话框校验（offscreen，无需显示器）
- `tests/test_env_check.py`：本机环境速查（权重查找 / 就绪判断 / 一行摘要，不联网）
- `tests/test_installer.py`：安装器（阶段与权重、状态文件读写与损坏兜底、续装 / 重置判定、
  完整安装链路、PyTorch 下载中途取消后继续、可选组件失败降级、回滚保留 / 删除缓存；
  用假下载器与假 pip，不联网、**不碰真实桌面快捷方式与注册表**）
- `tests/test_setup_page.py`：自检页渲染（三步向导）、方案下拉的可用性标注、下载进度与状态流转
- `tests/test_setup_worker.py`：自检后台线程（缓存状态上报、续传 / 丢弃断点、跳过已下载与 torch、
  未选任何项不算失败）
- `tests/test_config.py` / `test_cli.py`：配置默认值 / YAML 加载 / 命令行覆盖与参数解析
- `tests/test_detector.py` / `test_video_io.py` / `test_processor.py`：检测器结构化输出（桩对象模拟
  ultralytics `Results`）、视频读写、流水线统计与轻量端到端
- `tests/test_ffmpeg_tools.py`：缺少 ffmpeg 时的优雅降级 + 音画时长对齐
- `tests/test_size_guard.py`：体积守护（超出体积上限时自动压缩到限制内）

## 依赖版本说明

版本已在 2026-09 对齐 PyPI 最新稳定版；`requirements.txt` 里为精确锁定，便于复现。

| 库 | 版本 | 作用 |
| --- | --- | --- |
| ultralytics | 8.4.163 | YOLO 模型加载、推理、NMS、画框 |
| opencv-python | 5.0.0.93 | 视频读写、切片缩放、截图（5.x 遇兼容问题可回退 4.14.0.94） |
| torch / torchvision | 2.14.0 / 0.29.0 | 深度学习框架，按 CPU/CUDA 选择安装源 |
| numpy | 2.5.3 | 数组运算 |
| tqdm | 4.70.1 | 处理进度条 |
| pyyaml | 6.0.3 | 读取 YAML 配置 |
| ffmpeg-python | 0.2.0 | 音轨探测（合并/转码/元数据走 ffmpeg 命令行兜底，更稳） |
| PySide6-Essentials | 6.11.2 | GUI（LGPL，打包分发省心；可换 PyQt6 6.11.0） |
| pyinstaller / auto-py-to-exe | 6.22.3 / 2.50.1 | 打包 Windows 独立 exe |

## 变更记录

### 未发布 · 处理提速（结果逐像素不变）

- **批处理只加载一次模型**：`run_batch` 原先对每个视频都重新构造检测器（权重加载 +
  推理器初始化，GPU 上还要建 CUDA 上下文 / 选卷积算法，几秒一个视频），
  现在整批复用同一个实例（`processor.build_detector`）；加载失败仍退回
  "逐个文件重试、各自报错"的旧行为
- **ffprobe 结果缓存**（`utils/ffmpeg_tools.py:probe_media`）：按「路径 + 修改时间 + 大小」
  缓存探测结果 —— 处理一个视频原来会把同一个源文件探测 5 次左右、中间产物各 2 次，
  而每次都是一个独立子进程（实测 0.35s → 0.4ms）；文件被改写后自动失效，另提供
  `clear_probe_cache()`
- **嘴动分析改为按 ROI 惰性灰度**（`core/mouth.py:FrameGray`）：原先每个关键帧都要把整帧
  转灰度（1080p 约 13ms/帧，比 GPU 上的 YOLO 前向还贵），而真正要看的只有"嘴"与
  "上半脸"两块小矩形；现在只对用到的矩形做灰度（0.1ms 量级），人脸定位也只灰度搜索区。
  灰度是**逐像素**运算，所以数值与整帧换算**逐位相同**（已验证：各矩形、越界矩形、
  帧间差分都与旧口径一致）
- **读帧预读 + 写帧异步**（`core/video_io.py`）：解码与 mp4v 编码放进后台线程，与主线程的
  推理 / 裁剪重叠；帧序与内容与同步读写**逐字节一致**（端到端比对过
  默认 / tiles / scan / center / 不裁剪 / 抽帧批量六种场景的输出 sha256 与逐帧像素）；
  队列有背压、不会无限吃内存，写线程出错照常往上抛；设 `V2PV_SYNC_IO=1` 可一键
  退回单线程读写（排错用）

### 未发布 · 无人物显示（默认「全画面适配」）

- 新增「画面里没有主要人物时怎么显示」的四档策略（`no_person_mode`，默认 `fit`），
  替代原先"只能空镜巡视 / 回中"的单一兜底：
  - `fit` **全画面适配**：整幅画面等比缩小完整放进输出画布，留白用**同一帧的模糊放大版**
    填满（`blurred_cover`，降采样后模糊，代价可忽略）；一帧看全、镜头完全不移动 ——
    横屏素材转竖屏时不再"只能看到局部"，也不再有持续横移带来的晕动感；
  - `tiles` **全景 + 特写**：整幅画面收进上方通栏（`layout.tiles_layout`：首行高度 =
    通栏宽 ÷ 源比例，画面正好填满那一格），下方纵向排列若干**次要小人物**的**上半身特写**
    （复用 `compute_upper_body_box` 与 `build_anchors`，远景小人被放大成半身）；
    特写窗口在过渡后半段淡入，画面"收进上格、下面顺势露出"；
    没有够大的小人物、或竖屏里排不下（`MIN_CELL_PX`）时**自动退回全画面适配**；
  - `scan`（空镜巡视）/ `center`（画面居中）保留原行为，一行配置即可换回
- 过渡几何统一由 `layout.transition_frame` 给出：源侧"人物取景框 → 整幅画面"、画布侧
  "整块画布 → 目标矩形"，**两端比例一起插值**（宽与中心线性插值、比例线性插值、高由比例
  反推），因此中间每一帧都不拉伸、不越界、不出现空洞；进度按秒标定（`no_person_seconds`，
  默认 0.8 秒）并做缓入缓出，与实际帧率无关
- `crop.compose_multi_frame` 扩展：窗口支持 `fit=True`（**完整放下**而不是裁满格子）
  与 `alpha`（淡入），画布底色支持 `blur=True`（同一帧的模糊放大版）；
  `fit=False` + 纯色底时与旧行为**逐像素一致**（有测试兜住）
- `BoxSmoother` 新增 `no_person_mode` / `no_person_seconds`：超时无人帧返回"正在取出的
  那一块画面"（`MODE_FIT` / `MODE_TILES`）并暴露过渡几何；人物回来时进度归零，
  从"整幅画面"平滑缩回人物身上
- 新增 `core/noperson.py`：`NoPersonDisplay`（模糊底 / 特写数量 / 小人物下限）与
  `TilesPlanner`（关键帧挑小人物、按帧尺寸缓存格子、按过渡进度给窗口）
- 新增参数 `no_person_mode` / `no_person_seconds` / `no_person_blur` / `no_person_tiles_max` /
  `no_person_secondary_ratio`；CLI `--no-person-mode` / `--no-person-seconds` /
  `--[no-]blur-background` / `--no-person-tiles-max` / `--no-person-secondary-ratio`；
  GUI 第 2 步新增「没有人物时」下拉，示意图右侧会按档位画出"无人时输出长什么样"，
  「预览构图效果」与正片共用同一套排布逻辑
- 结果汇总新增「无人物显示」一行（说清当前档位与"为什么没看到效果"），
  「兜底帧数」补上全画面适配 / 全景+特写两类帧数
- 新增测试 `tests/test_no_person.py`

### 未发布 · 空镜巡视（无人物画面不再丢掉两侧内容）

- 新增兜底第二级「空镜巡视」（`core/smoothing.py` 的 `MODE_SCAN`）：保持超时后画面里
  仍然没有主要人物时，只要**源画面比目标取景框更大**（典型场景：横屏素材转竖屏），
  取景框就沿那条多出来的轴按正弦缓动缓慢往返，几个来回把居中裁剪会丢掉的左右 / 上下
  内容全部带到成片里；没有可巡视的空间（源比例与目标一致）或巡视被关闭时才回中
- 巡视从"当前取景框所在的位置"起步（不会先倒着跑到一端），两端速度自然减到 0，
  单程时长按秒标定并用真实帧率换算：24 / 30 / 60fps 素材的巡视速度与时长一致
- 新增参数 `smoothing_scan_seconds` / CLI `--scan-seconds`（默认 `12.0` 秒，`0` = 关闭）；
  结果汇总新增「空镜巡视 N 帧」，与"保持上一帧 / 回中"并列展示

### 未发布 · 可视化安装包（Setup.exe）

- 新增安装器 `video2personvideo.installer`：把程序当普通 Windows 软件安装，
  **自带独立 Python 运行时**、按显卡现下 PyTorch、下好 YOLO 权重，
  装完直接能用（不需要预装 Python、不需要管理员权限）
  - `installer/gui/*`：五步向导（欢迎 / 位置 / 组件 / 进度 / 完成），
    硬件自检给出推荐构建与理由，实时显示体积、速度、剩余时间与断点续传量
  - `installer/engine.py`：安装引擎，阶段化执行
    （准备目录 → 内嵌运行时 → pip → PyTorch → 依赖 → 程序本体 → YOLO → ffmpeg → 收尾），
    每步结束都落盘，下载与 pip 都可中止
  - `installer/journal.py`：安装状态文件 `install.json`（原子写 + 用户目录镜像），
    据此判定"未装完 / 已装好 / 目录被删"，并支持从任意阶段重置或继续
  - `installer/stages.py`：阶段模型，含权重（进度条按真实耗时分配）与
    **每一步的中断说明**（"在这一步退出会发生什么、下次怎么继续"）
  - `installer/runtime.py`：下载官方 embeddable Python 包 → 解压 → 改写 `._pth`
    （把 `Lib/site-packages` 与 `import site` 补上，否则 pip 装进去的包 import 不到）
    → 用 `get-pip.py` 引导 pip
  - `installer/windows.py`：快捷方式（PowerShell + WScript.Shell，无 pywin32 依赖）、
    卸载注册表项（出现在"应用和功能"）、下载缓存目录、启动入口、
    **"自己删自己"的延迟清理脚本**（卸载器跑在安装目录里时用）
  - `installer/payload.py`：随安装器的程序本体（wheel 或源码）、默认配置与依赖清单；
    依赖清单**剔除 torch / torchvision**，避免 pip 把按显卡装的 CUDA 版覆盖成 CPU 版
  - `installer/cli.py`：`--silent` 静默安装、`--uninstall` 卸载、`--repair` 重新安装、
    `--list-backends` 列出可用构建
- **中断语义**：取消 / 关窗口时弹三选一 —— 继续安装 / 暂停并退出（保留断点）/
  回滚并退出（删已写入文件，但**默认保留已下载的安装包**）；
  每个阶段都给出了对应的中断后果说明（见 README 表格）
- 新增 `utils/app_paths.py`：安装布局与资源定位（安装根识别、模型目录、
  随包 ffmpeg 目录、`register_bundled_tools()` 把随包目录挂到 PATH / DLL 搜索路径）
  - `torch_install.model_directory()` / `wheel_directory()` / `default_requirements_path()`
    改为走它，于是"源码跑 / PyInstaller 打包 / 安装器装出来的环境"三种形态都能找到资源
  - `ffmpeg_tools.find_ffmpeg()` / `find_ffprobe()` 支持 `V2PV_FFMPEG` 与随包目录，
    装出来的环境不需要用户自己装 ffmpeg
- `utils/torch_install.py`：`build_plan()` 新增 `python_version` / `tags`
  （把"给谁下 wheel"与"谁来下"分开，安装器要把 wheel 装进版本可能不同的内嵌解释器）；
  新增 `run_process_streaming()`（逐行回显、可中止），`run_pip_streaming()` 改为它的薄封装
- 新增打包配置 `build/Video2PersonVideo-Setup.spec`（安装器单文件 exe，
  显式排除 torch / ultralytics / opencv / numpy）与一键脚本 `scripts/build_installer.ps1`
- 新增 `v2pv-setup` 入口；新增测试 `tests/test_installer.py`
  （阶段模型 / 状态日志与续装判定 / 完整安装链路 / 取消与续装 / 可选组件降级 / 回滚，
  用假下载器与假 pip，不联网、不动真实桌面快捷方式与注册表）

### 未发布 · 跳过不必要的下载（本机环境速查）

- 新增 `utils/env_check.py`：本机环境速查（PyTorch 装没装 / 版本号、YOLO 权重在不在），
  只做 `find_spec` / 读 `dist-info` / `Path.exists` 这类**秒回且不联网**的判断，
  不 `import torch`（那要好几秒），因此能放心地在启动路径上调用；
  权重在仓库 `assets/models` 与当前工作目录两处查找 —— 后者是 ultralytics 自己
  找不到权重时的落地位置，也是用户"直接跑通"多半靠的那份
- 首次运行自检页新增两条跳过路径（`gui/setup_dialog.py:maybe_show_setup`）：
  已经自检过（`QSettings` 里的 `setup/completed`）直接进主界面；
  本机 PyTorch + 默认权重都齐了也直接跳过，且**不写标记**，
  依赖被卸掉后下次启动会重新提醒；主界面「环境自检」按钮走强制路径
- 自检页把"本机已经有什么"说清楚：`environment_summary()` 在方案区输出一行
  「本机环境：…（已有的内容不会重复下载）」，勾选项也会写明本机已装的 PyTorch 版本
- 下载执行细化（`gui/controller.py:SetupWorker`）：
  - 本机已装 PyTorch → 勾选项默认不勾，可只补下 YOLO 权重
    （`SetupRequest.skip_torch`，此时不碰 wheel、也不安装）
  - 已下好且校验通过的文件"只做校验不重复下载"；PyTorch 与 YOLO 都跳过时
    主按钮变成「确认已就绪（无需下载）」，算成功而不是失败
  - 缓存里的断点字节数、磁盘剩余空间在方案区一次性报出，空间不足会提示换目录
- 新增测试 `tests/test_env_check.py`（权重查找 / 就绪判断 / 一行摘要）与
  `tests/test_setup_worker.py`（缓存状态上报、续传 / 丢弃断点、跳过已下载、
  失败不中止、跳过 torch、未选任何项不算失败）

### 未发布 · 首次运行显卡自检与依赖下载（断点续传）

- 新增首次运行页：第一次打开界面先做一次硬件自检，据此推荐该装哪一套 PyTorch / YOLO；
  用户点「稍后再说」也能继续用，主界面左下角「环境自检」可随时重做
  （`gui/setup_dialog.py` + `gui/pages/setup_page.py` + `gui/controller.py:SetupWorker`）
- 新增 `utils/gpu_probe.py`：CPU / 显卡型号自检
  - 型号库覆盖主流型号（NVIDIA GeForce 7~50 系 / Quadro / Tesla / Hopper / Blackwell、
    AMD Radeon RX 400~9000 / Vega / Instinct、Intel Arc A·B / Iris Xe、Apple M1~M4、Adreno），
    匹配出厂商 / 家族 / 架构 / 算力
  - 多来源探测（已装 torch → `nvidia-smi` → Windows CIM / Linux `lspci` / macOS `system_profiler`），
    逐级降级、绝不抛异常；虚拟显示器与远程桌面注入的"假显卡"自动忽略
- 新增 `utils/torch_backends.py`：PyTorch 构建选择表（`cu118`/`cu121`/`cu124`/`cu126`/`cu128`/
  `cu129`/`cu130`/`rocm6.2`/`rocm6.4`/`xpu`/`cpu`），按**算力区间 + 驱动能力 + 操作系统**
  三重过滤后给出唯一推荐；不可用的档位仍列出并标注原因（纯函数，易测）
- 新增 `utils/downloader.py`：**断点续传**下载器（只用标准库）
  - `.part` 临时文件 + 完成才原子改名；`Range` 续传；服务器不支持 `Range` 或
    续传起点不符时自动整文件重下；`sha256` 校验、失败重试、可随时取消（保留断点）
- 新增 `utils/torch_install.py`：解析 PyTorch 官方 simple index（含 `sha256`）→
  按 Python 标签 / 平台标签 / 版本锁定挑 wheel → 生成下载计划 → `pip install --no-deps`
  装本地 wheel 再补齐其余依赖；打包环境只下载不安装并给出 pip 命令
- 版本优先跟随 `requirements.txt` 的锁定值，读不到时才取索引最新版
  （避免"下了 CUDA 版又被 `pip` 换回 CPU 版"这类事故）
- 新增测试 `tests/test_gpu_probe.py` / `test_torch_install.py` /
  `test_downloader.py` / `test_setup_page.py`

### 未发布 · 多人分屏（多人各占一个上半身小窗口）

- 新增多人分屏：画面里出现两个及以上主要人物时，每人一个上半身小窗口同时显示
  （此前整段视频只跟一个人），默认开启，可在界面第 2 步取消勾选或用 `--no-multi-person` 关闭
- 新增布局配置 `configs/multi_person_layout.yaml`（可选，缺失时用内置默认）：
  - 用**字符网格**描述排版（`["11", "23"]` = 上通栏大窗 + 下排两个），同号连续占格即合并单元格
  - 按「比例族 / 具体比例 + 人数」查表；查不到（人数超出、自定义比例）时按宽高比自动生成网格
  - 支持给行列加权（`{grid: [...], cols: [1.4, 1.0]}`）、缝隙与底色、人数切换滞回帧数
- 新增 `core/layout.py`（网格解析 / 查表 / 像素划分，纯函数）与 `core/multi.py`
  （选人排座 + 逐人取景 + 轨迹关联 + 逐人平滑）
  - **取谁**：过滤过低小人物后按主角打分取前 `multi_person_max` 个（默认 4）
  - **排哪**：`1` 号窗口给正在说话的人（没开启说话人跟随时给主角），其余按画面位置入座
  - **怎么稳**：每人一个"座位"（跨帧轨迹 + 独立平滑器），旁边的人动了不会影响他；
    人数抖动要连续多帧才换布局
  - **怎么不出丑**：每格按**该格自己的宽高比**取景（窄窗里的半身不被拉伸变形）；
    格子短边不足 96 像素时主动少显示几个人
- `core/crop.py` 新增 `compose_multi_frame`：把小窗口贴到底色画布上，输出尺寸仍恒定不变
- 新增参数 `multi_person`（默认开）/ `multi_person_max` / `multi_person_order` /
  `multi_person_layout_file`；CLI `--[no-]multi-person` / `--multi-person-max` /
  `--multi-person-order` / `--layout-config`；GUI 第 2 步与「预览构图效果」新增勾选与实时重算
- 单人画面与关闭该功能时输出**逐帧完全一致**（有专门测试兜住）；
  结果汇总新增「多人分屏」「分屏布局」两行
- 每个小窗口改为**只切该人物的上半身**（`core/framing.py:compute_upper_body_box`）：
  - 上半身范围优先取姿态关键点的"头顶 / 腰"，没有关键点时按"头到腰 ≈ 1.6 个框宽"估算
    （远景站姿会被收成半身，近景 / 坐姿这类矮胖框保持整条框）
  - 框高取"上半身高度 + 留白"与"人物宽度 + 留白"两个候选的较大者，
    因此人物左右不再剩下大片非人物场景；人物在画面里很小也只切上半身，不再连着腿一起框
  - 仍然只平移不缩放：宽高比恒等于格子比例、不越界、保留最小 / 最大尺寸保护
- 新增测试 `tests/test_layout.py`（布局纯函数）与 `tests/test_multi_person.py`（分屏流水线）

### 未发布 · 说话人跟随（多人物时对准正在说话的人）

- 新增音频-视觉主动说话人检测（YOLO 判断不了"谁在说话"，故自行补上音频侧）：
  - `core/audio.py`：ffmpeg 抽音轨 → 与视频帧一一对应的语音能量包络 + 语音/静音判定
  - `core/mouth.py`：嘴部 ROI 定位（姿态关键点 → Haar 人脸 → 人框比例三级降级）与
    「嘴部帧间差异 − 上半脸帧间差异」的嘴动强度（可抵消头部整体运动与镜头微动）
  - `core/speaker.py`：候选人 IoU 关联成轨迹，滑窗内做语音-嘴动相关 + 说话时嘴动对比度，
    带**切换滞回**（连续多帧、明显领先才换人），静音与单帧漏检都不换人
- `core/subject.py` 的主角打分支持外部偏置（`bonuses`），`core/pipeline.py` 在关键帧上
  为"正在说话的人"加偏置；单人画面 / 无音轨 / 无 ffmpeg / 检测不到语音时安静退回原规则
- 新增参数 `speaker_tracking`（默认开）/ `speaker_weight` / `speaker_window_frames` /
  `speaker_switch_margin` / `speaker_switch_hold` / `speaker_min_mouth_motion`；
  CLI `--speaker-tracking/--no-speaker-tracking` 等，GUI 第 2 步新增勾选项
- 结果汇总新增「说话人跟随」一行：已启用 / 未启用的原因 + 判定帧数 + 切换次数
- 新增测试 `test_audio.py` / `test_mouth.py` / `test_speaker.py` / `test_speaker_pipeline.py`
- **修复：多人物时判定出了说话人、镜头却没对准他**（融合逻辑的缺陷，非判定精度问题）：
  - 说话人偏置原先直接叠加在**未归一化**的主角打分上：常规得分可到 `2.4` 左右，
    而默认偏置只有 `1.8`，于是一个更大 / 更居中 / 时序更连续的人（尤其上一帧主角）
    会把偏置整段吞掉 —— 现在把主角打分归一化到 `0~1`（除以权重之和，不改变候选人排序），
    偏置与它同量纲，默认 `speaker_weight=1.8` 即"判定出说话人后他必定成为主角"，
    小于 `1.0` 则退化为"软优先"
  - `core/speaker.py` 的轨迹打分改用判定器自己的 `SpeakerParams`（原先固定用内置默认值，
    导致 `speaker_min_mouth_motion` / `min_samples` 等配置对打分完全不生效）
  - `core/pipeline.py` 在"画面里不足两个主要人物"的帧上也会推进嘴动分析器的参考帧，
    避免人物数量变化时把跨多帧的巨大像素差当成嘴动（虚高噪声）

### 未发布 · 镜头稳定（缓解"人动镜头就动"的晕动感）

- `core/smoothing.py` 重构为「自由活动区 + 速度限幅 + 加速度平滑 + 缩放慢速跟随」：
  人物在取景框内走动时镜头**完全不动**，越过边界后也只把人推回边界，不再拽回画面正中
- 新增 `SmoothingParams` 与「镜头跟随」档位预设（锁定 / 舒缓 / 标准 / 跟手），
  CLI `--camera-follow`、GUI 第 2 步与「预览构图效果」共用同一套预设
- 新增参数 `smoothing_max_speed` / `smoothing_accel` / `smoothing_zoom_alpha` /
  `smoothing_zoom_deadzone`；`smoothing_deadzone` 默认值由 `0.02` 提高到 `0.15`
  （「标准」档；此后默认档定为「跟手」`active`，其死区为 `0.06`）
  —— 旧默认几乎等于没有死区，人物一动镜头就跟着动
- 兜底回中改为「忽略死区 + 速度受限」：不再一帧跳到画面中心，也不会停在离中心一个死区的位置

### 0.2.0

- **新增人像构图裁剪**：按用户指定长宽比裁剪，裁剪框以主要人物为中心（`core/ratio.py`、
  `framing.py`、`subject.py`、`smoothing.py`、`crop.py`、`pipeline.py`）
- 新增三级兜底（保持 / 回中 / 全幅居中）、时序平滑与死区、抽帧检测 + 线性插值
- 新增批量帧推理（`--infer-batch`）与姿态关键点精修构图（`core/pose.py`，配 `*-pose.pt` 生效）
- 输出位置与命名改为 `<原名>_person<原后缀>` 且与原视频同目录，同名输出默认跳过
- 新增批量处理（文件夹递归遍历、失败隔离、两级进度、写权限预检）
- GUI 重构为四步向导，新增比例卡片对话框、构图预览、暗色主题（跟随系统 / `V2PV_THEME`）与高 DPI 适配
- 新增可选元数据标记与首帧构图预览图输出
- 新增体积守护（`size_guard` / `max_size_ratio` / `min_video_bitrate_kbps`）：
  输出体积超过原视频时按体积上限反推码率自动压成 H.264，汇总中报出体积倍数；
  批量汇总与 GUI 结果页同步展示体积，GUI 第 2 步新增开关
- 新增不变量验收测试组，`pytest` 覆盖 250+ 用例

### 0.1.0

- 首版：逐帧 YOLO 人像检测 + 画框标注、视频读写、音轨合并 / H.264 转码、
  PySide6 单窗口界面、设备自适应与 `--check` 环境自检

## 许可

Apache-2.0，见 `LICENSE`。使用 YOLO 权重时请遵守 Ultralytics 的许可条款（AGPL-3.0 / 企业授权）。
