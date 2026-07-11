# LYKON / Digital Athlete OS

LYKON 是产品品牌；VANTA COURT 是当前演示中的球队名称，NEXUS CLUB 是对手球队。网站中的品牌、球队和比赛复盘层级按这个关系组织。

黑金高级感篮球数字球员前端演示，按「个人 App + 球队 Web 控制台」的第三种产品形态搭建。

当前版本已接入 Three.js 与 Apache ECharts 的 CDN 运行时：Three.js 负责数字球员场景中的核心、轨道与粒子空间；ECharts 负责能力雷达、近期状态曲线和球队影响力柱状图。

## LYKON 品牌标记

`assets/lykon-approved-transparent.png` 是当前已确认并接入网站的透明背景 LYKON 字母融合狼骑士标记；`assets/lykon-approved.png` 保留为原始黑底稿，`assets/lykon-mark.svg` 和 `assets/proposals/` 保留作为早期矢量探索稿：

- 狼首保留耳朵、眼睛、鼻梁和向右的冲刺方向，保证小尺寸仍然可识别。
- 外部盾形轮廓对应骑士护甲，也适合护臂贴纸和夜光裁切。
- 左侧两条速度切线保留原图的运动感，但减少长尾和复杂笔画。
- 纯金色单色结构适配黑色、白色、金色和夜光材料；网站顶部和页脚使用同一资产。
- `LYKON` 作为字标，`HONOR THE INSTINCT` 作为品牌精神，不塞进小尺寸主符号。

## 当前演示范围

- Player App：数字球员、能力成长、近期状态、设备连接状态、球友匹配。
- Team Web：球队比分、球员 roster、球队影响力、进攻热区、教练视角。
- Digital Replay：NBA 2K 风格的数字球场复盘弹层，包含关键回合、移动轨迹、播放控制和 AI 球员报告。
- Intro Loader：进入网站先播放静音的 LYKON 动作片段，约 5 秒后自动淡出进入主页，也支持手动跳过；`assets/lykon-intro-enhanced.mp4` 为网页播放优化后的 2512×1440 版本。
- Three.js Field：数字球员卡中的动态 3D 核心场景，失败时自动保留 SVG 视觉兜底。
- ECharts Data Surface：能力雷达、状态曲线和团队影响力图表，支持 resize 和 hover tooltip。
- 数据全部使用前端 Mock，后续可将 `app.js` 中的静态展示替换为护臂团队的 API 数据。

## 运行

在当前目录执行：

```bash
python3 -m http.server 4173
```

然后打开 `http://127.0.0.1:4173`。

## 视觉与动效方向

- 黑色底、香槟金、低饱和绿作为主要状态色。
- 参考 React Bits 的 reveal、glow、radar、magnetic/tilt 和扫描式信息层级。
- 参考 Spline 的沉浸式 3D 场景感，用 CSS 3D 光环、数字球员 SVG、数字球场和轨迹动画先做成无外部依赖版本。
- 关键动效都做了 `prefers-reduced-motion` 降级。

## 后续接入数据时的建议

将硬件/算法输出统一成四类：

```js
{
  user: { id, name, position, deviceStatus, battery },
  game: { id, date, team, opponent, score, result, duration },
  events: [{ id, timestamp, type, position, success, impact }],
  profile: {
    overall, shooting, finishing, ballHandling,
    passing, defense, courtIQ, styleTags,
    strengths, recommendations
  }
}
```

前端只依赖「比赛事件、球员能力、回放时间轴」这三层数据，不需要绑定护臂底层传感器实现。
