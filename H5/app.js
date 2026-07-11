const state = {
  mode: 'app',
  replayOpen: false,
  insight: 'moments',
  playing: false,
  replayTime: 522,
  replaySpeed: 1,
  replayTimer: null,
  demoTimer: null,
  notificationsOpen: false,
  lastFocus: null,
  toastTimer: null,
};
const replayTotal = 40 * 60;
const replayEvents = {
  drive: { seconds: 522, fill: '21.8%', message: '已定位到右路突破 · 得分回合', summary: '你在右侧启动吸引协防，随后完成一次急停终结，突破成功率高于同位置球员 18%。', stage: 'DRIVE / SCORING' },
  assist: { seconds: 332, fill: '13.8%', message: '已定位到挡拆助攻 · 决策时间 0.8s', summary: '你在挡拆后 0.8 秒内完成弱侧识别，把防守重心转移成一次有效助攻。', stage: 'SCREEN / ASSIST' },
  defense: { seconds: 191, fill: '8%', message: '已定位到弱侧协防 · STOP', summary: '你从弱侧完成 4.2 米覆盖，提前切断传球线路，完成一次高价值协防。', stage: 'WEAK SIDE / STOP' },
};
const charts = [];
const $ = (selector, scope = document) => scope.querySelector(selector);
const $$ = (selector, scope = document) => [...scope.querySelectorAll(selector)];
const navGroupMap = { overview: 'player', athlete: 'player', signals: 'player', games: 'games', training: 'training', community: 'community' };

const subpages = {
  athlete: {
    eyebrow: 'PLAYER SPACE / DIGITAL ATHLETE',
    title: '数字球员',
    description: '查看你的球员定位、能力构成和最近一次数据更新。',
    content: `<div class="athlete-detail-hero"><div class="athlete-detail-mark">08</div><div><span class="eyebrow">LIN / 08 · VERIFIED ATHLETE</span><h3>TWO-WAY CREATOR</h3><p>真实动作正在持续更新你的数字球员画像。</p></div><span class="status-pill">LIVE DNA</span></div><div class="subpage-stat-grid"><div class="subpage-stat"><span>OVERALL</span><strong>78</strong><small>PLAYER RATING</small><i>▲ 4.2</i></div><div class="subpage-stat"><span>MOTION INDEX</span><strong>87</strong><small>LAST GAME</small><i>▲ 12.4%</i></div><div class="subpage-stat"><span>IMPACT</span><strong>84</strong><small>TOP 12%</small><i>STABLE</i></div></div><div class="subpage-section-head"><div><span class="eyebrow">PLAYER DNA / ROLE PROFILE</span><h3>你的比赛风格</h3></div><span class="gold-label">PG · SG</span></div><div class="dna-grid"><div><span>持球攻击</span><b>82</b><i style="width:82%"></i></div><div><span>挡拆组织</span><b>80</b><i style="width:80%"></i></div><div><span>弱侧防守</span><b>68</b><i style="width:68%"></i></div><div><span>左手终结</span><b>64</b><i style="width:64%"></i></div></div><p class="subpage-note highlight">✦ 下一次训练建议：增加左侧启动和低重心终结，让你的双向创造者画像更加完整。</p>`
  },
  training: {
    eyebrow: 'PLAYER SPACE / MOTION DATA',
    title: '训练轨迹',
    description: '查看护臂采集到的训练负荷、动作稳定性和下一次训练建议。',
    content: `<div class="subpage-stat-grid"><div class="subpage-stat"><span>TRAINING LOAD</span><strong>42</strong><small>MINUTES</small><i>▲ 18%</i></div><div class="subpage-stat"><span>MOTION QUALITY</span><strong>86</strong><small>INDEX</small><i>▲ 6.4%</i></div><div class="subpage-stat"><span>SESSIONS</span><strong>12</strong><small>THIS MONTH</small><i>▲ 3</i></div></div><div class="subpage-section-head"><div><span class="eyebrow">LAST 7 DAYS / TRAINING LOAD</span><h3>训练强度轨迹</h3></div><span class="status-pill">STABLE</span></div><div class="training-chart"><span style="height:34%"></span><span style="height:52%"></span><span style="height:44%"></span><span style="height:73%"></span><span style="height:61%"></span><span style="height:86%"></span><span style="height:68%"></span></div><div class="subpage-days"><span>MON</span><span>TUE</span><span>WED</span><span>THU</span><span>FRI</span><span>SAT</span><span>SUN</span></div><div class="subpage-section-head compact"><div><span class="eyebrow">NEXT SESSION</span><h3>向左突破 · 终结稳定性</h3></div><button class="gold-btn" data-subaction="device"><span>↗</span> 检查设备</button></div><p class="subpage-note">你的右路突破已经形成稳定优势。下一次训练建议增加左手启动和低重心终结，目标完成度 72%。</p>`
  },
  community: {
    eyebrow: 'NETWORK / PLAYER MATCH',
    title: '球友网络',
    description: '根据真实位置、比赛风格和近期状态，找到更适合你的队友。',
    content: `<div class="community-banner"><div><span class="eyebrow">NEARBY PLAYER SIGNAL</span><h3>124 位球友正在场上</h3><p>你当前的打法适合和防守型侧翼、外线射手组成三人组。</p></div><strong>92<small>% MATCH</small></strong></div><div class="subpage-section-head"><div><span class="eyebrow">RECOMMENDED / COMPLEMENTARY STYLE</span><h3>推荐队友</h3></div><button class="filter-btn">NEARBY <span>⌄</span></button></div><div class="community-grid"><article class="community-player"><div class="match-avatar avatar-a">Z</div><div><b>ZHAO / 23</b><span>SF · 防守型侧翼</span><small>和你的突破与组织互补</small></div><strong>92<small>%</small></strong></article><article class="community-player"><div class="match-avatar avatar-b">M</div><div><b>MIKE / 11</b><span>C · 篮板保护者</span><small>适合你的快攻节奏</small></div><strong>88<small>%</small></strong></article><article class="community-player"><div class="match-avatar avatar-c">A</div><div><b>ANNA / 04</b><span>SG · 外线射手</span><small>弱侧空间高度匹配</small></div><strong>84<small>%</small></strong></article></div><button class="wide-btn community-cta" data-subaction="create-game">创建一场球局 <span>→</span></button>`
  },
  device: {
    eyebrow: 'INTELLIGENT SLEEVE / STATUS',
    title: '设备管理',
    description: '查看护臂连接、电量和最近一次数据同步状态。',
    content: `<div class="device-hero"><div class="device-ring-large"><i></i></div><div><span class="eyebrow">PRIMARY DEVICE</span><h3>LYKON SLEEVE / 08</h3><p>智能护臂正在采集运动数据</p></div><span class="device-online">ONLINE</span></div><div class="subpage-stat-grid device-stats"><div class="subpage-stat"><span>BATTERY</span><strong>86</strong><small>PERCENT</small><i>充电状态良好</i></div><div class="subpage-stat"><span>LAST SYNC</span><strong>03</strong><small>MIN AGO</small><i>数据流正常</i></div><div class="subpage-stat"><span>DATA QUALITY</span><strong>98</strong><small>PERCENT</small><i>信号稳定</i></div></div><div class="device-list"><div><span>连接状态</span><b class="text-green">已连接</b></div><div><span>当前模式</span><b>比赛采集</b></div><div><span>固件版本</span><b>LYKON OS 1.0.8</b></div><div><span>数据权限</span><b>个人可见</b></div></div><button class="gold-btn device-sync" data-subaction="sync"><span>↻</span> 立即同步数据</button>`
  },
  signals: {
    eyebrow: 'PLAYER SIGNALS / EVOLUTION',
    title: '能力成长',
    description: '从训练、比赛和连续表现中查看你的能力变化。',
    content: `<div class="signal-tabs"><button class="is-active">30 DAYS</button><button>90 DAYS</button><button>SEASON</button></div><div class="signal-layout"><div class="signal-chart"><div class="signal-line"></div><span class="signal-point point-1"></span><span class="signal-point point-2"></span><span class="signal-point point-3"></span><span class="signal-point point-4"></span><span class="signal-point point-5"></span><div class="signal-axis"><span>JUN 12</span><span>JUN 21</span><span>JUN 28</span></div></div><div class="signal-side"><span class="eyebrow">OVERALL GROWTH</span><strong>+6.8%</strong><p>最近三场比赛的关键回合影响力持续上升。</p><div class="signal-mini-row"><span>投篮</span><b>+4</b></div><div class="signal-mini-row"><span>突破</span><b>+6</b></div><div class="signal-mini-row"><span>球商</span><b>+5</b></div></div></div><div class="subpage-note highlight">✦ 你的下一项成长建议：左侧突破后的终结稳定性。</div>`
  },
  roster: {
    eyebrow: 'TEAM CONSOLE / ROSTER',
    title: '球队球员',
    description: '查看球队中所有已连接球员的比赛状态与影响力。',
    content: `<div class="roster-table"><div class="roster-table-head"><span>PLAYER</span><span>POSITION</span><span>FORM</span><span>IMPACT</span></div><div class="roster-table-row"><b><i class="table-avatar">L</i> LIN YU</b><span>PG / CREATOR</span><span class="text-green">HOT</span><strong>84</strong></div><div class="roster-table-row"><b><i class="table-avatar avatar-light">Z</i> ZHAO MING</b><span>SF / WING</span><span class="text-gold">STEADY</span><strong>79</strong></div><div class="roster-table-row"><b><i class="table-avatar avatar-rose">M</i> MIKE CHEN</b><span>C / ANCHOR</span><span class="text-blue">RISING</span><strong>77</strong></div></div><div class="roster-insight"><span class="spark-icon">✦</span><p><b>阵容建议</b><br/>当前阵容的持球和防守互补度较高，建议让 23 号在第二阵容中承担更多弱侧协防。</p></div>`
  },
  tactical: {
    eyebrow: 'TEAM CONSOLE / TACTICAL MAP',
    title: '球队进攻热区',
    description: '查看全场出手位置、进攻路径和关键挡拆区域。',
    content: `<div class="tactical-hero"><div><span class="eyebrow">OFFENSIVE MAP / GAME 03</span><h3>右侧挡拆创造 6 次有效机会</h3><p>球队在右侧形成高频进攻入口，弱侧空间利用率持续上升。</p></div><div class="tactical-score"><strong>76</strong><span>TEAM IMPACT</span></div></div><div class="subpage-court"><div class="sub-court-half"></div><div class="sub-court-key"></div><div class="sub-court-arc"></div><i class="route route-1"></i><i class="route route-2"></i><i class="route route-3"></i><i class="route-shot shot-a"></i><i class="route-shot shot-b"></i><i class="route-shot shot-c"></i><span>VANTA / OFFENSE RECONSTRUCTION</span></div><button class="gold-btn replay-from-map" data-subaction="replay"><span>▶</span> 播放关键回合</button>`
  }
};

function showToast(message) {
  const toast = $('.toast');
  toast.querySelector('p').textContent = message;
  toast.classList.add('is-show');
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => toast.classList.remove('is-show'), 2800);
}

function finishIntro() {
  const loader = $('.intro-loader');
  if (!loader || loader.classList.contains('is-complete') || loader.classList.contains('is-exiting')) return;
  $('[data-intro-video]')?.pause();
  $('[data-intro-video-bg]')?.pause();
  loader.classList.add('is-exiting');
  document.body.classList.add('intro-revealing');
  document.body.classList.remove('intro-active');
  syncBodyLock();
  window.setTimeout(() => {
    loader.classList.add('is-complete');
    loader.setAttribute('aria-hidden', 'true');
  }, 720);
}

function bindIntroLoader() {
  const loader = $('.intro-loader');
  const video = $('[data-intro-video]');
  const backgroundVideo = $('[data-intro-video-bg]');
  if (!loader || !video) return;

  const progress = $('[data-intro-progress]');
  const updateProgress = () => {
    if (video.duration && Number.isFinite(video.duration) && progress) {
      progress.style.width = `${Math.min(100, (video.currentTime / video.duration) * 100)}%`;
    }
  };
  video.addEventListener('timeupdate', updateProgress);
  video.addEventListener('ended', finishIntro);
  video.addEventListener('error', () => window.setTimeout(finishIntro, 500));
  video.addEventListener('play', () => backgroundVideo?.play().catch(() => {}));
  video.addEventListener('pause', () => backgroundVideo?.pause());
  video.addEventListener('timeupdate', () => {
    if (backgroundVideo && Math.abs(backgroundVideo.currentTime - video.currentTime) > .08) backgroundVideo.currentTime = video.currentTime;
  });
  $('[data-intro-skip]')?.addEventListener('click', finishIntro);

  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    video.pause();
    finishIntro();
    return;
  }

  video.play().catch(() => {
    loader.classList.add('is-tap-required');
    window.setTimeout(finishIntro, 1800);
  });
  backgroundVideo?.play().catch(() => {});
}

function syncBodyLock() {
  document.body.style.overflow = document.body.classList.contains('intro-active') || state.replayOpen || $('.subpage-modal').classList.contains('is-open') ? 'hidden' : '';
}

function setHash(hash) {
  if (window.location.hash !== hash) window.history.pushState({}, '', hash);
}

function formatGameTime(seconds) {
  const safeSeconds = Math.max(0, Math.min(replayTotal, Math.round(seconds)));
  const minutes = String(Math.floor(safeSeconds / 60)).padStart(2, '0');
  const remainder = String(safeSeconds % 60).padStart(2, '0');
  return `${minutes}:${remainder}`;
}

function rememberFocus() {
  state.lastFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
}

function restoreFocus() {
  if (state.lastFocus && document.body.contains(state.lastFocus)) state.lastFocus.focus();
  state.lastFocus = null;
}

function setMode(mode) {
  state.mode = mode;
  $$('.mode-btn').forEach((button) => {
    const active = button.dataset.mode === mode;
    button.classList.toggle('is-active', active);
    button.setAttribute('aria-selected', String(active));
  });
  $$('.surface').forEach((surface) => surface.classList.toggle('is-visible', surface.dataset.surface === mode));
  window.setThreePaused?.(mode !== 'app');
  $$('.reveal').forEach((item) => item.classList.remove('is-in'));
  requestAnimationFrame(() => $$('.surface.is-visible .reveal').forEach((item, index) => setTimeout(() => item.classList.add('is-in'), index * 35)));
  window.setTimeout(resizeCharts, 80);
  if (mode === 'team') showToast('TEAM WEB 已切换 · 你正在查看 VANTA COURT 球队控制台');
}

function setNavGroup(group, { toggle = false } = {}) {
  const target = $(`.nav-section[data-nav-group="${group}"]`);
  if (!target) return;
  const shouldOpen = toggle ? !target.classList.contains('is-open') : true;
  $$('.nav-section').forEach((section) => {
    const isTarget = section === target;
    const open = isTarget ? shouldOpen : false;
    section.classList.toggle('is-open', open);
    const button = $('.nav-section-toggle', section);
    const content = $('.nav-section-content', section);
    button?.setAttribute('aria-expanded', String(open));
    content?.classList.toggle('is-open', open);
  });
}

function toggleNavGroup(group) {
  setNavGroup(group, { toggle: true });
}

function setCollapsible(toggle) {
  const target = document.getElementById(toggle.dataset.collapseToggle);
  if (!target) return;
  const open = !target.classList.contains('is-open');
  target.classList.toggle('is-open', open);
  target.setAttribute('aria-hidden', String(!open));
  toggle.setAttribute('aria-expanded', String(open));
  toggle.classList.toggle('is-open', open);
  const label = toggle.querySelector('span');
  if (label) label.textContent = open ? '收起能力数据' : '查看能力变化';
  window.setTimeout(resizeCharts, 120);
}

function setNav(target, { updateHistory = true } = {}) {
  const group = navGroupMap[target];
  if (group) setNavGroup(group);
  $$('.side-nav [data-nav], .mobile-nav [data-nav], .mobile-nav [data-action]').forEach((link) => {
    link.classList.toggle('is-active', (link.dataset.nav || link.dataset.action) === target);
  });
  if (updateHistory) setHash(`#${target}`);
  if (target === 'overview') {
    closeSubpage({ clearRoute: false });
    closeReplay({ clearRoute: false });
    window.scrollTo({ top: 0, behavior: 'smooth' });
    return;
  }
  if (target === 'games') {
    closeSubpage({ clearRoute: false });
    return openReplay({ push: false });
  }
  if (subpages[target]) {
    closeReplay({ clearRoute: false });
    return openSubpage(target, { push: false });
  }
  if (target === 'notifications') return openNotifications();
}

function enhanceSubpageContent(target) {
  const content = $('#subpage-content');
  content.querySelectorAll('button').forEach((button) => button.setAttribute('type', 'button'));
  if (target === 'signals') {
    const periods = ['30 DAYS', '90 DAYS', 'SEASON'];
    content.querySelectorAll('.signal-tabs button').forEach((button, index) => {
      button.dataset.period = periods[index].toLowerCase().replace(' ', '-');
      button.addEventListener('click', () => selectSignalPeriod(button.dataset.period));
    });
    content.querySelector('.signal-layout')?.setAttribute('data-period', '30-days');
  }
  if (target === 'community') {
    const filter = content.querySelector('.filter-btn');
    if (filter) {
      filter.dataset.subaction = 'community-filter';
      filter.dataset.filterIndex = '0';
    }
  }
  content.querySelectorAll('.gold-btn, .filter-btn, .wide-btn').forEach((button) => button.classList.add('subpage-action-ready'));
  if (window.matchMedia('(pointer: fine)').matches) {
    content.querySelectorAll('.subpage-stat, .community-player, .device-hero, .roster-table, .subpage-court').forEach((targetEl) => {
      targetEl.classList.add('spotlight-target');
      const spotlight = document.createElement('span');
      spotlight.className = 'component-spotlight';
      targetEl.appendChild(spotlight);
      targetEl.addEventListener('pointermove', (event) => {
        const rect = targetEl.getBoundingClientRect();
        targetEl.style.setProperty('--spot-x', `${event.clientX - rect.left}px`);
        targetEl.style.setProperty('--spot-y', `${event.clientY - rect.top}px`);
        targetEl.classList.add('is-hovering');
      });
      targetEl.addEventListener('pointerleave', () => targetEl.classList.remove('is-hovering'));
    });
    content.querySelectorAll('.gold-btn, .filter-btn, .wide-btn').forEach((targetEl) => {
      targetEl.classList.add('magnetic-target');
      targetEl.addEventListener('pointermove', (event) => {
        const rect = targetEl.getBoundingClientRect();
        targetEl.style.setProperty('--mag-x', `${(event.clientX - rect.left - rect.width / 2) * .12}px`);
        targetEl.style.setProperty('--mag-y', `${(event.clientY - rect.top - rect.height / 2) * .12}px`);
        targetEl.classList.add('is-magnetic');
      });
      targetEl.addEventListener('pointerleave', () => {
        targetEl.classList.remove('is-magnetic');
        targetEl.style.removeProperty('--mag-x');
        targetEl.style.removeProperty('--mag-y');
      });
    });
  }
}

function openSubpage(target, { push = true } = {}) {
  const page = subpages[target];
  if (!page) return;
  rememberFocus();
  const modal = $('.subpage-modal');
  $('#subpage-eyebrow').textContent = page.eyebrow;
  $('#subpage-title').textContent = page.title;
  $('#subpage-description').textContent = page.description;
  $('#subpage-content').innerHTML = page.content;
  enhanceSubpageContent(target);
  modal.classList.add('is-open');
  modal.setAttribute('aria-hidden', 'false');
  if (push) setHash(`#${target}`);
  syncBodyLock();
  window.setTimeout(() => {
    $('#subpage-content').classList.add('is-ready');
    $('.subpage-window').focus();
  }, 30);
}

function closeSubpage({ clearRoute = true } = {}) {
  const modal = $('.subpage-modal');
  modal.classList.remove('is-open');
  modal.setAttribute('aria-hidden', 'true');
  $('#subpage-content').classList.remove('is-ready');
  $('#subpage-content').innerHTML = '';
  if (clearRoute && !state.replayOpen) setHash('#overview');
  syncBodyLock();
  if (!state.replayOpen) restoreFocus();
}

function handleSubpageAction(action) {
  if (action === 'replay') {
    closeSubpage({ clearRoute: false });
    openReplay();
    return;
  }
  if (action === 'device') {
    closeSubpage({ clearRoute: false });
    openSubpage('device');
    return;
  }
  if (action === 'sync') {
    const button = $('#subpage-content .device-sync');
    if (!button || button.classList.contains('is-syncing')) return;
    button.classList.add('is-syncing');
    button.innerHTML = '<span>↻</span> 正在同步…';
    showToast('正在同步护臂数据 · 请稍候');
    setTimeout(() => {
      button.classList.remove('is-syncing');
      button.innerHTML = '<span>✓</span> 数据已同步';
      const lastSync = $('#subpage-content .device-stats .subpage-stat:nth-child(2) strong');
      if (lastSync) lastSync.textContent = 'NOW';
      showToast('设备数据同步完成 · 数据质量 98%');
    }, 1200);
    return;
  }
  if (action === 'community-filter') {
    const button = $('#subpage-content .filter-btn');
    const labels = ['NEARBY', 'STYLE MATCH', 'SAME LEVEL'];
    const index = (Number(button.dataset.filterIndex || 0) + 1) % labels.length;
    button.dataset.filterIndex = String(index);
    button.childNodes[0].textContent = `${labels[index]} `;
    showToast(`已切换球友筛选 · ${labels[index]}`);
    return;
  }
  if (action === 'create-game') {
    showToast('球局创建入口已准备好 · 等待你选择时间和场地');
  }
}

function openReplay({ push = true } = {}) {
  const modal = $('.replay-modal');
  rememberFocus();
  stopReplay();
  state.replayOpen = true;
  state.replayTime = replayEvents.drive.seconds;
  modal.classList.add('is-open');
  modal.setAttribute('aria-hidden', 'false');
  $('.replay-window h2').id = 'replay-title';
  $('.replay-header .eyebrow').textContent = 'LYKON / DIGITAL REPLAY · VANTA COURT';
  updateReplayScene();
  if (push) setHash('#games');
  syncBodyLock();
  window.setTimeout(() => $('.replay-window .close-btn').focus(), 30);
}

function closeReplay({ clearRoute = true } = {}) {
  const modal = $('.replay-modal');
  stopReplay();
  state.replayOpen = false;
  modal.classList.remove('is-open');
  modal.setAttribute('aria-hidden', 'true');
  if (clearRoute && !$('.subpage-modal').classList.contains('is-open')) setHash('#overview');
  syncBodyLock();
  $('.play-btn').textContent = '▶';
  restoreFocus();
}

function setInsight(insight) {
  state.insight = insight;
  $$('.insight-tabs button').forEach((button) => button.classList.toggle('is-active', button.dataset.insight === insight));
  $$('.insight-content').forEach((content) => content.classList.toggle('is-active', content.classList.contains(`insight-${insight}`)));
}

function selectSignalPeriod(period) {
  const config = {
    '30-days': { growth: '+6.8%', copy: '最近三场比赛的关键回合影响力持续上升。', values: ['+4', '+6', '+5'], axis: ['JUN 12', 'JUN 21', 'JUN 28'], note: '你的下一项成长建议：左侧突破后的终结稳定性。' },
    '90-days': { growth: '+12.4%', copy: '过去九十天，你的突破和球商成长最为明显。', values: ['+8', '+11', '+9'], axis: ['APR 12', 'MAY 16', 'JUN 28'], note: '九十天趋势显示：左侧终结已经进入稳定提升阶段。' },
    season: { growth: '+18.9%', copy: '本赛季的持续表现，让你的整体影响力进入上升区间。', values: ['+14', '+17', '+13'], axis: ['JAN 08', 'MAR 21', 'JUN 28'], note: '赛季建议：保持防守覆盖距离，继续放大双向价值。' },
  }[period];
  if (!config) return;
  const content = $('#subpage-content');
  const layout = content.querySelector('.signal-layout');
  if (!layout) return;
  layout.dataset.period = period;
  content.querySelectorAll('.signal-tabs button').forEach((button) => button.classList.toggle('is-active', button.dataset.period === period));
  const growth = content.querySelector('.signal-side > strong');
  const copy = content.querySelector('.signal-side > p');
  if (growth) growth.textContent = config.growth;
  if (copy) copy.textContent = config.copy;
  content.querySelectorAll('.signal-mini-row b').forEach((value, index) => { value.textContent = config.values[index]; });
  content.querySelectorAll('.signal-axis span').forEach((value, index) => { value.textContent = config.axis[index]; });
  const note = content.querySelector('.subpage-note');
  if (note) note.textContent = `✦ ${config.note}`;
  showToast(`能力趋势已切换 · ${period === '30-days' ? '30 DAYS' : period === '90-days' ? '90 DAYS' : 'SEASON'}`);
}

function selectEvent(event) {
  $$('.event-item').forEach((item) => item.classList.toggle('is-active', item.dataset.event === event));
  const config = replayEvents[event];
  if (!config) return;
  state.replayTime = config.seconds;
  $('.time-code').textContent = `${formatGameTime(state.replayTime)} / 40:00`;
  $('.scrub-fill').style.width = config.fill;
  $('.scrub-marker').style.left = config.fill;
  const summary = $('.ai-summary p');
  if (summary && config.summary) summary.textContent = config.summary;
  const stageLabel = $('.stage-hud b');
  if (stageLabel && config.stage) stageLabel.textContent = config.stage;
  updateReplayScene();
  showToast(config.message);
}

function seekReplay(event) {
  const scrubber = $('.scrubber');
  if (!scrubber) return;
  const rect = scrubber.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
  state.replayTime = ratio * replayTotal;
  updateReplayScene();
  showToast(`已跳转到比赛 ${formatGameTime(state.replayTime)}`);
}

function updateReplayScene() {
  const percentage = `${(state.replayTime / replayTotal) * 100}%`;
  $('.time-code').textContent = `${formatGameTime(state.replayTime)} / 40:00`;
  $('.scrub-fill').style.width = percentage;
  $('.scrub-marker').style.left = percentage;
  const stage = $('.digital-court');
  if (!stage) return;
  const travelX = Math.max(40, stage.clientWidth * .1);
  const travelY = Math.max(28, stage.clientHeight * .08);
  const wave = state.replayTime * .045;
  $('.p-lin').style.transform = `translate3d(${Math.sin(wave) * travelX}px, ${Math.cos(wave * .7) * travelY}px, 0)`;
  $('.p-zhao').style.transform = `translate3d(${Math.cos(wave * .8) * travelX * .65}px, ${Math.sin(wave) * travelY * .8}px, 0)`;
  $('.p-mike').style.transform = `translate3d(${Math.sin(wave * .6) * travelX * .8}px, ${Math.cos(wave * .9) * travelY}px, 0)`;
  $('.ball').style.transform = `translate3d(${Math.sin(wave * 1.2) * travelX * 1.25}px, ${Math.cos(wave * 1.3) * travelY * 1.6}px, 0)`;
  const current = Object.entries(replayEvents).sort(([, a], [, b]) => Math.abs(a.seconds - state.replayTime) - Math.abs(b.seconds - state.replayTime))[0];
  if (current && Math.abs(current[1].seconds - state.replayTime) < 12) {
    $$('.event-item').forEach((item) => item.classList.toggle('is-active', item.dataset.event === current[0]));
  }
  const frame = $('.stage-hud small');
  if (frame) frame.textContent = `FRAME ${formatGameTime(state.replayTime)}`;
}

function stopReplay() {
  state.playing = false;
  clearInterval(state.replayTimer);
  state.replayTimer = null;
  stopReplayDemo();
  $('.play-btn').textContent = '▶';
  $('.digital-court')?.classList.remove('is-playing');
}

function stopReplayDemo() {
  clearTimeout(state.demoTimer);
  state.demoTimer = null;
  const button = $('.replay-demo-btn');
  if (button) button.innerHTML = '<span>▶</span> WATCH AI RECAP';
  $('.replay-window')?.classList.remove('is-demo-playing');
}

function runReplayDemo() {
  if (!state.replayOpen) return;
  stopReplay();
  const button = $('.replay-demo-btn');
  const sequence = ['drive', 'assist', 'defense'];
  let index = 0;
  if (button) button.innerHTML = '<span>Ⅱ</span> AI RECAP PLAYING';
  $('.replay-window')?.classList.add('is-demo-playing');
  const playNext = () => {
    if (!state.replayOpen || index >= sequence.length) {
      stopReplayDemo();
      showToast('AI 复盘演示完成 · 球员画像已更新');
      return;
    }
    selectEvent(sequence[index]);
    index += 1;
    state.demoTimer = window.setTimeout(playNext, 2400);
  };
  playNext();
}

function toggleReplay() {
  stopReplayDemo();
  if (state.playing) {
    stopReplay();
    showToast('复盘已暂停 · 你可以继续查看关键回合');
    return;
  }
  state.playing = true;
  $('.play-btn').textContent = 'Ⅱ';
  $('.digital-court')?.classList.add('is-playing');
  showToast(`正在播放数字比赛关键回合 · ${state.replaySpeed}×`);
  state.replayTimer = window.setInterval(() => {
    state.replayTime += .8 * state.replaySpeed;
    if (state.replayTime >= replayTotal) {
      state.replayTime = replayTotal;
      updateReplayScene();
      stopReplay();
      showToast('复盘播放完成 · 已回到比赛末段');
      return;
    }
    updateReplayScene();
  }, 120);
}

function toggleReplaySpeed() {
  const speeds = [0.5, 1, 1.5, 2];
  const currentIndex = speeds.indexOf(state.replaySpeed);
  state.replaySpeed = speeds[(currentIndex + 1) % speeds.length];
  const button = $('.speed-btn');
  if (button) button.textContent = `${state.replaySpeed}×`;
  showToast(`复盘速度已切换 · ${state.replaySpeed}×`);
}

function openNotifications() {
  const drawer = $('.notification-drawer');
  state.notificationsOpen = true;
  drawer.classList.add('is-open');
  drawer.setAttribute('aria-hidden', 'false');
  $('[data-open-notifications]').setAttribute('aria-expanded', 'true');
  setTimeout(() => $('.notification-window .close-btn').focus(), 30);
}

function closeNotifications() {
  const drawer = $('.notification-drawer');
  state.notificationsOpen = false;
  drawer.classList.remove('is-open');
  drawer.setAttribute('aria-hidden', 'true');
  $('[data-open-notifications]').setAttribute('aria-expanded', 'false');
}

function markNotifications() {
  $$('.notification-item.is-new').forEach((item) => item.classList.remove('is-new'));
  $('.notification-drawer .eyebrow').textContent = 'SYSTEM FEED / ALL READ';
  $('.notification-dot').style.display = 'none';
  showToast('通知已全部标记为已读');
}

async function shareReplay() {
  const shareText = 'LYKON GAME 03 · LIN / 08 · PLAYER IMPACT 84';
  try {
    await navigator.clipboard.writeText(`${shareText} · ${window.location.origin}${window.location.pathname}#games`);
    showToast('复盘链接已复制 · 可分享给你的球友');
  } catch (error) {
    showToast('复盘已准备好 · 当前浏览器不允许自动复制');
  }
}

function syncRoute() {
  const target = window.location.hash.replace('#', '') || 'overview';
  if (target === 'games') return setNav('games', { updateHistory: false });
  if (target === 'overview') return setNav('overview', { updateHistory: false });
  if (subpages[target]) return setNav(target, { updateHistory: false });
  setNav('overview', { updateHistory: false });
}

function trapDialogFocus(event) {
  const dialog = state.replayOpen ? $('.replay-window') : $('.subpage-modal.is-open .subpage-window');
  if (!dialog || event.key !== 'Tab') return;
  const focusables = $$('button, a, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])', dialog).filter((item) => !item.disabled);
  if (!focusables.length) return;
  const first = focusables[0];
  const last = focusables[focusables.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}

function bindActions() {
  $$('.mode-btn').forEach((button) => button.addEventListener('click', () => setMode(button.dataset.mode)));
  $$('[data-nav-toggle]').forEach((button) => button.addEventListener('click', () => toggleNavGroup(button.dataset.navToggle)));
  $$('.side-nav [data-nav], .mobile-nav [data-nav]').forEach((link) => link.addEventListener('click', (event) => {
    event.preventDefault();
    setNav(link.dataset.nav);
  }));
  $$('[data-collapse-toggle]').forEach((toggle) => toggle.addEventListener('click', () => setCollapsible(toggle)));
  $$('[data-open-replay]').forEach((button) => button.addEventListener('click', openReplay));
  $$('[data-close-replay]').forEach((button) => button.addEventListener('click', closeReplay));
  $$('[data-close-subpage]').forEach((button) => button.addEventListener('click', closeSubpage));
  $$('[data-open-notifications]').forEach((button) => button.addEventListener('click', openNotifications));
  $$('[data-close-notifications]').forEach((button) => button.addEventListener('click', closeNotifications));
  $('[data-mark-notifications]').addEventListener('click', markNotifications);
  $$('.reference-dashboard [data-action="games"]').forEach((button) => button.addEventListener('click', (event) => {
    event.preventDefault();
    event.stopImmediatePropagation();
    openReplay();
  }));
  $$('[data-action]').forEach((button) => button.addEventListener('click', () => setNav(button.dataset.action)));
  $('#subpage-content').addEventListener('click', (event) => {
    const actionTarget = event.target.closest('[data-subaction]');
    if (actionTarget) handleSubpageAction(actionTarget.dataset.subaction);
  });
  $('.play-btn').addEventListener('click', toggleReplay);
  $('.speed-btn').addEventListener('click', toggleReplaySpeed);
  $('.replay-demo-btn').addEventListener('click', runReplayDemo);
  $('.scrubber').addEventListener('pointerdown', seekReplay);
  $$('.event-item').forEach((item) => item.addEventListener('click', () => selectEvent(item.dataset.event)));
  $$('.insight-tabs button').forEach((button) => button.addEventListener('click', () => setInsight(button.dataset.insight)));
  $('.share-btn').addEventListener('click', shareReplay);
  $('.add-roster').addEventListener('click', () => showToast('邀请链接已准备好，可以邀请新的球员加入球队'));
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      if (state.notificationsOpen) closeNotifications();
      else if (state.replayOpen) closeReplay();
      else if ($('.subpage-modal').classList.contains('is-open')) closeSubpage();
      return;
    }
    if (state.replayOpen || $('.subpage-modal').classList.contains('is-open')) trapDialogFocus(event);
  });
  window.addEventListener('hashchange', syncRoute);
  window.addEventListener('popstate', syncRoute);
}

function bindMotion() {
  const observer = new IntersectionObserver((entries) => entries.forEach((entry) => { if (entry.isIntersecting) entry.target.classList.add('is-in'); }), { threshold: .08 });
  $$('.reveal').forEach((item) => observer.observe(item));
  $$('.tilt-card').forEach((card) => {
    card.addEventListener('pointermove', (event) => {
      if (window.matchMedia('(pointer: coarse)').matches) return;
      const rect = card.getBoundingClientRect();
      const x = (event.clientX - rect.left) / rect.width - .5;
      const y = (event.clientY - rect.top) / rect.height - .5;
      card.style.transform = `perspective(1100px) rotateX(${y * -2}deg) rotateY(${x * 2}deg)`;
    });
    card.addEventListener('pointerleave', () => { card.style.transform = ''; });
  });
}

function initAdvancedMotion() {
  const finePointer = window.matchMedia('(pointer: fine)');
  if (!finePointer.matches) return;
  const root = document.documentElement;
  const spotlightTargets = $$('.panel, .match-card, .event-item, .mini-court');
  spotlightTargets.forEach((target) => {
    target.classList.add('spotlight-target');
    const spotlight = document.createElement('span');
    spotlight.className = 'component-spotlight';
    target.appendChild(spotlight);
    target.addEventListener('pointermove', (event) => {
      const rect = target.getBoundingClientRect();
      target.style.setProperty('--spot-x', `${event.clientX - rect.left}px`);
      target.style.setProperty('--spot-y', `${event.clientY - rect.top}px`);
      target.classList.add('is-hovering');
    });
    target.addEventListener('pointerleave', () => target.classList.remove('is-hovering'));
  });

  const magnets = $$('.mode-btn, .outline-btn, .gold-btn, .wide-btn, .play-btn, .close-btn, .share-btn, .mini-action, .text-btn, .filter-btn, .player-action, .replay-demo-btn');
  magnets.forEach((target) => {
    target.classList.add('magnetic-target');
    target.addEventListener('pointermove', (event) => {
      const rect = target.getBoundingClientRect();
      const x = (event.clientX - rect.left - rect.width / 2) * .12;
      const y = (event.clientY - rect.top - rect.height / 2) * .12;
      target.style.setProperty('--mag-x', `${x}px`);
      target.style.setProperty('--mag-y', `${y}px`);
      target.classList.add('is-magnetic');
    });
    target.addEventListener('pointerleave', () => {
      target.classList.remove('is-magnetic');
      target.style.removeProperty('--mag-x');
      target.style.removeProperty('--mag-y');
    });
  });

  const cursorOrb = document.createElement('span');
  cursorOrb.className = 'cursor-orb';
  document.body.appendChild(cursorOrb);
  window.addEventListener('pointermove', (event) => {
    root.style.setProperty('--cursor-x', `${event.clientX}px`);
    root.style.setProperty('--cursor-y', `${event.clientY}px`);
    cursorOrb.style.transform = `translate3d(${event.clientX}px, ${event.clientY}px, 0)`;
  }, { passive: true });
  document.addEventListener('pointerdown', () => cursorOrb.classList.add('is-pressed'));
  document.addEventListener('pointerup', () => cursorOrb.classList.remove('is-pressed'));
}

function initCharts() {
  if (!window.echarts) return;
  const gold = '#e9b867';
  const pale = '#fff0c7';
  const muted = '#6d6b66';
  const chartBase = { animationDuration: 1200, animationEasing: 'cubicOut' };
  const abilityEl = $('#ability-chart');
  if (abilityEl) {
    const abilityChart = window.echarts.init(abilityEl, null, { renderer: 'canvas' });
    abilityChart.setOption({
      ...chartBase,
      radar: { center: ['50%', '50%'], radius: '72%', splitNumber: 4, indicator: [{ max: 100 }, { max: 100 }, { max: 100 }, { max: 100 }, { max: 100 }, { max: 100 }], axisName: { show: false }, axisLine: { lineStyle: { color: 'rgba(232,186,103,.18)' } }, splitLine: { lineStyle: { color: 'rgba(232,186,103,.18)' } }, splitArea: { areaStyle: { color: ['rgba(255,255,255,.01)', 'rgba(221,169,76,.035)'] } } },
      series: [{ type: 'radar', symbol: 'circle', symbolSize: 4, lineStyle: { color: gold, width: 1.5, shadowBlur: 11, shadowColor: gold }, itemStyle: { color: pale, borderColor: gold }, areaStyle: { color: 'rgba(222,165,77,.24)' }, data: [{ value: [74, 82, 68, 80, 76, 79] }] }],
    });
    charts.push(abilityChart);
  }
  const impactEl = $('#impact-chart');
  if (impactEl) {
    const impactChart = window.echarts.init(impactEl, null, { renderer: 'canvas' });
    impactChart.setOption({
      ...chartBase,
      grid: { top: 6, right: 0, bottom: 0, left: 0 },
      xAxis: { type: 'category', show: false, data: ['JUN 12', 'JUN 15', 'JUN 18', 'JUN 21', 'JUN 24', 'JUN 28'] },
      yAxis: { type: 'value', min: 50, max: 100, show: false },
      tooltip: { trigger: 'axis', backgroundColor: '#17130d', borderColor: 'rgba(232,186,103,.35)', textStyle: { color: pale, fontSize: 10 }, formatter: (params) => `${params[0].axisValue}<br/><b>FORM INDEX ${params[0].value}</b>` },
      series: [{ type: 'line', smooth: true, symbol: 'none', data: [61, 66, 64, 73, 79, 87], lineStyle: { color: gold, width: 2, shadowBlur: 10, shadowColor: gold }, areaStyle: { color: new window.echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(232,181,92,.34)' }, { offset: 1, color: 'rgba(232,181,92,0)' }]) } }],
    });
    charts.push(impactChart);
  }
  const teamEl = $('#team-impact-chart');
  if (teamEl) {
    const teamChart = window.echarts.init(teamEl, null, { renderer: 'canvas' });
    teamChart.setOption({
      ...chartBase,
      grid: { top: 2, right: 0, bottom: 1, left: 0 },
      xAxis: { type: 'category', show: false, data: ['Q1', 'Q2', 'Q3', 'Q4', 'OT'] },
      yAxis: { type: 'value', show: false, max: 100 },
      series: [{ type: 'bar', barWidth: '38%', data: [34, 49, 43, 67, 88], itemStyle: { borderRadius: [3, 3, 0, 0], color: new window.echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: '#f4cd86' }, { offset: 1, color: '#86531c' }]) }, animationDelay: (index) => index * 80 }],
    });
    charts.push(teamChart);
  }
  if (charts.length) document.body.classList.add('echarts-ready');
}

function resizeCharts() {
  charts.forEach((chart) => chart.resize());
}

function initPlayerRender() {
  const image = $('#player-render');
  if (!image) return;
  const showRender = () => document.body.classList.add('face-render-ready');
  image.addEventListener('load', showRender, { once: true });
  image.addEventListener('error', () => image.remove(), { once: true });
  if (image.complete && image.naturalWidth) showRender();
}

async function initThreeScene() {
  const canvas = $('#player-canvas');
  if (!canvas) return;
  try {
    const THREE = await import('https://cdn.jsdelivr.net/npm/three@0.185.0/build/three.module.js');
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(30, 1, .1, 100);
    camera.position.set(0, .12, 4.55);
    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setClearColor(0x000000, 0);
    const field = new THREE.Group();
    scene.add(field);
    const gold = new THREE.MeshStandardMaterial({ color: 0xe2ae60, metalness: .82, roughness: .24, emissive: 0x5c3410, emissiveIntensity: .65 });
    const skin = new THREE.MeshStandardMaterial({ color: 0xf0ba69, metalness: .42, roughness: .3, emissive: 0x61340b, emissiveIntensity: .38 });
    const jersey = new THREE.MeshStandardMaterial({ color: 0x28292d, metalness: .5, roughness: .34, emissive: 0x21150a, emissiveIntensity: .48 });
    const jerseyEdge = new THREE.MeshStandardMaterial({ color: 0xb9782b, metalness: .72, roughness: .3, emissive: 0x4d260b, emissiveIntensity: .55 });
    const hair = new THREE.MeshStandardMaterial({ color: 0x080a0c, metalness: .65, roughness: .25, emissive: 0x090704, emissiveIntensity: .3 });
    const shoe = new THREE.MeshStandardMaterial({ color: 0xe9ddbd, metalness: .42, roughness: .28, emissive: 0x503819, emissiveIntensity: .24 });
    const ballMaterial = new THREE.MeshStandardMaterial({ color: 0xd7882d, metalness: .48, roughness: .34, emissive: 0x51240b, emissiveIntensity: .5 });
    const ballSeam = new THREE.MeshBasicMaterial({ color: 0x2b1710, transparent: true, opacity: .88 });
    const dimGold = new THREE.MeshBasicMaterial({ color: 0xc58b3d, transparent: true, opacity: .26, wireframe: true });
    const avatar = new THREE.Group();
    avatar.name = 'digital-athlete';
    avatar.position.set(0, -.18, .04);
    avatar.scale.setScalar(.9);
    field.add(avatar);

    const makePart = (geometry, material, position, scale) => {
      const mesh = new THREE.Mesh(geometry, material);
      mesh.position.set(...position);
      if (scale) mesh.scale.set(...scale);
      avatar.add(mesh);
      return mesh;
    };
    const makeLimb = (start, end, radius, material) => {
      const from = new THREE.Vector3(...start);
      const to = new THREE.Vector3(...end);
      const direction = to.clone().sub(from);
      const mesh = new THREE.Mesh(new THREE.CylinderGeometry(radius * .9, radius, direction.length(), 8), material);
      mesh.position.copy(from.clone().add(to).multiplyScalar(.5));
      mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
      avatar.add(mesh);
      return mesh;
    };

    // A deliberately faceted player keeps the card lightweight while making the
    // silhouette read as a real, lit 3D athlete instead of a flat illustration.
    makePart(new THREE.SphereGeometry(.225, 16, 12), skin, [0, 1.25, .1]);
    makePart(new THREE.SphereGeometry(.23, 16, 8), hair, [0, 1.37, -.06], [1, .58, 1]);
    makePart(new THREE.SphereGeometry(.14, 12, 8), skin, [0, 1.24, .29], [1.18, .78, .3]);
    makePart(new THREE.BoxGeometry(.22, .035, .018), hair, [0, 1.34, .34]);
    makePart(new THREE.CylinderGeometry(.1, .11, .16, 8), skin, [0, 1.03, .02]);
    makePart(new THREE.CylinderGeometry(.3, .36, .72, 6), jersey, [0, .65, 0]);
    makePart(new THREE.BoxGeometry(.25, .34, .028), new THREE.MeshStandardMaterial({ color: 0x4c3420, metalness: .6, roughness: .3, emissive: 0x3e1d08, emissiveIntensity: .55 }), [0, .67, .31]);
    makePart(new THREE.BoxGeometry(.028, .16, .018), jerseyEdge, [-.055, .68, .335]);
    makePart(new THREE.BoxGeometry(.028, .16, .018), jerseyEdge, [.055, .68, .335]);
    makePart(new THREE.IcosahedronGeometry(.045, 1), gold, [0, .88, .34]);
    makePart(new THREE.TorusGeometry(.315, .014, 6, 32), jerseyEdge, [0, .98, 0], [1, 1, .88]).rotation.x = Math.PI / 2;
    makePart(new THREE.TorusGeometry(.35, .018, 6, 32), jerseyEdge, [0, .31, 0], [1, 1, .88]).rotation.x = Math.PI / 2;
    makePart(new THREE.BoxGeometry(.68, .28, .34), jersey, [0, .17, .01]);
    makePart(new THREE.BoxGeometry(.7, .04, .36), jerseyEdge, [0, .31, .01]);

    const leftShoulder = [-.25, .84, .01];
    const leftElbow = [-.5, .57, .06];
    const leftHand = [-.36, .3, .16];
    const rightShoulder = [.25, .84, .01];
    const rightElbow = [.49, .64, .1];
    const rightHand = [.58, .38, .25];
    makePart(new THREE.SphereGeometry(.12, 10, 8), jerseyEdge, leftShoulder);
    makePart(new THREE.SphereGeometry(.12, 10, 8), jerseyEdge, rightShoulder);
    makeLimb(leftShoulder, leftElbow, .095, skin);
    makeLimb(leftElbow, leftHand, .085, skin);
    makeLimb(rightShoulder, rightElbow, .095, skin);
    makeLimb(rightElbow, rightHand, .085, skin);
    makePart(new THREE.SphereGeometry(.1, 10, 8), skin, leftElbow);
    makePart(new THREE.SphereGeometry(.1, 10, 8), skin, rightElbow);
    makePart(new THREE.SphereGeometry(.085, 10, 8), skin, leftHand);
    makePart(new THREE.SphereGeometry(.085, 10, 8), skin, rightHand);

    const leftHip = [-.17, .02, .01];
    const leftKnee = [-.22, -.45, .03];
    const leftAnkle = [-.25, -.91, .06];
    const rightHip = [.17, .02, .01];
    const rightKnee = [.22, -.45, .03];
    const rightAnkle = [.29, -.91, .08];
    makeLimb(leftHip, leftKnee, .115, jersey);
    makeLimb(leftKnee, leftAnkle, .09, jersey);
    makeLimb(rightHip, rightKnee, .115, jersey);
    makeLimb(rightKnee, rightAnkle, .09, jersey);
    makePart(new THREE.SphereGeometry(.11, 10, 8), jerseyEdge, leftKnee);
    makePart(new THREE.SphereGeometry(.11, 10, 8), jerseyEdge, rightKnee);
    makePart(new THREE.BoxGeometry(.34, .16, .5), shoe, [-.25, -1.03, .12], [1, 1, 1.08]);
    makePart(new THREE.BoxGeometry(.36, .16, .5), shoe, [.31, -1.03, .14], [1, 1, 1.08]);
    makePart(new THREE.BoxGeometry(.39, .035, .52), jerseyEdge, [-.25, -1.105, .12]);
    makePart(new THREE.BoxGeometry(.41, .035, .52), jerseyEdge, [.31, -1.105, .14]);

    const ballGroup = new THREE.Group();
    ballGroup.position.set(.73, .35, .35);
    avatar.add(ballGroup);
    ballGroup.add(new THREE.Mesh(new THREE.SphereGeometry(.16, 16, 12), ballMaterial));
    const seamOne = new THREE.Mesh(new THREE.TorusGeometry(.16, .008, 8, 32), ballSeam);
    seamOne.rotation.y = Math.PI / 2;
    ballGroup.add(seamOne);
    const seamTwo = new THREE.Mesh(new THREE.TorusGeometry(.16, .008, 8, 32), ballSeam);
    seamTwo.rotation.x = Math.PI / 2;
    seamTwo.rotation.z = .55;
    ballGroup.add(seamTwo);
    const shadow = new THREE.Mesh(new THREE.CircleGeometry(.48, 32), new THREE.MeshBasicMaterial({ color: 0xb6732a, transparent: true, opacity: .18 }));
    shadow.rotation.x = -Math.PI / 2;
    shadow.position.set(0, -1.14, .02);
    shadow.scale.set(1.45, .42, 1);
    avatar.add(shadow);
    const halo = new THREE.Mesh(new THREE.TorusGeometry(.78, .012, 8, 64), dimGold);
    halo.rotation.x = Math.PI / 2.2;
    field.add(halo);
    const haloTwo = new THREE.Mesh(new THREE.TorusGeometry(1.02, .008, 8, 64), dimGold);
    haloTwo.rotation.x = Math.PI / 2.8;
    haloTwo.rotation.z = .38;
    field.add(haloTwo);
    const particleCount = 360;
    const positions = new Float32Array(particleCount * 3);
    for (let index = 0; index < particleCount; index += 1) {
      const radius = .85 + Math.random() * .55;
      const angle = Math.random() * Math.PI * 2;
      positions[index * 3] = Math.cos(angle) * radius;
      positions[index * 3 + 1] = (Math.random() - .5) * 1.8;
      positions[index * 3 + 2] = Math.sin(angle) * radius * .55;
    }
    const particleGeometry = new THREE.BufferGeometry();
    particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const particles = new THREE.Points(particleGeometry, new THREE.PointsMaterial({ color: 0xf2c679, size: .018, transparent: true, opacity: .62, blending: THREE.AdditiveBlending }));
    field.add(particles);
    scene.add(new THREE.AmbientLight(0x6f4a23, 1.35));
    const keyLight = new THREE.PointLight(0xf0c477, 6, 7);
    keyLight.position.set(1.8, 1.7, 2.8);
    scene.add(keyLight);
    const fillLight = new THREE.PointLight(0x315d52, 3, 5);
    fillLight.position.set(-2, -.5, 1.5);
    scene.add(fillLight);
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(rect.width, 1);
      const height = Math.max(rect.height, 1);
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    };
    resize();
    window.addEventListener('resize', resize);
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let paused = document.hidden || reduceMotion || state.mode !== 'app';
    let animationFrame = null;
    const animate = (time) => {
      animationFrame = null;
      if (paused) return;
      const seconds = time * .001;
      field.rotation.y = seconds * .05;
      field.rotation.x = Math.sin(seconds * .45) * .06;
      avatar.rotation.y = Math.sin(seconds * .34) * .2;
      avatar.rotation.z = Math.sin(seconds * .52) * .012;
      avatar.position.y = -.18 + Math.sin(seconds * 1.15) * .024;
      ballGroup.rotation.y = seconds * .62;
      halo.rotation.z = seconds * -.32;
      haloTwo.rotation.z = seconds * .24;
      particles.rotation.y = seconds * -.08;
      renderer.render(scene, camera);
      animationFrame = window.requestAnimationFrame(animate);
    };
    window.setThreePaused = (nextPaused) => {
      paused = nextPaused || document.hidden || reduceMotion;
      if (!paused && !animationFrame) animationFrame = window.requestAnimationFrame(animate);
    };
    document.addEventListener('visibilitychange', () => window.setThreePaused(document.hidden));
    canvas.setAttribute('aria-label', '3D digital basketball player avatar');
    const status = $('.three-status');
    if (status) status.innerHTML = '<i></i> 3D ATHLETE ONLINE';
    document.body.classList.add('three-ready');
    if (!paused) animationFrame = window.requestAnimationFrame(animate);
    else renderer.render(scene, camera);
  } catch (error) {
    canvas.remove();
  }
}

bindIntroLoader();
bindActions();
bindMotion();
initAdvancedMotion();
initCharts();
initPlayerRender();
initThreeScene();
window.addEventListener('resize', resizeCharts);
syncRoute();
setTimeout(() => $$('.surface.is-visible .reveal').forEach((item) => item.classList.add('is-in')), 100);
