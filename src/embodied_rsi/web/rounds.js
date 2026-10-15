/* Shared live RSI state and durable per-round comparisons. */
(() => {
  'use strict';
  const node = id => document.getElementById(id);
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const pct = n => n == null ? '—' : `${(n * 100).toFixed(1)}%`;
  const loss = n => n == null ? '—' : n.toFixed(4);
  const colors = ['#73e3ba', '#7aaaff', '#c49bff', '#ffbb76', '#f58ead', '#68d8ee'];
  let report = null, live = {}, polling = false, phaseHTML = '', iterationKey = '', successHTML = '', barKey = '';
  const statusName = {running:'进行中', completed:'完成', failed:'失败', cancelled:'已取消'};
  const host = node('round-comparison');
  if (host) host.innerHTML = `<section class="panel round-panel"><div class="panel-head"><div><h2>连续采集 · 按模型版本分轮</h2><p>记录每局完成后的累计成功率；换模型后另起一轮，服务重启继续累计。探索采集与独立评测分别统计。</p></div><a class="button" href="/api/rounds/export/collection-rounds">导出 CSV ↓</a></div><div class="table-wrap"><table><thead><tr><th>运行模型</th><th>完整尝试</th><th>成功</th><th>失败</th><th>采集成功率</th><th>决策方式</th></tr></thead><tbody id="round-collection"></tbody></table></div><div id="collection-curves" class="round-chart"></div><div id="collection-legend" class="round-legend"></div><details class="episode-results"><summary>逐局任务执行记录 · 每局结果与累计成功率</summary><p>显示最近 50 局；全部记录可导出 CSV。累计成功率在各模型版本内单独计算。</p><div class="table-wrap"><table><thead><tr><th>模型轮次</th><th>本版本第几局</th><th>场景</th><th>任务结果</th><th>累计成功 / 总局数</th><th>累计成功率</th><th>完成时间</th></tr></thead><tbody id="episode-results"></tbody></table></div></details><p id="round-updated">加载记录中…</p><a class="metric-link" href="/api/rounds/export/collection-episodes">导出逐局成功率与场景种子 ↓</a></section>
  <section class="panel round-panel"><div class="panel-head"><div><h2>每轮训练与任务完成率对比</h2><p>Loss 为训练 batch 损失。每轮使用新的配对评测场景；以本轮候选与父模型的差值判断提升。</p></div><a class="button" href="/api/rounds/export/training-rounds">导出 CSV ↓</a></div><div class="table-wrap"><table><thead><tr><th>训练轮</th><th>动作样本 / 成功轨迹</th><th>Loss 首次 → 末次</th><th>已记录更新</th><th>开发完成率 前 → 选中</th><th>选中更新</th><th>评测 候选 / 父模型</th><th>配对进度</th><th>同批提升</th><th>发布结果</th></tr></thead><tbody id="round-training"></tbody></table></div><div id="round-loss-curves" class="round-chart"></div><div id="round-loss-legend" class="round-legend"></div><p>历史轮次保留原有每 5 步记录；新启动的训练每次更新记录 loss。没有记录的点不补写。</p><a class="metric-link" href="/api/rounds/export/training-loss">导出全部已记录 loss ↓</a></section>`;

  function curves(id, legend, series, rate = false) {
    if (!node(id)) return;
    const points = series.flatMap(s => s.points);
    if (!points.length) { node(id).innerHTML = '<p>等待记录</p>'; return; }
    const maxX = Math.max(1, ...points.map(p => p[0]));
    const maxY = rate ? 1 : Math.max(.001, ...points.map(p => p[1])) * 1.1;
    const x = v => 52 + v / maxX * 840, y = v => 138 - v / maxY * 112;
    node(id).innerHTML = `<svg viewBox="0 0 920 168" role="img" aria-label="${rate?'每个模型版本的逐局采集成功率':'每轮训练的 loss 曲线'}">${[0,.5,1].map(f => `<line x1="52" y1="${y(maxY*f)}" x2="892" y2="${y(maxY*f)}" stroke="#263847"/><text x="2" y="${y(maxY*f)+4}" fill="#96aaba" font-size="10">${rate?pct(maxY*f):(maxY*f).toFixed(2)}</text>`).join('')}${series.map((s,i)=>`<polyline fill="none" stroke="${colors[i%colors.length]}" stroke-width="2" points="${s.points.map(p=>`${x(p[0])},${y(p[1])}`).join(' ')}"/>${s.points.map(p=>`<circle cx="${x(p[0])}" cy="${y(p[1])}" r="2" fill="${colors[i%colors.length]}"><title>${escape(s.name)} · ${p[0]} ${rate?'局':'步'} · ${rate?pct(p[1]):loss(p[1])}</title></circle>`).join('')}`).join('')}<text x="52" y="160" fill="#96aaba" font-size="10">0</text><text x="892" y="160" text-anchor="end" fill="#96aaba" font-size="10">${maxX} ${rate?'局（各版本独立累计）':'次更新'}</text></svg>`;
    node(legend).innerHTML = series.map((s,i)=>`<span style="color:${colors[i%colors.length]}">● ${escape(s.name)}</span>`).join('');
  }
  function renderReport() {
    if (!report || !host) return;
    node('round-collection').innerHTML = report.collection.map(c=>`<tr title="${escape(c.model_version)}" class="${c.model_version===live.version?'current-model':''}"><td>${c.model_round==null?'指定模型':c.model_round===0?'V0 · 初始模型':`V${c.model_round}`}${c.model_version===live.version?' · 正在运行':''}<small class="round-id">${escape(c.model_version)}</small></td><td>${c.episodes}</td><td>${c.successes}</td><td>${c.failures}</td><td class="green">${pct(c.success_rate)}</td><td>${c.protocols.map(p=>p==='stochastic'?'随机探索':'确定性决策').join(' / ')}</td></tr>`).join('') || '<tr><td colspan="6">等待机械臂完成第一局采集</td></tr>';
    node('round-training').innerHTML = report.training.map(r=>{
      const state = r.deployment ? `已部署 V${r.model_round}` : r.gate_passed===false ? '未通过 · 保留当前模型' : r.evaluation_status==='running' ? '评测中' : ['failed','cancelled'].includes(r.evaluation_status) ? `评测${statusName[r.evaluation_status]}` : r.status==='running' ? '训练中' : r.gate_passed===true ? '通过 · 等待部署' : r.status==='completed' ? '等待评测' : statusName[r.status] || '等待评测';
      return `<tr><td title="${escape(r.training)}">第 ${r.round} 轮</td><td>${r.training_samples??'—'} / ${r.successful_episodes??'—'}</td><td>${loss(r.loss_first)} → ${loss(r.loss_last)}</td><td>${r.logged_updates} / ${r.updates??'—'}</td><td>${pct(r.dev_before)} → ${pct(r.dev_selected)}</td><td>${r.selected_step??'—'}</td><td>${pct(r.candidate_success_rate)} / ${pct(r.parent_success_rate)}${r.evaluation_status==='running'?'（临时）':''}</td><td>${r.evaluation?`${r.evaluated_pairs} / ${r.total_pairs}`:'—'}</td><td>${r.paired_gain==null?'—':`${r.paired_gain>=0?'+':''}${(r.paired_gain*100).toFixed(1)} pp`}</td><td class="${r.deployment?'green':''}">${escape(state)}</td></tr>`;
    }).join('') || '<tr><td colspan="10">等待第一轮训练</td></tr>';
    curves('collection-curves','collection-legend',report.collection.map(c=>({name:c.model_round==null?c.model_version:`V${c.model_round}`,points:c.curve.map(p=>[p.episode,p.success_rate])})),true);
    curves('round-loss-curves','round-loss-legend',report.training.map(r=>({name:`第 ${r.round} 轮`,points:r.loss_curve.map(p=>[p.step,p.loss])})));
    const episodes=report.collection.flatMap(c=>c.curve.map(point=>({...point,model_version:c.model_version,model_round:c.model_round}))).sort((a,b)=>b.ts-a.ts).slice(0,50);
    node('episode-results').innerHTML=episodes.map(e=>`<tr><td title="${escape(e.model_version)}">${e.model_round==null?'指定模型':`V${e.model_round}`}</td><td>第 ${e.episode} 局</td><td>${escape(e.seed)}</td><td class="${e.success?'green':'episode-failure'}">${e.success?'✓ 完成':'✗ 未完成'}</td><td>${e.successes} / ${e.episode}</td><td>${pct(e.success_rate)}</td><td>${new Date(e.ts*1000).toLocaleTimeString('zh-CN')}</td></tr>`).join('')||'<tr><td colspan="7">等待第一局完整任务结果</td></tr>';
    node('round-updated').textContent = `记录更新：${new Date(report.generated_at*1000).toLocaleTimeString('zh-CN')} · 每 5 秒保存到本机`;
    renderState();
  }
  function renderSuccessBars() {
    const chart=node('success-bar-chart'); if(!chart||!report)return;
    const evaluation=node('success-bar-mode').value==='evaluation';
    const rows=evaluation?report.training.map(r=>({label:`第 ${r.round} 轮`,rates:[r.parent_success_rate,r.candidate_success_rate],run:r.training,partial:r.evaluation_status!=='completed',detail:`${r.evaluated_pairs}/${r.total_pairs??'—'} 对场景 · ${r.deployment?'已发布':r.evaluation_status==='running'?'评测中':'未发布'}`})):report.collection.map(c=>({label:c.model_round==null?'指定模型':`V${c.model_round}`,rates:[c.success_rate],partial:false,detail:`${c.successes}/${c.episodes} 局完成`}));
    const key=JSON.stringify([evaluation,rows]); if(key===barKey)return; barKey=key;
    node('success-bar-note').textContent=evaluation?'每轮在同一批新场景比较候选与父模型；半透明点为进行中的结果。':'各运行版本的探索采集成绩；成功局数 ÷ 已完成局数。';
    const series=evaluation?[{name:'当轮父模型',color:'#7aaaff'},{name:'训练后候选',color:'#73e3ba'}]:[{name:'探索采集成功率',color:'#73e3ba'}];
    node('success-bar-legend').innerHTML=series.map(s=>`<span style="color:${s.color}">● ${s.name}</span>`).join('');
    if(!rows.length){chart.innerHTML='<div class="history-empty"><span>等待第一轮成绩</span><p>完成采集和训练后，真实评测结果会显示在这里。</p></div>';node('success-bar-summary').textContent='每轮如实记录，包含持平、下降与未发布。';return;}
    const width=640,height=300,left=48,right=28,top=28,bottom=242;
    const x=i=>rows.length===1?(left+width-right)/2:left+i/(rows.length-1)*(width-left-right), y=v=>bottom-v*(bottom-top);
    let svg=`<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="历史轮次任务成功率曲线"><title>完整任务成功率，按真实记录绘制</title>`;
    svg+=[0,.25,.5,.75,1].map(rate=>`<line x1="${left}" x2="${width-right}" y1="${y(rate)}" y2="${y(rate)}" stroke="#2a4150" stroke-dasharray="3 5"/><text x="${left-10}" y="${y(rate)+4}" text-anchor="end" fill="#91a9bb" font-size="11">${Math.round(rate*100)}%</text>`).join('');
    series.forEach((s,j)=>{
      let segment=[];
      const flush=()=>{if(segment.length>1)svg+=`<polyline points="${segment.join(' ')}" fill="none" stroke="${s.color}" stroke-width="2.5" stroke-linejoin="round"/>`;segment=[];};
      rows.forEach((r,i)=>{const rate=r.rates[j]; if(Number.isFinite(rate))segment.push(`${x(i)},${y(rate)}`);else flush();}); flush();
      svg+=rows.map((r,i)=>{const rate=r.rates[j];if(!Number.isFinite(rate))return '';const title=`${r.label} · ${s.name} ${pct(rate)} · ${r.detail}${r.partial?' · 临时成绩':''}`;return `<g class="history-point" data-index="${i}" tabindex="0" role="button" aria-label="${escape(title)}"><title>${escape(title)}</title><circle cx="${x(i)}" cy="${y(rate)}" r="12" fill="transparent"/><circle cx="${x(i)}" cy="${y(rate)}" r="4.5" fill="${s.color}" stroke="#102330" stroke-width="2" opacity="${r.partial?'.5':'1'}"/></g>`;}).join('');
    });
    const tick=Math.max(1,Math.ceil(rows.length/8));
    svg+=rows.map((r,i)=>i%tick===0||i===rows.length-1?`<text x="${x(i)}" y="${bottom+25}" text-anchor="middle" fill="#9eb8c8" font-size="11">${escape(r.label)}</text>`:'').join('');
    chart.innerHTML=svg+'</svg>'; chart.style.setProperty('--bar-width','0px');
    const latest=rows.at(-1), rate=latest.rates.at(-1);
    node('success-bar-summary').textContent=`${evaluation?`已记录 ${rows.length} 轮学习`:`已记录 ${rows.length} 个运行版本`} · 最新${evaluation?'候选':'采集'} ${pct(rate)}${latest.partial?'（评测中）':''} · ${latest.detail}`;
    chart.querySelectorAll('.history-point').forEach(point=>{
      const select=()=>{const r=rows[Number(point.dataset.index)];if(!r.run)return;const card=Array.from(document.querySelectorAll('.iteration-card')).find(c=>c.dataset.run===r.run);if(card){card.open=true;card.scrollIntoView({behavior:'smooth',block:'nearest'});}};
      point.onclick=select;point.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}};
    });
  }
  if(node('success-bar-mode'))node('success-bar-mode').onchange=()=>{barKey='';renderSuccessBars();};
  function renderIteration(currentModel) {
    const panel=node('rsi-iteration');if(!panel||!report)return;
    const rows=report.training;
    const key=JSON.stringify([live.version,live.status,report.active.map(p=>[p.run,p.kind,p.stage,p.stale]),rows.map(r=>[r.training,r.status,r.logged_updates,r.loss_last,r.evaluation_status,r.evaluated_pairs,r.candidate_success_rate,r.parent_success_rate,r.gate_passed,r.deployment,r.model_round]),currentModel?.episodes]);
    if(key===iterationKey)return;iterationKey=key;
    const expanded=new Set(Array.from(panel.querySelectorAll('details[open]')).map(d=>d.dataset.run));
    const latest=rows.at(-1),model=currentModel?.model_round==null?'当前模型':`V${currentModel.model_round}`;
    const phases=report.active.filter(p=>!p.stale);
    const active=[['running','starting','loading_model'].includes(live.status)&&live.continuous,phases.some(p=>p.kind==='self-train'),phases.some(p=>p.kind==='evaluate'),phases.some(p=>p.kind==='deploy')];
    panel.innerHTML=`<div class="iteration-title"><h3>RSI 每轮变化</h3><span class="pill">${model}${latest?` · 第 ${latest.round} 轮学习`:''}</span></div><div class="iteration-flow">${['采集','训练','评测','更新'].map((label,i)=>`<span class="${active[i]?'iteration-active':''}">${label}</span>`).join('')}</div><p class="iteration-help">每轮配对比较父模型与候选模型；通过发布门槛后切换版本。</p><div class="iteration-versions" aria-label="运行模型版本与采集成功率">${report.collection.map(c=>`<div class="${c.model_version===live.version?'version-current':''}" title="${escape(c.model_version)}"><b>${c.model_round==null?'指定模型':`V${c.model_round}`}</b><span>${pct(c.success_rate)}</span><small>${c.successes}/${c.episodes} 局成功</small></div>`).join('')}</div><p class="iteration-help">上方为随机探索采集成功率；每轮发布依据下方独立配对评测。</p>${rows.length?rows.slice().reverse().map(r=>{
      const published=!!r.deployment,cancelled=['cancelled','failed'].includes(r.evaluation_status)||['cancelled','failed'].includes(r.status);
      const status=published?`已发布 V${r.model_round}`:r.gate_passed===false?'未通过 · 保留当前模型':cancelled?'已中止 · 未发布':r.evaluation_status==='running'?'评测中':r.status==='running'?'训练中':r.gate_passed===true?'通过 · 等待更新':'等待评测';
      const partial=r.evaluation&&r.evaluation_status!=='completed';
      const gain=r.paired_gain==null?'等待最终结果':`${r.paired_gain>=0?'+':''}${(r.paired_gain*100).toFixed(1)} pp`;
      return `<details class="iteration-card ${published?'iteration-published':r.gate_passed===false||cancelled?'iteration-retained':''}" data-run="${escape(r.training)}" ${r===latest||expanded.has(r.training)?'open':''}><summary><b>第 ${r.round} 轮</b><span>${status}</span></summary><div class="iteration-metrics"><div><span>训练 loss · 首次 → 末次</span><b>${loss(r.loss_first)} → ${loss(r.loss_last)}</b><small>${r.logged_updates} / ${r.updates??'—'} 次更新${r.selected_step==null?'':` · 选中第 ${r.selected_step} 步`}</small></div><div><span>任务成功率 · 父模型 → 候选${partial?'（临时）':''}</span><b>${pct(r.parent_success_rate)} → ${pct(r.candidate_success_rate)}</b><small>${r.evaluation?`${r.evaluated_pairs} / ${r.total_pairs} 配对场景`:'训练后独立评测'} · ${gain}</small></div><div class="iteration-data"><span>真实成功轨迹</span><b>${r.successful_episodes??'—'} 局 / ${r.training_samples??'—'} 动作样本</b></div></div></details>`;
    }).join(''):'<p>等待第一轮训练。采集成功轨迹满额后自动开始。</p>'}<a class="metric-link" href="/training#round-comparison">查看全部轮次、回放与成功率曲线 →</a>`;
  }
  function renderState() {
    const phases = report?.active || [];
    const current = phases.filter(p=>!p.stale);
    const training = current.some(p=>p.kind==='self-train' || p.stage==='success_replay_training');
    const evaluation = current.some(p=>p.kind==='evaluate' || p.stage==='independent_evaluation');
    const deploying = current.some(p=>p.kind==='deploy' || p.stage==='deploying');
    const collecting = ['running','starting','loading_model'].includes(live.status) && live.continuous;
    const round = report?.training.at(-1);
    const currentModel = report?.collection.find(c=>c.model_version===live.version);
    if(node('evolution-round')) {
      const modelRound=currentModel?.model_round??report?.training.find(r=>r.deployment===live.version)?.model_round??(live.version?.includes('v1.0-0.8b')?0:null);
      const version=live.policy==='teacher'?'规则对照':modelRound!=null?`运行模型 V${modelRound}`:live.version?'指定运行模型':'运行模型待确认';
      node('evolution-round').textContent=version+(round?` · 第 ${round.round} 轮学习`:' · 准备首次学习');
      const number=live.model_episode_index??(currentModel?currentModel.episodes+(['running','starting','loading_model','paused'].includes(live.status)?1:0):null);
      node('evolution-episode').textContent=number?`${live.policy==='teacher'?'本次':'本版本'}第 ${number} 局`:'等待第一局任务';
    }
    renderIteration(currentModel);
    renderSuccessBars();
    const entries = [
      ['数据采集',collecting?'持续采集中':live.status==='paused'?'已暂停':'等待采集',collecting,`${currentModel?`${currentModel.successes}/${currentModel.episodes} 成功 · ${pct(currentModel.success_rate)}`:'等待完整轨迹'} · ${live.sampling||'机械臂仿真'}`],
      ['模型训练',training?'模型训练中':'等待下一轮',training,training&&round?`第 ${round.round} 轮 · 更新 ${round.loss_curve.at(-1)?.step||0}/${round.updates} · loss ${loss(round.loss_last)}`:'成功轨迹达到阈值后启动'],
      ['独立评测',evaluation?'配对评测中':'等待候选模型',evaluation,evaluation&&round?`${round.evaluated_pairs}/${round.total_pairs} 对 · 当前候选 ${pct(round.candidate_success_rate)}`:'同场景比较候选与父模型'],
      ['模型部署',deploying?'验证并部署中':'当前运行版本',deploying,live.version||'评测通过后更新'],
    ];
    const html = entries.map(([title,state,active,detail],i)=>`<article class="rsi-phase ${active?'is-active':''}"><span class="phase-number">0${i+1}</span><h3>${title}</h3><b>${state}</b><small title="${escape(detail)}">${escape(detail)}</small></article>`).join('');
    if (node('rsi-phases') && html !== phaseHTML) { node('rsi-phases').innerHTML = html; phaseHTML = html; }
    if (node('rsi-current')) node('rsi-current').textContent = deploying?'DEPLOYING':evaluation?'EVALUATING':training?'TRAINING':collecting?'COLLECTING':'STANDBY';
    if(node('simulation-success')){
      const label=currentModel?.model_round==null?'当前模型':`V${currentModel.model_round}`;
      const last=currentModel?.curve.at(-1);
      const html=`<div><span>${label} · 完整任务成功率</span><strong>${currentModel?pct(currentModel.success_rate):'—'}</strong><small>${currentModel?`${currentModel.successes} 成功 / ${currentModel.episodes} 局 · ${currentModel.failures} 失败`:'等待本版本完成第一局任务'}</small></div><div class="recent-episode"><span>最近一局</span><b class="${last?.success?'green':'episode-failure'}">${last?`场景 ${escape(last.seed)} · ${last.success?'✓ 完成':'✗ 未完成'}`:'尚无结果'}</b><small>每局完成后持久保存，训练页查看逐局记录与轮次对比</small></div>`;
      if(html!==successHTML){node('simulation-success').innerHTML=html;successHTML=html;}
    }
    if (node('model-round-rate')) {
      node('model-round-rate').textContent = currentModel ? pct(currentModel.success_rate) : '—';
      node('model-round-count').textContent = currentModel ? `${currentModel.episodes} 局 · ${currentModel.successes} 成功 · ${currentModel.failures} 失败` : '等待当前模型完成采集';
    }
  }
  window.RSIMonitor = {updateSimulation(value) { live=value; renderState(); }};
  async function poll() {
    if (polling) return;
    polling=true;
    try {
      const response=await fetch('/api/rounds',{cache:'no-store'});
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      report=await response.json(); renderReport(); renderState();
      window.dispatchEvent(new CustomEvent('rsi-rounds',{detail:report}));
    } catch (error) {
      if(node('round-updated')) node('round-updated').textContent=`记录暂时不可用 · ${error.message}`;
      window.dispatchEvent(new Event('rsi-rounds-unavailable'));
    }
    finally { polling=false; }
  }
  poll(); setInterval(poll,5000);
})();
