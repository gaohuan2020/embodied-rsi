'use strict';
const $ = id => document.getElementById(id);
let selected = null, events = [], cursor = 0, lastOverview = null, fetching = false;
let dataId=null, trajectoryId=null, trajectoryOffset=0, selectionPinned=false;
const fmt = (n, digits=3) => Number.isFinite(n) ? n.toFixed(digits) : '—';
const pct = n => Number.isFinite(n) ? `${(n*100).toFixed(1)}%` : '—';
const escape = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(path, options) { const r = await fetch(path, options); const b = await r.json(); if(!r.ok) throw new Error(typeof b.detail === 'string' ? b.detail : JSON.stringify(b.detail)); return b; }
function chart(id, series, fixedMax=null) {
  const container=$(id), values=series.flatMap(s=>s.points.map(p=>p[1])).filter(Number.isFinite);
  if(!values.length){container.innerHTML='<div class="empty">暂无训练指标</div>';return;}
  const w=Math.max(260,container.clientWidth),h=Math.max(115,container.clientHeight||180),l=40,t=10,b=22, xmax=Math.max(id==='version-chart'?2:1,...series.flatMap(s=>s.points.map(p=>p[0]))), ymin=fixedMax?0:Math.min(0,...values)*1.1,ymax=fixedMax||Math.max(.001,...values)*1.1;
  const x=v=>l+v/xmax*(w-l-8),y=v=>h-b-(v-ymin)/(ymax-ymin)*(h-t-b);
  let svg=`<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" role="img" aria-label="训练指标曲线">`;
  for(let i=0;i<4;i++){const v=ymin+(ymax-ymin)*i/3;svg+=`<line x1="${l}" y1="${y(v)}" x2="${w}" y2="${y(v)}"/><text x="0" y="${y(v)+3}">${v.toFixed(ymax>10?0:2)}</text>`;}
  for(const [seriesIndex,s] of series.entries()){const points=s.points.filter(p=>Number.isFinite(p[1]));svg+=`<polyline fill="none" stroke="${s.color}" stroke-width="2" vector-effect="non-scaling-stroke" points="${points.map(p=>`${x(p[0])},${y(p[1])}`).join(' ')}"/>`;if(id==='study-chart')for(const point of points)svg+=`<circle cx="${x(point[0])}" cy="${y(point[1])}" r="${seriesIndex?2:4}" fill="${s.color}"/>`;if(points.length===1)svg+=`<circle cx="${x(points[0][0])}" cy="${y(points[0][1])}" r="3" fill="${s.color}"/>`;}
  const chosen=lastOverview?.runs.find(r=>r.id===selected)?.summary?.selected_step;if(id==='task-chart'&&Number.isFinite(chosen))svg+=`<line x1="${x(chosen)}" x2="${x(chosen)}" y1="10" y2="${h-b}" style="stroke:#73e3ba;stroke-dasharray:4 4"/><text x="${Math.min(x(chosen)+5,w-80)}" y="20">选中第 ${chosen} 步</text>`;
  svg+=`<text x="${l}" y="${h-3}">0</text><text x="${w-35}" y="${h-3}">${xmax}</text></svg>`;container.innerHTML=svg;
}
function metricSeries(rows,key,color){return {color,points:rows.filter(r=>Number.isFinite(r[key])).map(r=>[r.step,r[key]])};}
function renderEvents(){
  const metrics=events.filter(e=>e.kind==='metric').map(e=>e.payload), last=metrics.at(-1)||{};
  const latest=key=>[...metrics].reverse().find(m=>Number.isFinite(m[key]))?.[key];
  $('loss').textContent=fmt(latest('policy_loss')??latest('loss'));$('accuracy').textContent=pct(latest('val_accuracy'));$('step-label').textContent=`STEP ${last.step??'—'}`;
  $('loss-sub').textContent=`验证损失 ${fmt(latest('val_loss'))}`;$('throughput').textContent=`${fmt(latest('steps_per_second'),1)} steps/s`;
  chart('loss-chart',[metricSeries(metrics,metrics.some(m=>Number.isFinite(m.loss))?'loss':'policy_loss','#73e3ba'),metricSeries(metrics,'val_loss','#7aaaff')]);
  chart('task-chart',[metricSeries(metrics,'train_success_rate','#73e3ba'),metricSeries(metrics,'dev_success_rate','#7aaaff')],1);
  const run=lastOverview?.runs.find(r=>r.id===selected);$('training-sample-count').textContent=latest('training_samples')??run?.summary?.training_samples??'—';$('dev-best').textContent=pct(run?.summary?.success_rate??metrics.filter(m=>Number.isFinite(m.dev_success_rate)).reduce((best,m)=>Math.max(best,m.dev_success_rate),-Infinity));$('loss-current').textContent=fmt(latest('loss')??latest('policy_loss'));$('best-step').textContent=Number.isFinite(run?.summary?.selected_step)?`第 ${run.summary.selected_step} 步`:'训练中';$('loss-label').textContent=run?.kind==='initialize'||run?.kind==='train'?'示范初始化':run?.kind==='task-train'?'整局策略梯度':'成功轨迹重放';
  chart('grad-chart',[metricSeries(metrics,'grad_norm','#73e3ba')]);chart('acc-chart',[metricSeries(metrics,'val_accuracy','#7aaaff')],1);
  const alerts=events.filter(e=>e.kind==='alert');$('alert-count').textContent=alerts.length;
  $('alerts').className=alerts.length?'alerts':'alerts empty';$('alerts').innerHTML=alerts.length?alerts.slice(-8).reverse().map(e=>`<div class="alert-item ${escape(e.payload.severity)}"><b>${escape(e.payload.code)}</b><br>${escape(e.payload.message)}</div>`).join(''):'当前实验暂无异常记录';
}
function renderOverview(o){
  lastOverview=o;const runs=o.runs;const activeTraining=runs.find(r=>r.status==='running'&&['self-train','initialize','task-train','train'].includes(r.kind));if(!selectionPinned&&activeTraining&&selected!==activeTraining.id){selected=activeTraining.id;events=[];cursor=0;}
  if(!selected&&runs.length)selected=(runs.find(r=>r.kind==='self-train')||runs.find(r=>r.kind==='initialize')||runs.find(r=>r.kind==='task-train'&&r.backend.includes('rsi'))||runs.find(r=>r.kind==='train'&&r.backend.includes('rsi'))||runs.find(r=>r.kind==='task-train')||runs[0]).id;
  $('run-select').innerHTML=runs.filter(r=>['self-train','initialize','task-train','train'].includes(r.kind)||r.id===selected).map(r=>`<option value="${escape(r.id)}" ${selected===r.id?'selected':''}>${escape(r.id)}</option>`).join('')||'<option>暂无实验</option>';
  $('run-description').textContent=selected?`${runs.find(r=>r.id===selected)?.backend||''} · ${selected}`:'选择实验查看训练过程';
  $('episodes').textContent=runs.filter(r=>['collect','explore'].includes(r.kind)).reduce((sum,r)=>sum+(r.summary.episodes||0),0);
  $('run-count').textContent=`${runs.length} RUNS`;
  $('runs').innerHTML=runs.map(r=>{let result=r.kind==='task-train'?`任务完成率 ${pct(r.summary.success_rate)} · 最佳更新 ${r.summary.selected_step??'—'}`:r.kind==='simulation'?`${r.summary.success?'任务完成':'任务未完成'} · ${r.summary.steps??'—'} 动作`:r.kind==='deploy'?`部署 ${r.summary.status||r.status}`:r.kind==='cycle'?`自动闭环 · ${r.summary.rounds?.at(-1)?.status||r.status}`:r.kind==='train'?`示范预热 · 单步准确率 ${pct(r.summary.final_dev?.accuracy)}`:r.kind==='collect'?`${r.summary.episodes||'—'} 局 · 成功率 ${pct(r.summary.success_rate)}`:r.kind==='evaluate'?`成功率 ${pct(r.summary.success_rate)} · ${r.summary.gate?.passed?'达标':'未晋级'}`:'—';if(['self-train','initialize'].includes(r.kind))result=`${r.kind==='initialize'?'示范初始化':'成功轨迹重放'} · 完成率 ${pct(r.summary.success_rate)} · ${r.summary.training_samples??'—'} 样本`;if(r.kind==='explore')result=`模型探索 · ${r.summary.successful_episodes??'—'}/${r.summary.episodes??'—'} 完成 · ${pct(r.summary.success_rate)}`;if(r.kind==='study')result=`搬运入盘多轮测试 · ${r.summary.conclusion||r.status}`;if(r.kind==='self-improve')result=`RSI 自我提升 · ${r.summary.rounds?.at(-1)?.status||r.status}`;const status=r.stale?'stale':r.status;return `<tr data-id="${escape(r.id)}"><td>${escape(r.id)}</td><td>${escape(r.backend)}</td><td><span class="status ${escape(status)}">${escape(status)}</span></td><td>${escape(result)}</td><td>${new Date(r.created*1000).toLocaleString()}</td></tr>`;}).join('')||'<tr><td colspan="5">暂无实验。点击「启动实验」开始采集与训练。</td></tr>';
  document.querySelectorAll('tr[data-id]').forEach(el=>el.onclick=()=>selectRun(el.dataset.id));
  renderStudy(runs);const r=o.resources,g=r.gpus?.[0];$('gpu-name').textContent=g?.name||'无 GPU 读数';$('gpu-util').textContent=Number.isFinite(g?.util_percent)?`${g.util_percent}%`:'—';$('gpu-bar').value=g?.util_percent||0;
  $('ram-value').textContent=`${fmt(r.ram_used_gib,1)} / ${fmt(r.ram_total_gib,1)} GiB`;$('ram-bar').value=r.ram_percent||0;
  $('cpu-value').textContent=`${fmt(r.cpu_percent,0)}%`;$('cpu-bar').value=r.cpu_percent||0;$('gpu-temp').textContent=Number.isFinite(g?.temperature_c)?`${g.temperature_c} °C`:'—';
  $('gpu-memory').textContent=Number.isFinite(g?.memory_used_mib)?`${fmt(g.memory_used_mib/1024,1)} GiB`:'设备未报告（统一内存）';
  $('champion').textContent=o.champion?.checkpoint?.split('/').slice(-2,-1)[0]||'尚未晋级';$('dataset-count').textContent=o.datasets.length;
  updateOptions('dataset-select',o.datasets.map(d=>({id:d.id,label:`${d.id} · ${d.cases.train} train`})), '选择数据集');
  updateOptions('checkpoint-select',runs.filter(r=>['train','task-train','self-train','initialize'].includes(r.kind)&&r.status==='completed').map(r=>({id:r.id,label:`${r.id} · ${r.backend}`})),'自动接续当前部署模型／初始 RSI 模型');
  updateOptions('data-select',o.datasets.map(d=>({id:d.id,label:`${d.id} · ${d.objective==='successful_episode_replay'?'模型成功轨迹':'示范初始化'} · ${d.cases.train} 样本`})),'选择数据集');
  if(!$('data-select').value&&o.datasets.length)$('data-select').value=(o.datasets.filter(d=>d.objective==='successful_episode_replay').at(-1)||o.datasets.at(-1)).id;
  if($('data-select').value!==dataId)renderData();
  const exploration=runs.filter(r=>r.kind==='explore'&&r.config.mode!=='existing_model_replay'&&(r.status==='running'||r.summary.episodes>0));updateOptions('trajectory-select',exploration.map(r=>({id:r.id,label:`${r.id} · ${r.backend} · ${r.summary.successful_episodes??'—'}/${r.summary.episodes??'—'} 成功`})),'选择探索批次');
  if(!$('trajectory-select').value&&exploration.length)$('trajectory-select').value=exploration[0].id;
  if($('trajectory-select').value!==trajectoryId){trajectoryOffset=0;renderTrajectories();}
  $('deployed-version').textContent=o.rsi_champion?.version||'RSI 尚未通过发布评测';
  $('deployed-at').textContent=o.rsi_champion?.deployed_at?`${new Date(o.rsi_champion.deployed_at*1000).toLocaleString()} · RSI 加载验证通过`:o.champion?`参考模型 ${o.champion.version||o.champion.sha256.slice(0,12)} 保留`:'等待自动部署';
  const deployments=o.deployments||[];$('deployment-history').className=deployments.length?'':'empty';$('deployment-history').innerHTML=deployments.length?deployments.map(r=>`<div class="history-row"><b>${escape(r.id)}</b><span>${escape(r.status)} · ${pct(r.summary.report?.success_rate)} 任务完成率</span><span>${new Date(r.created*1000).toLocaleString()}</span></div>`).join(''):'暂无自动部署记录';
  const active=o.jobs.find(j=>j.status==='running'), stale=runs.find(r=>r.id===selected)?.stale;
  $('notice').hidden=!active&&!stale;$('notice').innerHTML=active?`后台任务 ${escape(active.action)} 正在运行。<button id="stop-job">停止任务</button>`:stale?'当前实验超过 60 秒没有心跳，请检查训练进程。':'';
  if(active)$('stop-job').onclick=async()=>{await api(`/api/jobs/${active.id}/stop`,{method:'POST'});};
  let report=runs.find(r=>r.kind==='evaluate'&&r.status==='completed')?.summary;
  const rsiReports=runs.filter(r=>r.kind==='evaluate'&&r.status==='completed'&&(r.summary.model_backend==='rsi-jev'||runs.find(t=>r.summary.candidate_checkpoint?.includes(t.id)&&t.backend.includes('rsi')))).reverse();
  chart('version-chart',[{color:'#7aaaff',points:rsiReports.map((r,i)=>[i+1,r.summary.success_rate])},{color:'#8b9cac',points:rsiReports.map((r,i)=>[i+1,r.summary.champion_success_rate])}],1);
  if(rsiReports.length)report=rsiReports.at(-1).summary;$('success-title').textContent=rsiReports.length?'RSI 独立任务完成率':'参考实验完成率（非 RSI）';$('rsi-progress-label').textContent=rsiReports.length?`最新同批 ${pct(rsiReports.at(-1).summary.champion_success_rate)} → ${pct(rsiReports.at(-1).summary.success_rate)}`:'尚无 RSI 独立任务评测 · 参考模型单独报告';
  $('success-rate').textContent=pct(report?.success_rate);$('gain').textContent=Number.isFinite(report?.paired_gain)?`${report.paired_gain>=0?'+':''}${(report.paired_gain*100).toFixed(1)} pp`:'—';$('success-sub').textContent=report?`${report.pairs} 个独立场景 · ${report.gate.passed?'已通过发布门槛':'候选未发布'}`:'完整任务 · 三类机械臂场景';
  if(report){$('gate-badge').textContent=report.gate.passed?'通过晋级标准':'保留当前版本';$('evaluation-body').className='';$('evaluation-body').innerHTML=`<div class="eval-comparison"><div><span>父模型</span><strong>${pct(report.champion_success_rate)}</strong></div><span class="eval-arrow">→</span><div><span>候选模型</span><strong>${pct(report.success_rate)}</strong></div></div><p class="eval-ci">提升 95% 区间：${(report.gain_ci95[0]*100).toFixed(1)} ～ ${(report.gain_ci95[1]*100).toFixed(1)} 个百分点</p><div class="gate-checks">${Object.entries(report.gate.checks).map(([k,v])=>`<span class="${v?'passed':'rejected'}">${v?'✓':'✗'} ${escape(({enough_pairs:'评测数量',task_coverage:'评测任务覆盖',success_gain:'成功率提升',confidence_interval:'提升置信区间',no_task_regression:'各任务无退化',no_risk_increase:'风险不增加'})[k]||k)}</span>`).join('')}</div>`;}
}
function updateOptions(id,items,placeholder){const el=$(id),value=el.value;el.innerHTML=`<option value="">${placeholder}</option>`+items.map(i=>`<option value="${escape(i.id)}">${escape(i.label)}</option>`).join('');if(items.some(i=>i.id===value))el.value=value;}
async function selectRun(id){selectionPinned=true;selected=id;events=[];cursor=0;renderEvents();await poll();}
async function renderData(){dataId=$('data-select').value;if(!dataId)return;const manifest=lastOverview.datasets.find(d=>d.id===dataId);$('data-stats').innerHTML=[['尝试总局数',manifest.episodes_total??'—'],['成功轨迹',manifest.successful_episodes??'示范集'],['训练动作样本',manifest.cases.train],['失败轨迹',manifest.failed_episodes??'—']].map(([k,v])=>`<div>${k}<b>${escape(v)}</b></div>`).join('');try{const data=await api(`/api/datasets/${dataId}/samples?limit=8`);$('data-samples').className=data.samples.length?'':'empty';$('data-samples').innerHTML=data.samples.length?data.samples.map(c=>`<details class="sample-record"><summary>${escape(c.provenance.task||c.source)} · ${escape(c.provenance.group)} · ${escape(c.provenance.choice||c.provenance.label_source)} · ${c.provenance.episode_success?'模型完成的任务':'初始化样本'}</summary><pre>${escape(JSON.stringify({...c,state:JSON.parse(c.state)},null,2))}</pre></details>`).join(''):'当前没有成功轨迹，训练尚未开始。';}catch(e){$('data-samples').textContent=e.message;}}
async function renderTrajectories(){trajectoryId=$('trajectory-select').value;if(!trajectoryId)return;try{const data=await api(`/api/runs/${trajectoryId}/episodes?outcome=${$('trajectory-outcome').value}&offset=${trajectoryOffset}&limit=5`);$('trajectory-page').textContent=`${Math.min(trajectoryOffset+1,data.total)}–${Math.min(trajectoryOffset+5,data.total)} / ${data.total}`;$('trajectory-prev').disabled=trajectoryOffset===0;$('trajectory-next').disabled=trajectoryOffset+5>=data.total;$('trajectory-samples').className=data.episodes.length?'':'empty';$('trajectory-samples').innerHTML=data.episodes.length?data.episodes.map(e=>`<details class="sample-record"><summary>${e.success?'✓ 完成':'✗ 失败'} · ${escape(e.task)}:${e.seed} · ${e.steps} 动作 · ${escape(e.failure||'稳定完成')} · ${escape(e.source_policy||'')}</summary><pre>${escape(JSON.stringify(e,null,2))}</pre></details>`).join(''):'此筛选条件下暂无轨迹。';}catch(e){$('trajectory-samples').textContent=e.message;}}
function renderPipeline(run,rows=[]){const stages=rows.filter(e=>e.kind==='stage'),last=stages.at(-1)?.payload,self=run?.kind==='self-improve',keys=self?['exploring','success_replay_training','independent_evaluation','deploying']:['initialization','task_training','independent_evaluation','deploying'],labels=self?['① RSI 探索与轨迹收集','② 成功轨迹重放训练','③ 完整任务成功率评测','④ 提升后自动部署']:['① 示范初始化（可跳过）','② 完整任务奖励训练','③ 独立任务完成率评测','④ 达标后自动部署'];let index=keys.indexOf(last?.stage);if(last?.stage==='completed'||last?.stage==='retained_current_model')index=4;$('pipeline').innerHTML=keys.map((k,i)=>`<div class="pipeline-stage ${i===3&&last?.status==='retained_current_model'?'blocked':i>=1&&last?.status==='no_success_data'?'blocked':i===index?'current':i<index?'done':''}">${i===3&&last?.status==='retained_current_model'?'④ 未发布 · 保留当前模型':labels[i]}</div>`).join('');$('pipeline-message').textContent=run?`${run.backend.toUpperCase()} · ${run.status} · ${last?.status==='no_success_data'?'没有成功轨迹，跳过训练并保留失败数据':last?.status==='retained_current_model'||last?.stage==='retained_current_model'?'评测未证明提升，继续使用当前模型':last?.status==='deployed'?'优化模型已自动部署':({exploring:'采集模型轨迹',success_replay_training:'成功轨迹训练中',independent_evaluation:'独立随机场景评测中',deploying:'正在验证并部署'})[last?.stage]||last?.stage||'等待开始'}`:'启动 RSI 自我提升，收集模型完成的任务轨迹后训练。';}
async function poll(){if(fetching)return;fetching=true;try{const o=await api('/api/overview');renderOverview(o);if(selected){const data=await api(`/api/runs/${selected}/events?after=${cursor}`);events.push(...data.events);if(data.events.length)cursor=data.events.at(-1).seq;renderEvents();}const cycle=o.runs.find(r=>['self-improve','cycle'].includes(r.kind));renderPipeline(cycle,cycle?(await api(`/api/runs/${cycle.id}/events`)).events:[]);const activeExplore=o.runs.find(r=>r.id===trajectoryId);if(activeExplore?.status==='running')await renderTrajectories();$('connection').textContent='● LIVE';}catch(e){$('connection').textContent='连接中断';}finally{fetching=false;}}
$('run-select').onchange=e=>selectRun(e.target.value);
$('open-job').onclick=()=>{$('job-error').textContent='';$('job-dialog').showModal();};$('close-job').onclick=()=>$('job-dialog').close();
$('action').onchange=()=>{const action=$('action').value;document.querySelector('[name="rounds"]').value=action==='transfer-study'?3:1;$('seed-start').value=action==='evaluate'?10000:0;$('backend').disabled=action==='collect';if($('backend').disabled)$('backend').value='compact';$('dataset-select').disabled=!['train','evaluate','initialize-rsi'].includes(action);$('checkpoint-select').disabled=action==='transfer-study';if(action==='transfer-study')$('checkpoint-select').value='';document.querySelector('[name="steps"]').value=action==='train'?500:action==='initialize-rsi'?10:['self-improve','transfer-study'].includes(action)?100:6;};
$('action').onchange();
$('data-select').onchange=renderData;$('trajectory-select').onchange=()=>{trajectoryOffset=0;renderTrajectories();};$('trajectory-outcome').onchange=()=>{trajectoryOffset=0;renderTrajectories();};$('trajectory-prev').onclick=()=>{trajectoryOffset=Math.max(0,trajectoryOffset-5);renderTrajectories();};$('trajectory-next').onclick=()=>{trajectoryOffset+=5;renderTrajectories();};
$('job-form').onsubmit=async e=>{e.preventDefault();const form=new FormData(e.target),body=Object.fromEntries(form);body.backend=$('backend').value;body.tasks=[$('job-task').value];for(const key of ['steps','episodes','explore_episodes','rounds','seed_start','audit_episodes'])body[key]=Number(body[key]);body.dataset=body.dataset||null;body.checkpoint=body.checkpoint||null;try{await api('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});$('job-dialog').close();selectionPinned=false;await poll();}catch(err){$('job-error').textContent=err.message;}};
renderEvents();poll();setInterval(poll,2000);

let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{if(lastOverview)renderOverview(lastOverview);renderEvents();},120);});

let studyCursor=0,studyEvents=[],studyId=null;
function renderStudy(runs) {
  const run=runs.find(r=>r.kind==='study');
  $('study-panel').hidden=!run;
  if(!run)return;
  $('study-status').textContent=run.status==='completed'?'测试完成':run.status==='failed'?'测试失败':'测试运行中';
  $('study-protocol').textContent=`仅搬运入盘 · ${run.config.rounds} 轮 · 每轮 ${run.config.explore_episodes} 局采集 / ${run.config.release_episodes} 对新场景发布评测 · 最后 ${run.config.audit_episodes} 个保留场景`;
  const points=run.summary.audit||[],rounds=run.summary.rounds||[];
  chart('study-chart',[
    {color:'#7aaaff',points:points.map((p,i)=>[i,p.success_rate])},
    {color:'#73e3ba',points:(run.summary.deployed_audit_rates||[]).map((p,i)=>[i,p])}
  ],1);
  $('study-rounds').innerHTML=rounds.length?`<table><thead><tr><th>学习轮</th><th>本轮自主成功</th><th>累计成功轨迹</th><th>训练样本</th><th>选中更新步</th><th>新场景候选 / 当前</th><th>发布结果</th></tr></thead><tbody>${rounds.map(r=>{
    const collection=runs.find(x=>x.id===r.collection),training=runs.find(x=>x.id===r.training);
    return `<tr><td>V${r.round}</td><td>${collection?.summary.successful_episodes??'—'} / ${collection?.summary.episodes??'—'}</td><td>${r.successful_episodes??'—'}</td><td>${training?.summary.training_samples??'—'}</td><td>${training?.summary.selected_step??'—'}</td><td>${pct(r.success_rate)} / ${pct(r.parent_success_rate)}</td><td>${r.status==='deployed'?'已发布':r.status==='no_success_data'?'无成功数据 · 未训练':'未证明提升 · 保留当前'}</td></tr>`;
  }).join('')}</tbody></table>`:'';
  $('study-conclusion').textContent=run.summary.conclusion?`${run.summary.conclusion} · 真正发布 ${run.summary.updated_models} 次 · 蓝色为选中候选、绿色为实际运行版本。选中第 0 步表示训练未改善开发集完成率，保留训练前权重。`:'不预设成功率上升；候选退化、持平和未通过发布均如实保留。';
  if(studyId!==run.id){studyId=run.id;studyCursor=0;studyEvents=[];}
  api(`/api/runs/${run.id}/events?after=${studyCursor}`).then(data=>{
    studyEvents.push(...data.events);
    if(data.events.length)studyCursor=data.events.at(-1).seq;
    const audit=studyEvents.filter(e=>e.kind==='audit_progress').at(-1);
    $('study-progress').textContent=run.status==='completed'?`相同保留场景完成率：${points.map(p=>`${p.label} ${(p.success_rate*100).toFixed(1)}% (${p.successful_episodes}/${p.episodes})`).join(' → ')}`:audit?`最终保留场景审计 ${audit.payload.model}：${audit.payload.completed}/${audit.payload.total} · ${audit.payload.successful_episodes} 成功`:'正在采集、训练和独立发布评测；所有轮次完成后才开启固定场景审计。';
  }).catch(()=>{});
}
