'use strict';
const $ = id => document.getElementById(id);
let selected = null, events = [], cursor = 0, lastOverview = null, fetching = false;
const fmt = (n, digits=3) => Number.isFinite(n) ? n.toFixed(digits) : '—';
const pct = n => Number.isFinite(n) ? `${(n*100).toFixed(1)}%` : '—';
const escape = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
async function api(path, options) { const r = await fetch(path, options); const b = await r.json(); if(!r.ok) throw new Error(typeof b.detail === 'string' ? b.detail : JSON.stringify(b.detail)); return b; }
function chart(id, series, fixedMax=null) {
  const container=$(id), values=series.flatMap(s=>s.points.map(p=>p[1])).filter(Number.isFinite);
  if(!values.length){container.innerHTML='<div class="empty">暂无训练指标</div>';return;}
  const w=Math.max(260,container.clientWidth),h=id==='loss-chart'?220:115,l=40,t=10,b=22, xmax=Math.max(1,...series.flatMap(s=>s.points.map(p=>p[0]))), ymax=fixedMax||Math.max(.001,...values)*1.1;
  const x=v=>l+v/xmax*(w-l-8),y=v=>h-b-v/ymax*(h-t-b);
  let svg=`<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" role="img" aria-label="训练指标曲线">`;
  for(let i=0;i<4;i++){const v=ymax*i/3;svg+=`<line x1="${l}" y1="${y(v)}" x2="${w}" y2="${y(v)}"/><text x="0" y="${y(v)+3}">${v.toFixed(ymax>10?0:2)}</text>`;}
  for(const s of series){const points=s.points.filter(p=>Number.isFinite(p[1]));svg+=`<polyline fill="none" stroke="${s.color}" stroke-width="2" vector-effect="non-scaling-stroke" points="${points.map(p=>`${x(p[0])},${y(p[1])}`).join(' ')}"/>`;if(points.length===1)svg+=`<circle cx="${x(points[0][0])}" cy="${y(points[0][1])}" r="3" fill="${s.color}"/>`;}
  svg+=`<text x="${l}" y="${h-3}">0</text><text x="${w-35}" y="${h-3}">${xmax}</text></svg>`;container.innerHTML=svg;
}
function metricSeries(rows,key,color){return {color,points:rows.filter(r=>Number.isFinite(r[key])).map(r=>[r.step,r[key]])};}
function renderEvents(){
  const metrics=events.filter(e=>e.kind==='metric').map(e=>e.payload), last=metrics.at(-1)||{};
  const latest=key=>[...metrics].reverse().find(m=>Number.isFinite(m[key]))?.[key];
  $('loss').textContent=fmt(latest('loss'));$('accuracy').textContent=pct(latest('val_accuracy'));$('step-label').textContent=`STEP ${last.step??'—'}`;
  $('loss-sub').textContent=`验证损失 ${fmt(latest('val_loss'))}`;$('throughput').textContent=`${fmt(latest('steps_per_second'),1)} steps/s`;
  chart('loss-chart',[metricSeries(metrics,'loss','#73e3ba'),metricSeries(metrics,'val_loss','#7aaaff')]);
  chart('grad-chart',[metricSeries(metrics,'grad_norm','#73e3ba')]);chart('acc-chart',[metricSeries(metrics,'val_accuracy','#7aaaff')],1);
  const alerts=events.filter(e=>e.kind==='alert');$('alert-count').textContent=alerts.length;
  $('alerts').className=alerts.length?'alerts':'alerts empty';$('alerts').innerHTML=alerts.length?alerts.slice(-8).reverse().map(e=>`<div class="alert-item ${escape(e.payload.severity)}"><b>${escape(e.payload.code)}</b><br>${escape(e.payload.message)}</div>`).join(''):'当前实验暂无异常记录';
}
function renderOverview(o){
  lastOverview=o;const runs=o.runs;
  if(!selected&&runs.length)selected=runs[0].id;
  $('run-select').innerHTML=runs.map(r=>`<option value="${escape(r.id)}" ${selected===r.id?'selected':''}>${escape(r.id)}</option>`).join('')||'<option>暂无实验</option>';
  $('run-description').textContent=selected?`${runs.find(r=>r.id===selected)?.backend||''} · ${selected}`:'选择实验查看训练过程';
  $('episodes').textContent=runs.filter(r=>r.kind==='collect').reduce((sum,r)=>sum+(r.summary.episodes||0),0);
  $('run-count').textContent=`${runs.length} RUNS`;
  $('runs').innerHTML=runs.map(r=>{let result=r.kind==='train'?`验证准确率 ${pct(r.summary.final_dev?.accuracy)}`:r.kind==='collect'?`${r.summary.episodes||'—'} 局 · 成功率 ${pct(r.summary.success_rate)}`:r.kind==='evaluate'?`成功率 ${pct(r.summary.success_rate)} · ${r.summary.gate?.passed?'达标':'未晋级'}`:'—';const status=r.stale?'stale':r.status;return `<tr data-id="${escape(r.id)}"><td>${escape(r.id)}</td><td>${escape(r.backend)}</td><td><span class="status ${escape(status)}">${escape(status)}</span></td><td>${escape(result)}</td><td>${new Date(r.created*1000).toLocaleString()}</td></tr>`;}).join('')||'<tr><td colspan="5">暂无实验。点击「启动实验」开始采集与训练。</td></tr>';
  document.querySelectorAll('tr[data-id]').forEach(el=>el.onclick=()=>selectRun(el.dataset.id));
  const r=o.resources,g=r.gpus?.[0];$('gpu-name').textContent=g?.name||'无 GPU 读数';$('gpu-util').textContent=Number.isFinite(g?.util_percent)?`${g.util_percent}%`:'—';$('gpu-bar').value=g?.util_percent||0;
  $('ram-value').textContent=`${fmt(r.ram_used_gib,1)} / ${fmt(r.ram_total_gib,1)} GiB`;$('ram-bar').value=r.ram_percent||0;
  $('cpu-value').textContent=`${fmt(r.cpu_percent,0)}%`;$('cpu-bar').value=r.cpu_percent||0;$('gpu-temp').textContent=Number.isFinite(g?.temperature_c)?`${g.temperature_c} °C`:'—';
  $('gpu-memory').textContent=Number.isFinite(g?.memory_used_mib)?`${fmt(g.memory_used_mib/1024,1)} GiB`:'设备未报告（统一内存）';
  $('champion').textContent=o.champion?.checkpoint?.split('/').slice(-2,-1)[0]||'尚未晋级';$('dataset-count').textContent=o.datasets.length;
  updateOptions('dataset-select',o.datasets.map(d=>({id:d.id,label:`${d.id} · ${d.cases.train} train`})), '选择数据集');
  updateOptions('checkpoint-select',runs.filter(r=>r.kind==='train'&&r.status==='completed').map(r=>({id:r.id,label:`${r.id} · ${r.backend}`})),'从初始模型开始');
  const active=o.jobs.find(j=>j.status==='running'), stale=runs.find(r=>r.id===selected)?.stale;
  $('notice').hidden=!active&&!stale;$('notice').innerHTML=active?`后台任务 ${escape(active.action)} 正在运行。<button id="stop-job">停止任务</button>`:stale?'当前实验超过 60 秒没有心跳，请检查训练进程。':'';
  if(active)$('stop-job').onclick=async()=>{await api(`/api/jobs/${active.id}/stop`,{method:'POST'});};
  const report=runs.find(r=>r.kind==='evaluate'&&r.status==='completed')?.summary;
  if(report){$('gate-badge').textContent=report.gate.passed?'通过晋级标准':'保留当前版本';$('evaluation-body').className='';$('evaluation-body').innerHTML=`<div class="evaluation-grid"><div><strong>${pct(report.success_rate)}</strong><span>候选模型成功率</span></div><div><strong>${pct(report.paired_gain)}</strong><span>配对提升 · ${report.pairs} 局</span></div><div><strong>${pct(report.gain_ci95[0])} ~ ${pct(report.gain_ci95[1])}</strong><span>95% bootstrap 区间</span></div></div><p>${Object.entries(report.gate.checks).map(([k,v])=>`${v?'✓':'✗'} ${escape(k)}`).join(' &nbsp; ')}</p>`;}
}
function updateOptions(id,items,placeholder){const el=$(id),value=el.value;el.innerHTML=`<option value="">${placeholder}</option>`+items.map(i=>`<option value="${escape(i.id)}">${escape(i.label)}</option>`).join('');if(items.some(i=>i.id===value))el.value=value;}
async function selectRun(id){selected=id;events=[];cursor=0;renderEvents();await poll();}
async function poll(){if(fetching)return;fetching=true;try{const o=await api('/api/overview');renderOverview(o);if(selected){const data=await api(`/api/runs/${selected}/events?after=${cursor}`);events.push(...data.events);if(data.events.length)cursor=data.events.at(-1).seq;renderEvents();}$('connection').textContent='● LIVE';}catch(e){$('connection').textContent='连接中断';}finally{fetching=false;}}
$('run-select').onchange=e=>selectRun(e.target.value);
$('open-job').onclick=()=>{$('job-error').textContent='';$('job-dialog').showModal();};$('close-job').onclick=()=>$('job-dialog').close();
$('action').onchange=()=>{const action=$('action').value;$('seed-start').value=action==='evaluate'?10000:0;$('backend').disabled=action==='cycle'||action==='collect';if($('backend').disabled)$('backend').value='compact';};
$('job-form').onsubmit=async e=>{e.preventDefault();const form=new FormData(e.target),body=Object.fromEntries(form);body.backend=$('backend').value;for(const key of ['steps','episodes','rounds','seed_start'])body[key]=Number(body[key]);body.dataset=body.dataset||null;body.checkpoint=body.checkpoint||null;try{await api('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});$('job-dialog').close();await poll();}catch(err){$('job-error').textContent=err.message;}};
renderEvents();poll();setInterval(poll,2000);
