'use strict';
const $=id=>document.getElementById(id),keys=['surface_area_m2g','void_fraction','lcd','pld'];
const base=(window.MOF_API_BASE||'').replace(/\/$/,'');
let examples=[],mode='example',ready=false,busy=false,stopped=false,current,gas='CO2',counter=0,polling=0;
const results=new Map();
const api=path=>base+path;
function controls(){
  $('example').disabled=busy||!examples.length;$('cif').disabled=busy;
  keys.forEach(k=>$(k).disabled=busy);$('reset').disabled=busy;$('predict').disabled=busy||!ready;
  $('batch').disabled=busy||!ready||!examples.length;$('csv').disabled=!results.size;
  $('result-select').disabled=busy;document.querySelectorAll('[data-mode]').forEach(b=>b.disabled=busy);
  const ids=[...results.keys()],i=current?ids.indexOf(current.run_id):-1;
  $('previous').disabled=busy||i<=0;$('next').disabled=busy||i<0||i===ids.length-1;
  document.querySelectorAll('#history button').forEach(b=>b.disabled=busy);
}
function fail(e){$('error').hidden=false;$('error').textContent=e.message||String(e);}
async function jsonRequest(path,options={},retries=0){
  const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),180000);
  try{const r=await fetch(api(path),{...options,signal:controller.signal});const data=await r.json();
    if(r.status===429&&retries<3){await new Promise(resolve=>setTimeout(resolve,2000));return jsonRequest(path,options,retries+1);}
    if(!r.ok)throw new Error(data.error||`HTTP ${r.status}`);return data;
  }catch(e){if(e.name==='AbortError')throw new Error('서버 응답이 지연됩니다. 잠시 후 다시 시도하세요.');throw e;}finally{clearTimeout(timer);}
}
async function checkServer(){
  try{const s=await jsonRequest('/api/status');ready=s.ready;$('status').textContent=ready?'준비됨 · 서버 CPU':s.error?'서버 모델 준비 실패':'서버 준비 중';$('limits').textContent=`대칭 확장 후 최대 ${s.max_atoms}개 원자. 무료 서버 메모리 보호를 위한 제한입니다.`;
    if(!ready&&s.error)fail(new Error(s.error));if(!ready&&!s.error&&polling++<60)setTimeout(checkServer,2000);
  }catch(e){ready=false;$('status').textContent='서버 연결 실패';fail(new Error('추론 서버에 연결할 수 없습니다. 서버 주소와 실행 상태를 확인하세요.'));}controls();
}
function selected(){return examples.find(e=>e.mof_id===$('example').value);}
function displayInputs(){const e=selected();if(!e)return;keys.forEach(k=>$(k).value=e.descriptors[k]);$('cif-link').href=api(`/api/examples/${encodeURIComponent(e.mof_id)}/cif`);updateNote();}
function updateNote(){const changed=mode==='example'&&selected()&&keys.some(k=>Number($(k).value)!==selected().descriptors[k]);$('input-note').className=changed?'muted changed':'muted';$('input-note').textContent=mode==='upload'?'업로드한 CIF에 대응하는 네 기술자를 직접 입력하세요.':changed?'원래 구조와 다른 입력을 가정한 민감도 탐색입니다. GCMC 참고값은 보이지만 정확도 평가는 하지 않습니다.':'예제의 원래 값입니다. GCMC 시뮬레이션 기준값과 비교합니다.';}
function setMode(value){mode=value;document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===value));$('example-input').hidden=value!=='example';$('upload-input').hidden=value!=='upload';$('batch-controls').hidden=value!=='example';$('reset').textContent=value==='example'?'원래 값':'값 지우기';if(value==='example')displayInputs();else{keys.forEach(k=>$(k).value='');updateNote();}$('error').hidden=true;controls();}
function descriptors(){const d=Object.fromEntries(keys.map(k=>[k,Number($(k).value)]));if(keys.some(k=>!$(k).value.trim()||!Number.isFinite(d[k])))throw new Error('네 기술자를 유한한 숫자로 입력하세요.');if(d.surface_area_m2g<=0||d.void_fraction<=0||d.void_fraction>1||d.lcd<=0||d.pld<=0||d.pld>d.lcd)throw new Error('표면적·LCD·PLD > 0, 공극률 0 초과 1 이하, PLD ≤ LCD 조건을 확인하세요.');return d;}
async function predict(e,d,upload=false){
  let result;
  if(upload){const f=$('cif').files[0];if(!f||!f.name.toLowerCase().endsWith('.cif')||!f.size||f.size>2000000)throw new Error('0바이트 초과 2 MB 이하의 .cif 파일을 선택하세요.');const form=new FormData();form.append('cif',f);keys.forEach(k=>form.append(k,d[k]));result=await jsonRequest('/api/predict',{method:'POST',body:form});}
  else result=await jsonRequest('/api/predict/example',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mof_id:e.mof_id,descriptors:d})});
  result.run_id=++counter;result.source=upload?'upload':'example';results.set(result.run_id,result);current=result;history();render();return result;
}
function selectResult(id){const r=results.get(Number(id));if(r){current=r;render();}}
function history(){
  $('history-area').hidden=false;$('result-nav').hidden=false;
  $('result-select').replaceChildren(...Array.from(results.values(),r=>new Option(`#${r.run_id} · ${r.filename} · ${r.source==='upload'?'업로드':r.modified?'직접 입력':'원래 값'}`,r.run_id)));
  $('history').replaceChildren(...Array.from(results.values(),r=>{const tr=document.createElement('tr'),td=document.createElement('td'),button=document.createElement('button');button.type='button';button.className='history-link';button.textContent=`${r.filename} / #${r.run_id} 보기`;button.addEventListener('click',()=>selectResult(r.run_id));td.append(button);tr.append(td);for(const text of[r.source==='upload'?'업로드':r.modified?'직접 입력':'원래 값',r.timing.inference_seconds.toFixed(2)+' s']){const cell=document.createElement('td');cell.textContent=text;tr.append(cell);}return tr;}));
}
function gasRows(){const offset=gas==='CO2'?0:5;return current.predictions[gas].map((r,i)=>({...r,reference:current.reference?current.reference[offset+i]:null,error:current.comparable?Math.abs(r.uptake_mmol_g-current.reference[offset+i]):null}));}
function render(){
  if(!current)return;$('empty').hidden=true;$('output').hidden=false;$('result-select').value=current.run_id;
  $('result-name').textContent=current.filename;$('timing').textContent=`구조 ${current.timing.graph_seconds.toFixed(2)} s · 추론 ${current.timing.inference_seconds.toFixed(2)} s`;
  const d=current.descriptors;$('result-inputs').textContent=`#${current.run_id} · 표면적 ${d.surface_area_m2g} m²/g · 공극률 ${d.void_fraction} · LCD ${d.lcd} Å · PLD ${d.pld} Å`;
  $('structure').textContent=`원자 ${current.structure.atoms} · 주기 간선 ${current.structure.periodic_edges} · ${current.structure.elements.join(', ')}`;
  document.querySelectorAll('[data-gas]').forEach(b=>b.classList.toggle('active',b.dataset.gas===gas));
  const rows=gasRows(),label=gas==='CO2'?'CO₂':'CH₄';$('values').replaceChildren(...rows.map(r=>{const tr=document.createElement('tr');for(const t of[label,r.pressure_bar,r.uptake_mmol_g.toFixed(6),r.reference===null?'—':r.reference.toFixed(6),r.error===null?'—':r.error.toFixed(6)]){const td=document.createElement('td');td.textContent=t;tr.append(td);}return tr;}));
  $('reference-note').textContent=current.comparable?`${label} 5개 압력 MAE ${(rows.reduce((s,r)=>s+r.error,0)/5).toFixed(6)} mmol/g. 기준값은 실험값이 아닌 GCMC 시뮬레이션입니다.`:current.reference?'변경 입력에 대해 원래 GCMC를 정답으로 평가할 수 없습니다. 원래 구조의 참고 곡선만 표시합니다.':'이 구조에는 등록된 GCMC 기준값이 없어 예측만 표시합니다.';
  const w=600,h=280,p={l:60,r:20,t:42,b:48},maximum=Math.max(...rows.flatMap(r=>[r.uptake_mmol_g,r.reference||0]))*1.12||1,logs=rows.map(r=>Math.log10(r.pressure_bar));
  const x=i=>p.l+(logs[i]-logs[0])/(logs[4]-logs[0])*(w-p.l-p.r),y=v=>h-p.b-v/maximum*(h-p.t-p.b);
  const line=(values,color,dash)=>`<path d="${values.map((v,i)=>`${i?'L':'M'}${x(i)},${y(v)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2.5" ${dash?'stroke-dasharray="6 4"':''}/>${values.map((v,i)=>`<circle cx="${x(i)}" cy="${y(v)}" r="3.5" fill="white" stroke="${color}" stroke-width="2"/>`).join('')}`;
  $('chart').innerHTML=`<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg"><text x="60" y="13" font-size="11" fill="#626e77">${label} · 흡착량 (mmol g⁻¹)</text><text x="60" y="30" fill="#1f5f8b" font-size="11">━ 모델 예측</text>${current.reference?'<text x="180" y="30" fill="#be7035" font-size="11">┄ GCMC 기준 (원래 입력)</text>':''}${[0,.25,.5,.75,1].map(t=>`<line x1="60" x2="580" y1="${y(t*maximum)}" y2="${y(t*maximum)}" stroke="#e1e6e9"/><text x="51" y="${y(t*maximum)+4}" text-anchor="end" font-size="11" fill="#626e77">${(t*maximum).toFixed(maximum<1?2:1)}</text>`).join('')}${current.reference?line(rows.map(r=>r.reference),'#be7035',true):''}${line(rows.map(r=>r.uptake_mmol_g),'#1f5f8b',false)}${rows.map((r,i)=>`<text x="${x(i)}" y="254" text-anchor="middle" font-size="11" fill="#626e77">${r.pressure_bar}</text>`).join('')}<text x="300" y="274" text-anchor="middle" font-size="11" fill="#626e77">압력 (bar, 로그 축)</text></svg>`;controls();
}
function csvCell(value){if(value===null||value===undefined)return '';let t=String(value);if(/^[=+@\t\r]/.test(t)||(/^\-/.test(t)&&!Number.isFinite(Number(t))))t="'"+t;return /[",\r\n]/.test(t)?'"'+t.replaceAll('"','""')+'"':t;}
function csv(){const rows=[['run_id','structure','source','surface_area_m2g','void_fraction','lcd','pld','gas','pressure_bar','prediction_mmol_g','gcmc_reference_mmol_g','absolute_error_mmol_g','comparable','cif_sha256','execution']];for(const r of results.values())for(const g of ['CO2','CH4'])r.predictions[g].forEach((v,i)=>{const ref=r.reference?r.reference[(g==='CO2'?0:5)+i]:null;rows.push([r.run_id,r.filename,r.source,...keys.map(k=>r.descriptors[k]),g,v.pressure_bar,v.uptake_mmol_g,ref,r.comparable?Math.abs(v.uptake_mmol_g-ref):null,r.comparable,r.cif_sha256,r.execution]);});return '\ufeff'+rows.map(row=>row.map(csvCell).join(',')).join('\r\n');}
$('csv').addEventListener('click',()=>{const url=URL.createObjectURL(new Blob([csv()],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download='MOF_predictions_and_GCMC.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);});
$('example').addEventListener('change',()=>{displayInputs();const match=[...results.values()].filter(r=>r.mof_key===selected().mof_id&&r.source==='example').at(-1);if(match)selectResult(match.run_id);});
document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>setMode(b.dataset.mode)));
document.querySelectorAll('[data-gas]').forEach(b=>b.addEventListener('click',()=>{gas=b.dataset.gas;render();}));
keys.forEach(k=>$(k).addEventListener('input',updateNote));$('reset').addEventListener('click',()=>{if(mode==='example')displayInputs();else{keys.forEach(k=>$(k).value='');updateNote();}$('error').hidden=true;});
$('result-select').addEventListener('change',()=>selectResult($('result-select').value));
for(const [id,delta]of[['previous',-1],['next',1]])$(id).addEventListener('click',()=>{const ids=[...results.keys()];selectResult(ids[ids.indexOf(current.run_id)+delta]);});
$('cif').addEventListener('change',()=>{$('file-info').textContent=$('cif').files[0]?.name||'파일을 선택하세요.';keys.forEach(k=>$(k).value='');updateNote();});
$('descriptor-form').addEventListener('submit',e=>{e.preventDefault();if(!$('predict').disabled)$('predict').click();});
$('predict').addEventListener('click',async()=>{if(!$('descriptor-form').reportValidity())return;let d;try{d=descriptors();}catch(e){fail(e);return;}busy=true;controls();$('error').hidden=true;$('progress').textContent='서버에서 구조 계산·추론 중…';try{await predict(selected(),d,mode==='upload');$('progress').textContent='완료 · 결과 선택에서 이전 결과를 다시 열 수 있습니다.';}catch(e){fail(e);$('progress').textContent='계산 중단';}finally{busy=false;controls();}});
$('batch').addEventListener('click',async()=>{busy=true;stopped=false;controls();$('error').hidden=true;$('stop').hidden=false;try{let n=0;for(const e of examples){if(stopped)break;$('example').value=e.mof_id;displayInputs();$('progress').textContent=`${n+1} / 20 · ${e.mof_id} 서버 예측 중…`;await predict(e,e.descriptors);n++;}$('progress').textContent=`${n} / 20 완료 · 결과 선택이나 실행 기록의 보기를 누르세요.`;}catch(e){fail(e);}finally{busy=false;$('stop').hidden=true;controls();}});
$('stop').addEventListener('click',()=>stopped=true);$('check-server').addEventListener('click',()=>{polling=0;checkServer();});
(async()=>{controls();try{const response=await jsonRequest('/api/examples');examples=response.examples;$('example').replaceChildren(...examples.map(e=>new Option(`${e.mof_id} · ${e.atoms} atoms`,e.mof_id)));displayInputs();}catch(e){fail(e);}await checkServer();controls();})();
