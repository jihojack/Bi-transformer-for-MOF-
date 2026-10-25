'use strict';
// DOM-level logic tests, not a claim of rendered-browser or research-model QA.
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const root=path.resolve(__dirname,'..'),source=fs.readFileSync(path.join(root,'frontend/app.js'),'utf8');
const html=fs.readFileSync(path.join(root,'frontend/index.html'),'utf8');
const catalog=JSON.parse(fs.readFileSync(path.join(root,'app/examples/catalog.json'),'utf8')).examples;
const excluded=JSON.parse(fs.readFileSync(path.join(root,'app/examples/demo_selection.json'),'utf8')).excluded_mof_ids;
const examples=catalog.filter(e=>!excluded.includes(e.mof_id));
const tick=()=>new Promise(resolve=>setImmediate(resolve));

class Element {
  constructor(tag='div'){this.tagName=tag.toUpperCase();this.children=[];this.listeners={};this.dataset={};this.value='';this.hidden=false;this.disabled=false;this.textContent='';this.files=[];this.classList={toggle(){}};}
  addEventListener(type,handler){(this.listeners[type]??=[]).push(handler);}
  replaceChildren(...children){this.children=children;if(this.tagName==='SELECT')this.value=children[0]?.value||'';}
  append(...children){this.children.push(...children);}
  reportValidity(){return true;}
  async fire(type){for(const listener of this.listeners[type]||[])await listener({preventDefault(){}});}
}

function prediction(mofId){
  const e=examples.find(e=>e.mof_id===mofId);
  return {filename:mofId+'.cif',mof_key:mofId,descriptors:e.descriptors,cif_sha256:e.cif_sha256,
    predictions:{CO2:[.01,.05,.1,.5,2.5].map((p,i)=>({pressure_bar:p,uptake_mmol_g:i+1})),CH4:[.05,.5,.9,2.5,4.5].map((p,i)=>({pressure_bar:p,uptake_mmol_g:i+1}))},
    structure:{atoms:e.atoms,periodic_edges:e.periodic_edges,elements:e.elements},timing:{graph_seconds:.1,inference_seconds:.2},
    reference:Array.from({length:10},(_,i)=>i%5+1.1),comparable:true,modified:false,execution:'server_cpu'};
}

function harness(statusFetch){
  const elements=new Map([...html.matchAll(/id="([^"]+)"/g)].map(m=>[m[1],new Element(m[1]==='example'||m[1]==='result-select'?'select':'div')]));
  elements.get('error').hidden=true;
  const modeButtons=['example','upload'].map(mode=>Object.assign(new Element('button'),{dataset:{mode}}));
  const gasButtons=['CO2','CH4'].map(gas=>Object.assign(new Element('button'),{dataset:{gas}}));
  const timers=new Map(),requests=[],documentEvents={},windowEvents={};let timerId=0;
  const document={visibilityState:'visible',getElementById(id){assert.ok(elements.has(id),'Unknown/deleted element: '+id);return elements.get(id);},
    querySelector(selector){assert.equal(selector,'[data-mode="example"]');return modeButtons[0];},
    querySelectorAll(selector){if(selector==='[data-mode]')return modeButtons;if(selector==='[data-gas]')return gasButtons;if(selector==='#history button')return elements.get('history').children.flatMap(tr=>tr.children.flatMap(td=>td.children.filter(e=>e.tagName==='BUTTON')));throw new Error('Unexpected selector '+selector);},
    createElement(tag){return new Element(tag);},addEventListener(type,fn){documentEvents[type]=fn;}};
  const reply=data=>({ok:true,status:200,json:async()=>data});
  const context=vm.createContext({document,window:{MOF_API_BASE:'',addEventListener(type,fn){windowEvents[type]=fn;}},
    Option:class {constructor(text,value){this.textContent=text;this.value=String(value);}},AbortController,Blob,URL,FormData,
    setTimeout(fn,ms){const id=++timerId;timers.set(id,{fn,ms});return id;},clearTimeout(id){timers.delete(id);},
    fetch:async(url,options={})=>{requests.push({url,options});if(url==='/api/status')return statusFetch?statusFetch(reply):reply({ready:true,max_atoms:200});if(url==='/api/examples')return reply({examples});if(url==='/api/predict/example')return reply(prediction(JSON.parse(options.body).mof_id));throw new Error('Unexpected API route '+url);}});
  vm.runInContext(source,context);
  return {context,elements,modeButtons,timers,requests,windowEvents,documentEvents,
    async flush(){await tick();await tick();},
    async retry(){assert.equal(timers.size,1);const [id,timer]=timers.entries().next().value;timers.delete(id);timer.fn();await tick();await tick();},
    async run(code){const result=vm.runInContext(code,context);await result;await tick();}};
}

test('18 curated examples, dynamic labels, no manual status button or healthy heartbeat',async()=>{
  const h=harness();await h.flush();
  assert.equal(examples.length,18);assert.ok(!html.includes('id="check-server"'));
  assert.equal(h.elements.get('example').children.length,18);
  assert.equal(h.modeButtons[0].textContent,'예제 18개');
  assert.equal(h.elements.get('batch').textContent,'18개 원래 값으로 순차 예측');
  assert.equal(h.elements.get('status').textContent,'준비됨 · 서버 CPU');
  assert.equal(h.timers.size,0);assert.equal(h.elements.get('predict').disabled,false);
});

test('network cold start retries and loads the catalog after recovery',async()=>{
  let attempts=0;const h=harness(reply=>{if(++attempts===1)throw new Error('network unavailable');return reply({ready:true,max_atoms:200});});
  await h.flush();assert.match(h.elements.get('status').textContent,/재연결/);
  await h.retry();assert.equal(attempts,2);assert.equal(h.elements.get('example').children.length,18);assert.equal(h.timers.size,0);
});

test('temporary HTML loading page is retried instead of leaving an empty selector',async()=>{
  let attempts=0;const h=harness(reply=>++attempts===1?{ok:true,status:200,json:async()=>{throw new SyntaxError('HTML');}}:reply({ready:true,max_atoms:200}));
  await h.flush();await h.retry();assert.equal(h.elements.get('example').children.length,18);
});

test('eight failed checks stop; online event recovers and clears only server error',async()=>{
  let recover=false,attempts=0;const h=harness(reply=>{attempts++;if(!recover)throw new Error('offline');return reply({ready:true,max_atoms:200});});
  await h.flush();for(let i=1;i<8;i++)await h.retry();
  assert.equal(attempts,8);assert.equal(h.timers.size,0);assert.equal(h.elements.get('error').hidden,false);
  recover=true;h.windowEvents.online();await h.flush();
  assert.equal(h.elements.get('error').hidden,true);assert.equal(h.elements.get('example').children.length,18);
  await h.run("fail(new Error('prediction error')); checkServer(true)");
  assert.equal(h.elements.get('error').hidden,false);assert.equal(h.elements.get('error').textContent,'prediction error');
});

test('overlapping online/visibility events do not issue duplicate status requests',async()=>{
  let resolve;const h=harness(reply=>new Promise(done=>{resolve=()=>done(reply({ready:true,max_atoms:200}));}));
  h.windowEvents.online();h.documentEvents.visibilitychange();
  assert.equal(h.requests.filter(r=>r.url==='/api/status').length,1);
  resolve();await h.flush();assert.equal(h.elements.get('example').children.length,18);
});

test('permanent model errors stay unhealthy and do not retry indefinitely',async()=>{
  const h=harness(reply=>reply({ready:false,error:'model failed',max_atoms:200}));await h.flush();
  assert.equal(h.elements.get('predict').disabled,true);assert.equal(h.timers.size,0);
  assert.equal(h.elements.get('error').textContent,'model failed');
});

test('all 18 sequential predictions remain browsable and CSV retains every run',async()=>{
  const h=harness();await h.flush();await h.elements.get('batch').fire('click');
  assert.match(h.elements.get('progress').textContent,/18 \/ 18 완료/);
  assert.equal(h.elements.get('result-select').children.length,18);
  const calls=h.requests.filter(r=>r.url==='/api/predict/example');assert.equal(calls.length,18);
  assert.ok(calls.every(r=>!excluded.includes(JSON.parse(r.options.body).mof_id)));
  h.elements.get('result-select').value='5';await h.elements.get('result-select').fire('change');
  assert.equal(h.elements.get('result-name').textContent,examples[4].mof_id+'.cif');
  const csv=vm.runInContext('csv()',h.context);assert.equal(csv.split('\r\n').length,181);
  assert.ok(excluded.every(id=>!csv.includes(id)));
});
