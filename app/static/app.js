const form = document.querySelector('#predict-form');
const fileInput = document.querySelector('#cif-input');
const dropZone = document.querySelector('#drop-zone');
const fileTitle = document.querySelector('#file-title');
const fileDetail = document.querySelector('#file-detail');
const modelStatus = document.querySelector('#model-status');
const predictButton = document.querySelector('#predict-button');
const exportButton = document.querySelector('#export-button');
const emptyState = document.querySelector('#empty-state');
const busyState = document.querySelector('#busy-state');
const errorState = document.querySelector('#error-state');
const resultContent = document.querySelector('#result-content');
const chart = document.querySelector('#isotherm-chart');

let currentResult = null;
let currentGas = 'CO2';

function setView(view) {
  emptyState.hidden = view !== 'empty';
  busyState.hidden = view !== 'busy';
  errorState.hidden = view !== 'error';
  resultContent.hidden = view !== 'result';
  exportButton.hidden = view !== 'result';
}

async function pollStatus() {
  try {
    const response = await fetch('/api/status');
    const status = await response.json();
    if (status.ready) {
      modelStatus.className = 'status ready';
      modelStatus.querySelector('span').textContent = `준비됨 · ${status.device}`;
      predictButton.disabled = false;
      return;
    }
    if (status.error) {
      modelStatus.className = 'status error';
      modelStatus.querySelector('span').textContent = '모델 로드 실패';
      document.querySelector('#error-message').textContent = status.error;
      setView('error');
      return;
    }
    setTimeout(pollStatus, 800);
  } catch (error) {
    modelStatus.className = 'status error';
    modelStatus.querySelector('span').textContent = '서버 연결 실패';
    document.querySelector('#error-message').textContent = error.message;
    setView('error');
  }
}

function showFile(file) {
  if (!file) return;
  fileTitle.textContent = file.name;
  fileDetail.textContent = `${(file.size / 1024).toFixed(1)} KB · 준비됨`;
  dropZone.classList.add('drag');
}

fileInput.addEventListener('change', () => showFile(fileInput.files[0]));
['dragenter', 'dragover'].forEach(type => dropZone.addEventListener(type, event => {
  event.preventDefault();
  dropZone.classList.add('drag');
}));
['dragleave', 'drop'].forEach(type => dropZone.addEventListener(type, event => {
  event.preventDefault();
  if (type === 'dragleave') dropZone.classList.remove('drag');
}));
dropZone.addEventListener('drop', event => {
  const file = [...event.dataTransfer.files].find(item => item.name.toLowerCase().endsWith('.cif'));
  if (!file) return;
  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileInput.files = transfer.files;
  showFile(file);
});

document.querySelector('#load-example').addEventListener('click', async () => {
  try {
    const [metaResponse, cifResponse] = await Promise.all([fetch('/api/example-metadata'), fetch('/api/example')]);
    if (!metaResponse.ok || !cifResponse.ok) throw new Error('예제 파일을 읽을 수 없습니다.');
    const meta = await metaResponse.json();
    const file = new File([await cifResponse.blob()], 'example_mof.cif', {type: 'chemical/x-cif'});
    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
    Object.entries(meta.descriptors).forEach(([name, value]) => { form.elements[name].value = value; });
    showFile(file);
  } catch (error) {
    document.querySelector('#error-message').textContent = error.message;
    setView('error');
  }
});

form.addEventListener('submit', async event => {
  event.preventDefault();
  if (!form.reportValidity()) return;
  if (Number(form.elements.pld.value) > Number(form.elements.lcd.value)) {
    document.querySelector('#error-message').textContent = 'PLD는 LCD보다 클 수 없습니다. 입력값을 확인하세요.';
    setView('error');
    return;
  }
  predictButton.disabled = true;
  setView('busy');
  try {
    const response = await fetch('/api/predict', {method: 'POST', body: new FormData(form)});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || '예측에 실패했습니다.');
    currentResult = payload;
    currentGas = 'CO2';
    renderResult();
    setView('result');
  } catch (error) {
    document.querySelector('#error-message').textContent = error.message;
    setView('error');
  } finally {
    predictButton.disabled = false;
  }
});

document.querySelectorAll('.gas-tabs button').forEach(button => button.addEventListener('click', () => {
  currentGas = button.dataset.gas;
  renderGas();
}));

function renderResult() {
  document.querySelector('#result-filename').textContent = currentResult.filename;
  document.querySelector('#stat-atoms').textContent = currentResult.structure.atoms.toLocaleString();
  document.querySelector('#stat-edges').textContent = currentResult.structure.periodic_edges.toLocaleString();
  document.querySelector('#stat-elements').textContent = currentResult.structure.elements.join(', ');
  document.querySelector('#stat-time').textContent = `${currentResult.timing.total_seconds.toFixed(2)} s`;
  renderGas();
}

function renderGas() {
  if (!currentResult) return;
  document.querySelectorAll('.gas-tabs button').forEach(button => {
    button.classList.toggle('active', button.dataset.gas === currentGas);
  });
  const rows = currentResult.predictions[currentGas];
  const label = currentGas === 'CO2' ? 'CO₂' : 'CH₄';
  document.querySelector('#gas-title').textContent = `${label} · 298 K`;
  document.querySelector('#prediction-table').innerHTML = rows.map(row =>
    `<tr><td>${label}</td><td>${row.pressure_bar}</td><td>${row.uptake_mmol_g.toFixed(6)}</td></tr>`
  ).join('');
  renderChart(rows, label);
}

function renderChart(rows, gasLabel) {
  const width = Math.max(480, chart.clientWidth || 680);
  const height = 220;
  const pad = {l: 52, r: 18, t: 18, b: 38};
  const xs = rows.map(row => Math.log10(row.pressure_bar));
  const ys = rows.map(row => row.uptake_mmol_g);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMax = Math.max(...ys) * 1.12 || 1;
  const x = value => pad.l + (value - xMin) / (xMax - xMin || 1) * (width - pad.l - pad.r);
  const y = value => height - pad.b - value / yMax * (height - pad.t - pad.b);
  const ticks = [0, .25, .5, .75, 1].map(fraction => ({value: yMax * fraction, py: y(yMax * fraction)}));
  const path = rows.map((row, index) => `${index ? 'L' : 'M'}${x(xs[index]).toFixed(1)},${y(row.uptake_mmol_g).toFixed(1)}`).join(' ');

  chart.innerHTML = `<svg viewBox="0 0 ${width} ${height}" aria-label="${gasLabel} 예측 등온선">
    ${ticks.map(tick => `<line x1="${pad.l}" x2="${width-pad.r}" y1="${tick.py}" y2="${tick.py}" stroke="#e1e6e9"/><text x="${pad.l-8}" y="${tick.py+4}" text-anchor="end" fill="#68747c" font-size="9">${tick.value.toFixed(yMax < 1 ? 2 : 1)}</text>`).join('')}
    <path d="${path}" fill="none" stroke="#1f5f8b" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
    ${rows.map((row, index) => `<g><circle cx="${x(xs[index])}" cy="${y(row.uptake_mmol_g)}" r="4" fill="#fff" stroke="#1f5f8b" stroke-width="2"/><text x="${x(xs[index])}" y="${height-14}" text-anchor="middle" fill="#68747c" font-size="9">${row.pressure_bar}</text></g>`).join('')}
    <text x="${(pad.l+width-pad.r)/2}" y="${height-2}" text-anchor="middle" fill="#68747c" font-size="9">압력 (bar, log scale)</text>
  </svg>`;
}

exportButton.addEventListener('click', async () => {
  if (!currentResult) return;
  const response = await fetch('/api/export', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({prediction: currentResult})});
  if (!response.ok) return;
  const url = URL.createObjectURL(await response.blob());
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `${currentResult.mof_key}_prediction.csv`;
  anchor.click();
  URL.revokeObjectURL(url);
});

window.addEventListener('resize', () => currentResult && renderChart(currentResult.predictions[currentGas], currentGas));
setView('empty');
pollStatus();
