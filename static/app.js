const tierInput = document.querySelectorAll('input[name="tier"]');
const fileInput = document.querySelector('#capture-files');
const dropzone = document.querySelector('#dropzone');
const uploadTitle = document.querySelector('#upload-title');
const uploadHint = document.querySelector('#upload-hint');
const guidance = document.querySelector('#tier-guidance');
const form = document.querySelector('#capture-form');
const message = document.querySelector('#form-message');
const rows = document.querySelector('#capture-rows');
const toast = document.querySelector('#toast');
let toastTimer;

const tierCopy = {
  photos: { title: 'Choose room folders', hint: 'One folder per room · 2–8 stills in each', mode: { multiple: true, directory: true, accept: 'image/jpeg,image/png,image/heic,image/webp' }, note: 'Photo names and folder structure are kept in the local raw bundle.' },
  video: { title: 'Choose a walkthrough clip', hint: 'One original .MOV, .MP4, or .M4V file', mode: { multiple: false, directory: false, accept: '.mov,.mp4,.m4v,video/*' }, note: 'Transfer the original clip without recompression. Maximum request size: 512 MB.' },
  lidar: { title: 'Choose a sensor bundle', hint: 'ZIP must contain depth, pose, and intrinsics files', mode: { multiple: false, directory: false, accept: '.zip,application/zip' }, note: 'Filename checks only. Timestamps, units, calibration, and sensor quality are not verified.' },
};

function selectedTier() {
  return document.querySelector('input[name="tier"]:checked').value;
}

function updateUploadControl() {
  const copy = tierCopy[selectedTier()];
  uploadTitle.textContent = copy.title;
  uploadHint.textContent = copy.hint;
  guidance.textContent = copy.note;
  fileInput.multiple = copy.mode.multiple;
  fileInput.setAttribute('accept', copy.mode.accept);
  if (copy.mode.directory) {
    fileInput.setAttribute('webkitdirectory', '');
    fileInput.setAttribute('directory', '');
  } else {
    fileInput.removeAttribute('webkitdirectory');
    fileInput.removeAttribute('directory');
  }
  fileInput.value = '';
  updateFileLabel();
}

function updateFileLabel() {
  const files = [...fileInput.files];
  if (!files.length) return;
  const total = files.reduce((size, file) => size + file.size, 0);
  const sizeLabel = total > 1024 * 1024 ? `${(total / 1024 / 1024).toFixed(1)} MB` : `${Math.ceil(total / 1024)} KB`;
  uploadTitle.textContent = `${files.length} file${files.length === 1 ? '' : 's'} selected · ${sizeLabel}`;
  uploadHint.textContent = files.length <= 4 ? files.map(file => file.webkitRelativePath || file.name).join(' · ') : 'Ready for local validation';
}

function setMessage(text, isError = false) {
  message.textContent = text;
  message.classList.toggle('error', isError);
}

function notify(text) {
  toast.textContent = text;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 2800);
}

function makeCell(text, className) {
  const cell = document.createElement('td');
  cell.textContent = text;
  if (className) cell.className = className;
  return cell;
}

function renderCaptures(captures) {
  document.querySelector('#capture-count').textContent = `${captures.length} capture${captures.length === 1 ? '' : 's'}`;
  rows.replaceChildren();
  if (!captures.length) {
    const tr = document.createElement('tr');
    tr.className = 'empty-row';
    const td = document.createElement('td');
    td.colSpan = 6;
    td.innerHTML = '<span class="empty-grid">▦</span><strong>No captures yet</strong><small>Validated local inputs will appear here.</small>';
    tr.append(td);
    rows.append(tr);
    return;
  }
  for (const capture of captures) {
    const tr = document.createElement('tr');
    tr.append(makeCell(capture.title, 'capture-name'));
    const tier = makeCell(capture.tier);
    const tag = document.createElement('span');
    tag.className = 'tier-tag';
    tag.textContent = capture.tier;
    tier.replaceChildren(tag);
    tr.append(tier, makeCell(capture.room_count ?? '—'), makeCell('Validated', 'status-tag'), makeCell(capture.processing_status === 'not_measured' ? 'Not measured' : 'Not run'));
    const actions = document.createElement('td');
    actions.className = 'row-actions';
    const runButton = document.createElement('button');
    runButton.type = 'button';
    runButton.textContent = capture.processing_status === 'not_measured' ? 'REPLAY' : 'RUN';
    runButton.addEventListener('click', () => runCapture(capture.capture_id));
    const exportButton = document.createElement('button');
    exportButton.type = 'button';
    exportButton.textContent = 'JSON';
    exportButton.disabled = capture.processing_status !== 'not_measured';
    exportButton.addEventListener('click', () => { window.location.href = `/api/captures/${capture.capture_id}/output`; });
    actions.append(runButton, exportButton);
    tr.append(actions);
    rows.append(tr);
  }
}

async function refreshCaptures() {
  try {
    const response = await fetch('/api/captures');
    if (!response.ok) throw new Error('Could not load captures');
    renderCaptures(await response.json());
  } catch (error) {
    notify(error.message);
  }
}

async function runCapture(captureId) {
  try {
    const response = await fetch(`/api/captures/${captureId}/run`, { method: 'POST' });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Could not generate output');
    await refreshCaptures();
    notify('Contract exported with measurements marked not measured.');
  } catch (error) {
    notify(error.message);
  }
}

for (const input of tierInput) input.addEventListener('change', updateUploadControl);
fileInput.addEventListener('change', updateFileLabel);
document.querySelector('#refresh-captures').addEventListener('click', refreshCaptures);

form.addEventListener('submit', async event => {
  event.preventDefault();
  const files = [...fileInput.files];
  if (!files.length) {
    setMessage('Choose source files before validating.', true);
    return;
  }
  const button = form.querySelector('button[type="submit"]');
  button.disabled = true;
  setMessage('Validating and storing originals locally...');
  const data = new FormData();
  data.set('tier', selectedTier());
  data.set('title', document.querySelector('#capture-title').value);
  for (const file of files) data.append('files', file, file.webkitRelativePath || file.name);
  try {
    const response = await fetch('/api/captures', { method: 'POST', body: data });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Capture validation failed');
    setMessage(`Intake validated · ${result.capture_id.slice(0, 8)} · originals stored locally.`);
    form.reset();
    updateUploadControl();
    await refreshCaptures();
  } catch (error) {
    setMessage(error.message, true);
  } finally {
    button.disabled = false;
  }
});

for (const eventName of ['dragenter', 'dragover']) dropzone.addEventListener(eventName, event => { event.preventDefault(); dropzone.classList.add('dragover'); });
for (const eventName of ['dragleave', 'drop']) dropzone.addEventListener(eventName, event => { event.preventDefault(); dropzone.classList.remove('dragover'); });
dropzone.addEventListener('drop', event => {
  if (event.dataTransfer.files.length) {
    fileInput.files = event.dataTransfer.files;
    updateFileLabel();
  }
});

const labels = { workspace: 'Workspace', protocol: 'Capture protocol', benchmark: 'Benchmark', 'fix-loop': 'Fix loop', compliance: 'Compliance' };
for (const button of document.querySelectorAll('.nav-item')) {
  button.addEventListener('click', () => {
    document.querySelectorAll('.nav-item').forEach(item => item.classList.toggle('active', item === button));
    document.querySelectorAll('.view').forEach(view => view.classList.toggle('active', view.id === `view-${button.dataset.view}`));
    document.querySelector('#crumb-current').textContent = labels[button.dataset.view];
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
}

updateUploadControl();
refreshCaptures();