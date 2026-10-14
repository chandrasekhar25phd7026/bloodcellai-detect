const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

const html = fs.readFileSync('index.test.html', 'utf8');

function makeFile(File, name, content) {
  return new File([content], name, { type: 'text/plain' });
}

async function waitFor(pred, timeoutMs = 4000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (pred()) return true;
    await new Promise((r) => setTimeout(r, 20));
  }
  return false;
}

function setInputFiles(input, filesArr) {
  const dt = new (input.ownerDocument.defaultView.DataTransfer || Object)();
  // jsdom doesn't ship DataTransfer by default in all versions; build a minimal FileList-like
  const fileListLike = {
    length: filesArr.length,
    item: (i) => filesArr[i],
    [Symbol.iterator]: function* () { for (const f of filesArr) yield f; },
  };
  for (let i = 0; i < filesArr.length; i++) fileListLike[i] = filesArr[i];
  Object.defineProperty(input, 'files', { value: fileListLike, configurable: true });
}

async function run() {
  const dom = new JSDOM(html, {
    url: 'https://example.org/tool/',
    runScripts: 'dangerously',
    resources: 'usable',
    pretendToBeVisual: true,
  });
  const { window } = dom;
  window.setImmediate = setImmediate;
  // jsdom's Blob/File implementation has no .text()/.arrayBuffer() (real browsers have had these for years);
  // polyfill via FileReader, which jsdom DOES support properly, so the page's own real code is unmodified.
  if (!window.Blob.prototype.text) {
    window.Blob.prototype.text = function () {
      return new Promise((resolve, reject) => {
        const fr = new window.FileReader();
        fr.onload = () => resolve(fr.result);
        fr.onerror = () => reject(fr.error);
        fr.readAsText(this);
      });
    };
  }
 // jsdom doesn't implement this; JSZip's internal chunking needs it
  // jsdom lacks canvas/createImageBitmap support: our code must fall back gracefully, not crash.
  window.createImageBitmap = undefined;
  if (!window.URL.createObjectURL) window.URL.createObjectURL = () => 'blob:mock-url';

  await new Promise((r) => setTimeout(r, 50)); // let inline scripts finish executing (jszip + wiring IIFE)
  const doc = window.document;
  const File = window.File;

  // ---- Test 1: folder picker path, classification_folders layout ----
  const folderFiles = [
    makeFile(File, 'mydata/neutrophil/0.jpg', 'x'),
    makeFile(File, 'mydata/neutrophil/1.jpg', 'x'),
    makeFile(File, 'mydata/lymphocyte/0.jpg', 'x'),
    makeFile(File, 'mydata/lymphocyte/1.jpg', 'x'),
  ];
  for (const f of folderFiles) {
    Object.defineProperty(f, 'webkitRelativePath', { value: f.name, configurable: true });
  }
  const pickFolder = doc.getElementById('pick-folder');
  setInputFiles(pickFolder, folderFiles);
  pickFolder.dispatchEvent(new window.Event('change', { bubbles: true }));

  const ok1 = await waitFor(() => doc.getElementById('results').classList.contains('show'));
  const reportTitle1 = doc.querySelector('#report-v14 .report-title');
  console.log('TEST 1 (classification folders via folder picker):', ok1 ? 'rendered' : 'TIMED OUT',
    '| title=', reportTitle1 ? reportTitle1.textContent : null);
  if (!(reportTitle1 && reportTitle1.textContent.includes('Class-folder'))) {
    console.log('  FAIL: expected "Class-folder layout"');
    process.exitCode = 1;
  } else console.log('  PASS');

  // ---- Test 2: zip upload path, seg_dir_masks layout (v1.4 should get it WRONG, v1.5 should FIX it) ----
  const JSZip = window.JSZip;
  const zip = new JSZip();
  for (let i = 0; i < 6; i++) {
    zip.file(`images/${i}.png`, 'not-a-real-png-but-name-based-rules-dont-care');
    zip.file(`masks/${i}.png`, 'not-a-real-png-but-name-based-rules-dont-care');
  }
  const zipBuf = await zip.generateAsync({ type: 'uint8array' });
  const zipFile = new File([zipBuf], 'seg_dir_masks.zip', { type: 'application/zip' });
  const pickZip = doc.getElementById('pick-zip');
  setInputFiles(pickZip, [zipFile]);
  pickZip.dispatchEvent(new window.Event('change', { bubbles: true }));

  const ok2 = await waitFor(() => {
    const t = doc.querySelector('#report-v14 .report-title');
    return t && t.textContent.length > 0 && doc.getElementById('scan-summary').textContent.includes('seg_dir_masks');
  });
  const v14title = doc.querySelector('#report-v14 .report-title');
  const v15block = doc.getElementById('report-v15-wrap').innerHTML;
  console.log('TEST 2 (zip upload, seg_dir_masks):', ok2 ? 'rendered' : 'TIMED OUT',
    '| v1.4 title=', v14title ? v14title.textContent : null);
  const v14Wrong = v14title && v14title.textContent.includes('Class-folder'); // v1.4's known gap
  const v15Fixed = v15block.includes('Segmentation masks');
  console.log('  v1.4 shows its known gap (Class-folder layout):', v14Wrong ? 'PASS' : 'FAIL');
  console.log('  v1.5 panel shows the fix (Segmentation masks):', v15Fixed ? 'PASS' : 'FAIL');
  if (!v14Wrong || !v15Fixed) process.exitCode = 1;

  // check evidence panel is populated and mentions the fallback since createImageBitmap is undefined
  const v15text = v15block;
  console.log('  v1.5 evidence present:', v15text.includes('evidence-dump') ? 'PASS' : 'FAIL');
  if (!v15text.includes('evidence-dump')) process.exitCode = 1;

  // ---- Test 3: empty / abstain dataset ----
  const emptyFiles = [makeFile(File, 'weird/readme.md', 'nothing useful here')];
  for (const f of emptyFiles) Object.defineProperty(f, 'webkitRelativePath', { value: f.name, configurable: true });
  setInputFiles(pickFolder, emptyFiles);
  pickFolder.dispatchEvent(new window.Event('change', { bubbles: true }));
  const ok3 = await waitFor(() => {
    const t = doc.querySelector('#report-v14 .report-title');
    return t && t.textContent.includes('Not detected');
  });
  console.log('TEST 3 (no images/labels -> explicit "Not detected", never a silent guess):', ok3 ? 'PASS' : 'FAIL');
  if (!ok3) process.exitCode = 1;

  console.log('\n' + (process.exitCode ? 'SOME TESTS FAILED' : 'ALL E2E TESTS PASSED'));
}
run().catch((e) => { console.error('CRASH', e); process.exit(1); });
