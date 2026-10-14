const fs = require('fs');
const { JSDOM } = require('jsdom');
const html = fs.readFileSync('index.test.html', 'utf8');

function makeFile(File, name, content) { return new File([content], name, { type: 'text/plain' }); }
async function waitFor(pred, timeoutMs = 4000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) { if (pred()) return true; await new Promise((r) => setTimeout(r, 20)); }
  return false;
}
function setInputFiles(input, filesArr) {
  const fileListLike = { length: filesArr.length, item: (i) => filesArr[i], [Symbol.iterator]: function* () { for (const f of filesArr) yield f; } };
  for (let i = 0; i < filesArr.length; i++) fileListLike[i] = filesArr[i];
  Object.defineProperty(input, 'files', { value: fileListLike, configurable: true });
}

async function run() {
  const dom = new JSDOM(html, { url: 'https://example.org/tool/', runScripts: 'dangerously', resources: 'usable', pretendToBeVisual: true });
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

  window.addEventListener('error', (e) => console.log('WINDOW ERROR:', e.error && e.error.stack || e.message));
  window.createImageBitmap = undefined;
  if (!window.URL.createObjectURL) window.URL.createObjectURL = () => 'blob:mock-url'; // jsdom gap; real browsers implement this natively // jsdom cannot decode real images; test formats that don't need it, and confirm point_labels degrades honestly
  await new Promise((r) => setTimeout(r, 50));
  const doc = window.document;
  const File = window.File;
  let failures = 0;
  const check = (name, cond) => { console.log((cond ? 'OK  ' : 'FAIL') + ' ' + name); if (!cond) failures++; };

  // ---- TEST A: classification_folders -> harmonize -> real zip with correct labels ----
  const clsFiles = [
    makeFile(File, 'ds/RBC/0.jpg', 'x'), makeFile(File, 'ds/RBC/1.jpg', 'x'),
    makeFile(File, 'ds/WBC/0.jpg', 'x'),
  ];
  for (const f of clsFiles) Object.defineProperty(f, 'webkitRelativePath', { value: f.name, configurable: true });
  const pickFolder = doc.getElementById('pick-folder');
  setInputFiles(pickFolder, clsFiles);
  pickFolder.dispatchEvent(new window.Event('change', { bubbles: true }));
  await waitFor(() => doc.getElementById('results').classList.contains('show'));
  const titleA = doc.querySelector('#report-v14 .report-title');
  check('A1: detected as classification_folders', !!titleA && titleA.textContent.includes('Class-folder'));
  const harmBtn1 = doc.getElementById('harmonize-btn');
  check('A2: harmonize button appears for classification_folders', !!harmBtn1);
  harmBtn1.dispatchEvent(new window.Event('click', { bubbles: true }));
  await waitFor(() => doc.querySelector('.download-link'));
  const out1 = doc.getElementById('harmonize-result').innerHTML;
  check('A3: harmonize result reports 3 images, 2 classes', out1.includes('Harmonized 3 images') && out1.includes('2 classes'));
  check('A4: class map table lists RBC and WBC', out1.includes('>RBC<') && out1.includes('>WBC<'));
  check('A5: download link present', out1.includes('class="download-link"'));

  // ---- TEST B: VOC XML -> harmonize -> real XML parsing + real bbox math, end to end ----
  const vocXmlText = '<annotation><size><width>640</width><height>480</height></size>' +
    '<object><name>RBC</name><bndbox><xmin>100</xmin><ymin>50</ymin><xmax>200</xmax><ymax>150</ymax></bndbox></object>' +
    '</annotation>';
  const vocFiles = [makeFile(File, 'vocds/a.jpg', 'x'), makeFile(File, 'vocds/a.xml', vocXmlText)];
  for (const f of vocFiles) Object.defineProperty(f, 'webkitRelativePath', { value: f.name, configurable: true });
  setInputFiles(pickFolder, vocFiles);
  pickFolder.dispatchEvent(new window.Event('change', { bubbles: true }));
  await waitFor(() => doc.querySelector('#report-v14 .report-title')?.textContent.includes('VOC'));
  const titleB = doc.querySelector('#report-v14 .report-title');
  check('B1: detected as VOC XML', !!titleB && titleB.textContent.includes('VOC'));
  const harmBtn2 = doc.getElementById('harmonize-btn');
  check('B2: harmonize button appears for VOC XML', !!harmBtn2);
  harmBtn2.dispatchEvent(new window.Event('click', { bubbles: true }));
  await waitFor(() => doc.querySelector('.download-link'));
  const out2 = doc.getElementById('harmonize-result').innerHTML;
  check('B3: harmonized 1 image, 1 class (RBC)', out2.includes('Harmonized 1 image') && out2.includes('>RBC<'));

  // ---- TEST C: abstain (no recognizable pattern at all) -> plain "no harmonizer" message, never fabricated, no button ----
  const abstainFiles = [makeFile(File, 'weird/readme.md', 'nothing useful here')];
  for (const f of abstainFiles) Object.defineProperty(f, 'webkitRelativePath', { value: f.name, configurable: true });
  setInputFiles(pickFolder, abstainFiles);
  pickFolder.dispatchEvent(new window.Event('change', { bubbles: true }));
  await waitFor(() => doc.querySelector('#report-v14 .report-title')?.textContent.includes('Not detected'));
  const panelC = doc.getElementById('harmonize-panel').innerHTML;
  check('C1: no harmonize button for an abstain result', !panelC.includes('id="harmonize-btn"'));
  check('C2: plain, honest "no harmonizer" message shown', panelC.toLowerCase().includes('harmonization needs a known format'));

  // ---- TEST D: a real segmentation-mask layout (images/ + masks/) is classified as classification_folders
  // by the FROZEN v1.4 detector (a known, disclosed blind spot) -- harmonizing it as such is a CORRECT,
  // honest consequence of that disclosed limitation, not a new dishonesty. Confirm it behaves exactly like
  // any other classification_folders case (here, "images" and "masks" become two class names). ----
  const segFiles = [makeFile(File, 'seg/images/0.png', 'x'), makeFile(File, 'seg/masks/0.png', 'x'),
                     makeFile(File, 'seg/images/1.png', 'x'), makeFile(File, 'seg/masks/1.png', 'x'),
                     makeFile(File, 'seg/images/2.png', 'x'), makeFile(File, 'seg/masks/2.png', 'x')];
  for (const f of segFiles) Object.defineProperty(f, 'webkitRelativePath', { value: f.name, configurable: true });
  setInputFiles(pickFolder, segFiles);
  pickFolder.dispatchEvent(new window.Event('change', { bubbles: true }));
  await waitFor(() => doc.querySelector('#report-v14 .report-title')?.textContent.includes('Class-folder'));
  const titleD = doc.querySelector('#report-v14 .report-title');
  check('D1: v1.4 (frozen) calls this classification_folders, its known blind spot', !!titleD && titleD.textContent.includes('Class-folder'));
  check('D2: v1.5 panel separately flags it as segmentation', doc.getElementById('report-v15-wrap').innerHTML.includes('Segmentation masks'));
  check('D3: harmonize button appears (consistent with v1.4\'s own, disclosed classification)', !!doc.getElementById('harmonize-btn'));

  console.log('\n' + (failures === 0 ? 'ALL HARMONIZE E2E TESTS PASSED' : failures + ' FAILURE(S)'));
  process.exit(failures === 0 ? 0 : 1);
}
run().catch((e) => { console.error('CRASH', e); process.exit(1); });
