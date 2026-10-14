const { JSDOM } = require('jsdom');
global.DOMParser = new JSDOM().window.DOMParser;
const H = require('../harmonize.js');
const assert = require('assert');
let failures = 0;
function check(name, cond) { console.log((cond ? 'OK  ' : 'FAIL') + ' ' + name); if (!cond) failures++; }

// ---- VOC XML ----
const vocXml = `<annotation>
  <size><width>640</width><height>480</height></size>
  <object><name>RBC</name><bndbox><xmin>100</xmin><ymin>50</ymin><xmax>200</xmax><ymax>150</ymax></bndbox></object>
  <object><name>WBC</name><bndbox><xmin>300</xmin><ymin>200</ymin><xmax>340</xmax><ymax>260</ymax></bndbox></object>
</annotation>`;
const voc = H.parseVocXml(vocXml);
check('VOC width/height', voc.width === 640 && voc.height === 480);
check('VOC object count', voc.objects.length === 2);
const o0 = voc.objects[0];
// xc=((100+200)/2)/640=0.234375 yc=((50+150)/2)/480=0.208333 w=(200-100)/640=0.15625 h=(150-50)/480=0.208333
check('VOC box0 xc', Math.abs(o0.xc - 0.234375) < 1e-9);
check('VOC box0 yc', Math.abs(o0.yc - 100/480) < 1e-9);
check('VOC box0 w', Math.abs(o0.w - 100/640) < 1e-9);
check('VOC box0 h', Math.abs(o0.h - 100/480) < 1e-9);
check('VOC class names', voc.objects.map(o=>o.className).join(',') === 'RBC,WBC');

// malformed XML must not crash / silently fabricate data
let threw = false;
try { H.parseVocXml('<not xml'); } catch (e) { threw = true; }
check('VOC malformed throws rather than guessing', threw);

// ---- COCO JSON ----
const coco = {
  images: [{ id: 1, file_name: 'a.jpg', width: 200, height: 100 }, { id: 2, file_name: 'b.jpg', width: 100, height: 100 }],
  categories: [{ id: 5, name: 'Platelet' }, { id: 6, name: 'WBC' }],
  annotations: [
    { image_id: 1, category_id: 5, bbox: [10, 10, 20, 20] },   // xc=(10+10)/200=0.1 yc=(10+10)/100=0.2 w=20/200=0.1 h=20/100=0.2
    { image_id: 1, category_id: 6, bbox: [0, 0, 200, 100] },
    { image_id: 2, category_id: 5, bbox: [50, 50, 10, 10] },
  ],
};
const parsed = H.parseCocoJson(coco);
check('COCO images indexed', parsed.imagesByFilename.size === 2);
const recA = parsed.imagesByFilename.get('a.jpg');
check('COCO rec a.jpg annotation count', recA.annotations.length === 2);
const objsA = H.cocoRecordToObjects(recA, parsed.categories);
check('COCO box0 xc/yc/w/h', Math.abs(objsA[0].xc-0.1)<1e-9 && Math.abs(objsA[0].yc-0.2)<1e-9 && Math.abs(objsA[0].w-0.1)<1e-9 && Math.abs(objsA[0].h-0.2)<1e-9);
check('COCO class names resolved', objsA[0].className === 'Platelet' && objsA[1].className === 'WBC');

// ---- YOLO txt passthrough ----
const yoloObjs = H.parseYoloTxt('0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n\nnotaline\n');
check('YOLO parses 2 valid lines, skips junk', yoloObjs.length === 2);
check('YOLO values pass through unchanged', yoloObjs[0].xc === 0.5 && yoloObjs[1].yc === 0.3);

// ---- point labels, using REAL per-image size instead of a hardcoded 640x480 ----
const pts = H.parsePointTxt('369 429 4\n100 120 2\n', 1000, 500, 0.05);
check('point label count', pts.length === 2);
check('point label xc/yc use the REAL image size', Math.abs(pts[0].xc - 369/1000) < 1e-9 && Math.abs(pts[0].yc - 429/500) < 1e-9);
check('point label class kept as given', pts[0].className === 'class 4' && pts[1].className === 'class 2');
const ptsNoDims = H.parsePointTxt('369 429 4\n', 0, 0, 0.05);
check('point labels with unknown image size are skipped, not guessed', ptsNoDims.length === 0);

// ---- contiguous class-id remap (dynamic discovery, not a hardcoded registry) ----
const reg = H.buildClassRegistry(['WBC', 'RBC', 'RBC', 'Parasite', 'WBC']);
check('registry discovers all 3 classes', reg.idToName.length === 3);
check('registry ids sorted alphabetically: Parasite=0,RBC=1,WBC=2', reg.nameToId.get('Parasite')===0 && reg.nameToId.get('RBC')===1 && reg.nameToId.get('WBC')===2);

// ---- full YOLO export build ----
const perImage = [
  { stem: 'img1', objects: [{ className: 'RBC', xc: 0.5, yc: 0.5, w: 0.1, h: 0.1 }, { className: 'WBC', xc: 0.2, yc: 0.2, w: 0.05, h: 0.05 }] },
  { stem: 'img2', objects: [] },
];
const exp = H.buildYoloExport(perImage);
check('export label file for img1 has 2 lines', exp.labelFiles.get('train/labels/img1.txt').trim().split('\n').length === 2);
check('export label file for img2 is empty (no objects)', exp.labelFiles.get('train/labels/img2.txt') === '');
check('export class ids contiguous 0..1 for RBC/WBC alphabetically (RBC=0, WBC=1)',
  exp.labelFiles.get('train/labels/img1.txt').startsWith('0 0.500000') && exp.labelFiles.get('train/labels/img1.txt').includes('1 0.200000'));
check('data.yaml has nc: 2', exp.dataYaml.includes('nc: 2'));
check('data.yaml lists names in id order [RBC, WBC]', exp.dataYaml.includes('names: ["RBC", "WBC"]'));

console.log('\n' + (failures === 0 ? 'ALL HARMONIZE TESTS PASSED' : failures + ' FAILURE(S)'));
process.exit(failures === 0 ? 0 : 1);
