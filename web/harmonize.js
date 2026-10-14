/*
Harmonization — ports the REUSABLE, format-correct parts of the project's own
`bloodcell` Python package (universal data model, VOC/COCO/YOLO parsers, YOLO
export with contiguous class-id remapping) to JavaScript.

WHAT WAS DELIBERATELY NOT REUSED FROM THE ORIGINAL PACKAGE, AND WHY
  The original `parse_pascal_xml` and `get_classes()` map class names through a
  hardcoded, 6-dataset-specific table (CLASS_MAP / DATASET_REGISTRY: RBC, WBC,
  Platelet, Parasite). On a VOC/COCO dataset with any other class vocabulary,
  that table would silently DROP every unmatched object (the original code
  even logs a warning that it does this). That is exactly the kind of silent
  wrong-information this project has avoided everywhere else, so for a public
  tool fed arbitrary datasets, classes are discovered dynamically from the
  data itself instead: every class/category name actually seen is kept, in
  first-seen order, then remapped to a contiguous 0..N-1 id per the ORIGINAL
  package's export_yolo() convention (only the id *assignment* differs; the
  remapping mechanism itself, and the label/data.yaml file formats, are a
  direct port).

WHAT THIS DOES NOT COVER
  There is no general parser for custom_polygon_json / custom_bbox_json /
  segmentation_masks in the original package either (it has adapters for an
  earlier 6-dataset benchmark only: BCCD/VOC, a generic YOLO adapter, a
  generic COCO adapter, Chula/RBCMorphology-style point labels, and whole-
  image classification). Harmonization below covers exactly those five
  formats and says plainly when a detected format has no harmonizer.
*/
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.BloodCellHarmonize = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const SUPPORTED_FORMATS = new Set(['voc_xml', 'yolo_txt', 'coco_json', 'classification_folders', 'point_labels']);

  function isFloat(x) {
    if (x === '' || x == null) return false;
    return Number.isFinite(Number(x)) && /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(x.trim());
  }

  // ---------- VOC XML (direct port of parse_pascal_xml, minus the hardcoded CLASS_MAP) ----------
  // Returns {width, height, objects: [{className, xc, yc, w, h}]}
  function parseVocXml(text) {
    const doc = new DOMParser().parseFromString(text, 'application/xml');
    if (doc.getElementsByTagName('parsererror').length) throw new Error('not valid XML');
    const sizeEl = doc.querySelector('size');
    const width = parseFloat(sizeEl?.querySelector('width')?.textContent || '0');
    const height = parseFloat(sizeEl?.querySelector('height')?.textContent || '0');
    const objects = [];
    for (const obj of doc.querySelectorAll('object')) {
      const className = obj.querySelector('name')?.textContent?.trim();
      const box = obj.querySelector('bndbox');
      if (!className || !box || !width || !height) continue;
      const xmin = parseFloat(box.querySelector('xmin')?.textContent || 'NaN');
      const ymin = parseFloat(box.querySelector('ymin')?.textContent || 'NaN');
      const xmax = parseFloat(box.querySelector('xmax')?.textContent || 'NaN');
      const ymax = parseFloat(box.querySelector('ymax')?.textContent || 'NaN');
      if ([xmin, ymin, xmax, ymax].some((v) => !Number.isFinite(v))) continue;
      objects.push({
        className,
        xc: ((xmin + xmax) / 2) / width, yc: ((ymin + ymax) / 2) / height,
        w: (xmax - xmin) / width, h: (ymax - ymin) / height,
      });
    }
    return { width, height, objects };
  }

  // ---------- YOLO txt: already normalized; pass through, keeping the raw class id as a string "name" ----------
  // (there is no name available in a .txt file; the id itself is kept as the display name, e.g. "class 2")
  function parseYoloTxt(text) {
    const objects = [];
    for (const raw of text.split('\n')) {
      const line = raw.trim();
      if (!line) continue;
      const parts = line.split(/\s+/);
      if (parts.length !== 5 || !parts.every(isFloat)) continue;
      const [cls, xc, yc, w, h] = parts.map(Number);
      objects.push({ className: 'class ' + cls, xc, yc, w, h });
    }
    return objects;
  }

  // ---------- Point labels (ported from parse_chula_txt / parse_rbc_morphology_txt's x,y,class
  // convention, generalized: uses the REAL decoded image size per image instead of a hardcoded
  // 640x480 canvas, and a configurable box size given as a FRACTION of image width rather than a
  // fixed 70px, so it scales across images of different sizes). This is a disclosed heuristic, not
  // a format standard: there is no "correct" box size for a single labeled point. ----------
  function parsePointTxt(text, width, height, boxFraction) {
    const objects = [];
    const boxWFrac = boxFraction, boxHFrac = boxFraction * (width > 0 ? width : 1) / (height > 0 ? height : 1);
    for (const raw of text.split('\n')) {
      const line = raw.trim();
      if (!line) continue;
      const parts = line.split(/\s+/);
      if (parts.length < 2 || parts.length > 3 || !parts.slice(0, 2).every(isFloat)) continue;
      const x = Number(parts[0]), y = Number(parts[1]);
      const cls = parts.length === 3 ? parts[2] : '0';
      if (!width || !height) continue;
      objects.push({
        className: 'class ' + cls,
        xc: x / width, yc: y / height,
        w: boxWFrac, h: Math.min(1, boxHFrac),
      });
    }
    return objects;
  }

  // ---------- COCO JSON (direct port of parse_coco_json) ----------
  // Returns {imagesByFilename: Map(filename -> {width,height,annotations:[{className,xc,yc,w,h}]})}
  function parseCocoJson(data) {
    const categories = new Map();
    for (const c of data.categories || []) categories.set(c.id, c.name || ('Class_' + c.id));
    const imagesById = new Map();
    for (const im of data.images || []) {
      imagesById.set(im.id, { file_name: im.file_name || '', width: im.width || 0, height: im.height || 0, annotations: [] });
    }
    for (const ann of data.annotations || []) {
      const rec = imagesById.get(ann.image_id);
      if (rec) rec.annotations.push(ann);
    }
    const imagesByFilename = new Map();
    for (const rec of imagesById.values()) if (rec.file_name) imagesByFilename.set(rec.file_name, rec);
    return { imagesByFilename, categories };
  }

  function cocoRecordToObjects(rec, categories) {
    const { width, height } = rec;
    const objects = [];
    for (const ann of rec.annotations) {
      const className = categories.get(ann.category_id) || ('Class_' + ann.category_id);
      const [x, y, bw, bh] = ann.bbox || [0, 0, 0, 0];
      if (!width || !height) continue;
      objects.push({ className, xc: (x + bw / 2) / width, yc: (y + bh / 2) / height, w: bw / width, h: bh / height });
    }
    return objects;
  }

  // ---------- Contiguous class-id remap (direct port of export_yolo's remap step) ----------
  // classNamesInFirstSeenOrder: array of unique class names as encountered; returns a Map name->id
  // assigned by SORTED name (matches the original code's "sorted(id_to_name.keys())" ordering,
  // generalized from numeric ids to names since classes are discovered from data, not a registry).
  function buildClassRegistry(allClassNames) {
    const unique = [...new Set(allClassNames)].sort();
    const nameToId = new Map(unique.map((name, i) => [name, i]));
    return { nameToId, idToName: unique };
  }

  // ---------- Build a YOLO export in memory: {labelFiles: Map(relPath->text), imageFiles: Map(relPath->entry), dataYaml, classRegistry, warnings} ----------
  // `perImage` is an array of {stem, ext, objects: [{className,xc,yc,w,h}], sourceEntry}
  function buildYoloExport(perImage) {
    const allClassNames = [];
    for (const img of perImage) for (const o of img.objects) allClassNames.push(o.className);
    const { nameToId, idToName } = buildClassRegistry(allClassNames);
    const labelFiles = new Map();
    for (const img of perImage) {
      const lines = img.objects.map((o) => {
        const id = nameToId.get(o.className);
        return `${id} ${o.xc.toFixed(6)} ${o.yc.toFixed(6)} ${o.w.toFixed(6)} ${o.h.toFixed(6)}`;
      });
      labelFiles.set(`train/labels/${img.stem}.txt`, lines.join('\n') + (lines.length ? '\n' : ''));
    }
    const dataYaml = [
      'train: train/images', 'val: val/images', 'test: test/images',
      `nc: ${idToName.length}`, `names: [${idToName.map((n) => JSON.stringify(n)).join(', ')}]`,
    ].join('\n') + '\n';
    return { labelFiles, dataYaml, classRegistry: { nameToId, idToName } };
  }

  return { SUPPORTED_FORMATS, parseVocXml, parseYoloTxt, parsePointTxt, parseCocoJson, cocoRecordToObjects, buildClassRegistry, buildYoloExport };
});
