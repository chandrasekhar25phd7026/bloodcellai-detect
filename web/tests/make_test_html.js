const fs = require('fs');
const jszipSrc = fs.readFileSync(require.resolve('jszip/dist/jszip.min.js'), 'utf8');
let html = fs.readFileSync('blood-cell-dataset-detector.html', 'utf8');
html = html.replace(
  /<script src="https:\/\/cdnjs\.cloudflare\.com\/ajax\/libs\/jszip\/3\.10\.1\/jszip\.min\.js"><\/script>/,
  '<script>' + jszipSrc + '</script>'
);
// Also drop the google fonts <link> tags (irrelevant to logic, avoids any resource loader noise)
html = html.replace(/<link[^>]*fonts\.g(oogleapis|static)[^>]*>\n?/g, '');
fs.writeFileSync('index.test.html', html);
console.log('wrote index.test.html', html.length, 'bytes; jszip inlined:', html.includes('JSZip'));
