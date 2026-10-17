import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const MODULES = ['js/wgsl/common.js', 'js/wgsl/gen.js', 'js/wgsl/terrain.js', 'js/wgsl/sky.js', 'js/wgsl/post.js', 'js/main.js'];
const SCRIPT_TAG = '<script type="module" src="js/main.js"></script>';

const stripModuleSyntax = (src) => src
  .replace(/^import[\s\S]*?from\s+['"][^'"]+['"];\s*$/gm, '')
  .replace(/^export\s+(const|function|async function|let|class)\s/gm, '$1 ');

const code = MODULES.map((f) => stripModuleSyntax(fs.readFileSync(path.join(root, f), 'utf8'))).join('\n');
const page = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
if (!page.includes(SCRIPT_TAG)) throw new Error('index.html no longer loads js/main.js');

const out = path.join(root, process.argv[2] || 'moon.html');
fs.writeFileSync(out, page.replace(SCRIPT_TAG, () => `<script type="module">\n${code}\n</script>`));
console.log(`wrote ${path.relative(root, out)} ${(fs.statSync(out).size / 1024).toFixed(0)} KB`);
