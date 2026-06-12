import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const BASE_DIR = path.join(__dirname, '..', '..', 'data', 'Fold1');

function ensureDir() {
  fs.mkdirSync(BASE_DIR, { recursive: true });
}

function writeSplit(filename, qidStart, nQueries = 5, docsPerQuery = 4) {
  const p = path.join(BASE_DIR, filename);
  const lines = [];
  for (let qi = qidStart; qi < qidStart + nQueries; qi++) {
    for (let di = 1; di <= docsPerQuery; di++) {
      const label = [0,1,2,3,4][Math.floor(Math.random()*5)];
      const feats = [];
      for (let fi = 1; fi <= 6; fi++) {
        const val = (Math.random()).toFixed(4);
        feats.push(`${fi}:${val}`);
      }
      const line = `${label} qid:${qi} ${feats.join(' ')} # docid=Q${qi}D${di}`;
      lines.push(line);
    }
  }
  fs.writeFileSync(p, lines.join('\n') + '\n', 'utf8');
}

async function postSearch(server = 'http://localhost:3000', query = 'wireless headphones') {
  const url = server.replace(/\/$/, '') + '/api/search';
  const body = { query, weights: {} };
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const text = await res.text();
    console.log('Server response:', text);
  } catch (err) {
    console.error('Failed to call', url, err.message || err);
  }
}

async function main() {
  ensureDir();
  writeSplit('train.txt', 1, 6, 5);
  writeSplit('vali.txt', 1001, 3, 4);
  writeSplit('test.txt', 2001, 3, 4);
  console.log('Demo dataset generated in', BASE_DIR);

  console.log('\nAttempting example search request to dev server (http://localhost:3000)...');
  await postSearch();
}

main().catch(err => { console.error(err); process.exit(1); });
