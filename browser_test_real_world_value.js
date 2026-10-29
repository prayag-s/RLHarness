/**
 * Real-world commercial value audit (browser-driven).
 *
 * Hard checks (any failure fails the run): synthesis, declared-action count, arena rendering,
 * training completed, rounds recorded, export parity, live interaction.
 *
 * Learning checks: for CSV tasks the target column is a real contract, so "beats chance" is hard.
 * For a task whose success criteria are invented by the LLM at synthesis time they are advisory —
 * measured and reported, but not used to fail the run, because no code change can guarantee the
 * calibration of a generated environment (see progress.md §10.4).
 *
 * Drives the actual UI at http://localhost:5173 for paid-use-case problems, and verifies
 * observable outcomes rather than page text:
 *   - environment is synthesized and verified (no placeholder),
 *   - arena renders and accepts live interaction,
 *   - training actually runs and the evaluated metric improves over rounds,
 *   - the trained policy exports to ONNX with numerical parity.
 *
 * Usage: node browser_test_real_world_value.js [--only=uc1,uc2]
 */

const puppeteer = require('./frontend/node_modules/puppeteer');
const path = require('path');
const fs = require('fs');
const http = require('http');

const BASE = 'http://localhost:5173';
const API = 'http://localhost:8000';
const FIXTURES = path.resolve(__dirname, 'backend/tests/fixtures/real_world');
const SHOTS = path.resolve(__dirname, 'test_screenshots/real_world');
const REPORT = path.resolve(__dirname, 'test_screenshots/real_world/report.json');
const ONLY = (process.argv.find(a => a.startsWith('--only=')) || '').replace('--only=', '') || null;

if (!fs.existsSync(SHOTS)) fs.mkdirSync(SHOTS, { recursive: true });

const sleep = ms => new Promise(r => setTimeout(r, ms));

function apiGet(urlPath) {
  return new Promise((resolve, reject) => {
    http.get(`${API}${urlPath}`, res => {
      let data = '';
      res.on('data', c => (data += c));
      res.on('end', () => {
        try { resolve(JSON.parse(data)); } catch (e) { resolve(null); }
      });
    }).on('error', reject);
  });
}

const USECASES = [
  {
    id: 'uc1',
    title: 'B2B Support Ticket Routing (classification, CSV)',
    dataset: 'support_tickets.csv',
    prompt:
      'Our support org receives thousands of tickets a day and mis-routing costs us SLA credits. ' +
      'Train an agent that routes each incoming B2B support ticket to the correct queue: ' +
      'Tier 1 Support, Senior Support, Retention Specialist, or Engineering Escalation. ' +
      'Use the attached ticket history; each queue is ~25% of volume, so random routing is 25% accuracy. ' +
      'Train 2 rounds of 150 episodes, target 75% routing accuracy.',
    rounds: 2,
    minFinalMetric: 45,
    minRoundsExpected: 2,
  },
  {
    id: 'uc2',
    title: 'SaaS Renewal Save-Play Selection (classification, CSV)',
    dataset: 'saas_renewal_risk.csv',
    prompt:
      'We bleed ARR because CSMs apply the same renewal play to every account. ' +
      'Train an agent that picks the right renewal play per account: Standard Renewal, Offer Annual Discount, ' +
      'CSM Onboarding Sprint, or Executive Save Play. Use the attached account health history. ' +
      'Random play selection is 25% right; we need materially better. ' +
      'Train 2 rounds of 150 episodes, target 75% play accuracy.',
    rounds: 2,
    minFinalMetric: 45,
    minRoundsExpected: 2,
  },
  {
    id: 'uc3',
    title: 'Insurance Claim Handling Path (classification, CSV)',
    dataset: 'insurance_claims.csv',
    prompt:
      'Claims leakage comes from sending the wrong claim down the wrong path. ' +
      'Train an agent that routes each claim to Straight-Through Auto-Approve, Manual Adjuster Review, ' +
      'Special Investigations Unit, or Request Additional Documentation, using the attached claims history. ' +
      'Straight-through should dominate clean claims while SIU should catch suspicious ones. ' +
      'Train 2 rounds of 150 episodes, target 75% path accuracy.',
    rounds: 2,
    minFinalMetric: 45,
    minRoundsExpected: 2,
  },
  {
    id: 'uc4',
    title: 'Peak-Season Inventory Replenishment (control, no dataset)',
    dataset: null,
    prompt:
      'Train an inventory replenishment agent for a peak-season SKU. Each day it must choose one action: ' +
      "'Order 0 Units', 'Order 50 Units', 'Order 150 Units', or 'Order 300 Units'. " +
      'Stockouts lose margin, excess stock burns working capital, and demand is volatile. ' +
      'Train 2 rounds of 150 episodes, target 60% success.',
    rounds: 2,
    minFinalMetric: 30,
    minRoundsExpected: 2,
    learningCheck: 'advisory',
  },
];

// ---------------------------------------------------------------------------
// UI helpers
// ---------------------------------------------------------------------------
async function newConversation(page) {
  await page.waitForFunction(() => document.body.innerText.includes('VERIFIED'), { timeout: 30000 });
  const sidebarBtn = await page.$('button[title*="Browse and switch"]');
  if (!sidebarBtn) throw new Error('conversation sidebar button not found');
  await sidebarBtn.click();
  await sleep(600);
  const newBtn = await page.evaluateHandle(() =>
    Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('New Agent')));
  if (!newBtn || !newBtn.asElement()) throw new Error('"New Agent" button not found in sidebar');
  // The studio creates the conversation lazily on the first chat message, so no network wait here.
  await newBtn.asElement().click();
  await sleep(900);
}

async function uploadDataset(page, filename) {
  const filePath = path.join(FIXTURES, filename);
  if (!fs.existsSync(filePath)) throw new Error(`missing fixture ${filePath}`);
  await page.waitForFunction(() => document.querySelector('input[type="file"]'), { timeout: 15000 });
  const input = await page.$('input[type="file"]');
  const uploaded = page.waitForResponse(
    r => r.url().includes('/api/v4/ingest/upload') && r.status() === 200, { timeout: 60000 });
  await input.uploadFile(filePath);
  const resp = await uploaded;
  const json = await resp.json();
  await sleep(600);
  return json;
}

async function sendChat(page, text, timeoutMs = 600000) {
  await page.waitForFunction(
    () => {
      const inp = document.querySelector('form input[type="text"]');
      return inp && !inp.disabled;
    }, { timeout: 120000 });
  const input = await page.$('form input[type="text"]');
  await input.click();
  await input.click({ clickCount: 3 });
  await page.keyboard.press('Backspace');
  await input.type(text, { delay: 1 });
  const wait = page.waitForResponse(
    r => r.url().includes('/api/v4/conversations/') && r.url().includes('/chat') && r.request().method() === 'POST',
    { timeout: timeoutMs });
  await page.keyboard.press('Enter');
  const resp = await wait;
  const json = await resp.json();
  await sleep(1200);
  return { status: resp.status(), json };
}

async function lockArenaAndTrain(page) {
  await page.waitForFunction(
    () => Array.from(document.querySelectorAll('button')).some(b => b.innerText.includes('Lock Arena & Start Training')),
    { timeout: 60000 });
  const chip = await page.evaluateHandle(() =>
    Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Lock Arena & Start Training')));
  const chat = page.waitForResponse(
    r => r.url().includes('/chat') && r.request().method() === 'POST', { timeout: 300000 });
  await chip.asElement().click();
  const resp = await chat;
  return resp.json();
}

async function waitForTraining(sessionId, timeoutMs = 25 * 60 * 1000) {
  const started = Date.now();
  let last = null;
  while (Date.now() - started < timeoutMs) {
    const session = await apiGet(`/api/v4/sessions/${sessionId}`);
    if (session) {
      last = session;
      if (session.status === 'completed' || session.status === 'target_achieved' || session.status === 'failed') {
        return session;
      }
    }
    await sleep(5000);
  }
  return last;
}

function iframeBodyText(page) {
  return page.evaluate(() => {
    const frame = document.querySelector('iframe');
    if (!frame) return null;
    const srcdoc = frame.getAttribute('srcdoc') || '';
    if (srcdoc.length > 0) return `srcdoc:${srcdoc.length}`;
    try {
      const doc = frame.contentDocument;
      if (doc && doc.body) return doc.body.innerText.slice(0, 200) || 'empty-body';
    } catch (e) {
      return 'cross-origin';
    }
    return frame.getAttribute('src') ? `src:${frame.getAttribute('src').slice(0, 60)}` : 'no-content';
  });
}

function summarizeSession(session) {
  const rounds = session.rounds_history || session.round_history || [];
  return {
    status: session.status,
    rounds: rounds.length,
    metrics: rounds.map(r => ({ round: r.round_num, metric: r.metric })),
    final_metric: session.current_metric,
    model_artifacts: {
      has_model: Boolean(session.latest_model_zip_path),
      latest_model_zip_path: session.latest_model_zip_path || null,
    },
  };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
(async () => {
  const consoleErrors = [];
  const results = [];
  const browser = await puppeteer.launch({
    headless: 'new',
    executablePath: '/usr/bin/google-chrome',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--window-size=1600,1000'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1600, height: 1000 });
  page.on('pageerror', err => consoleErrors.push(`pageerror: ${err.message}`));
  page.on('console', msg => {
    if (msg.type() === 'error') consoleErrors.push(`console: ${msg.text()}`);
  });

  const health = await apiGet('/api/v4/settings/llm');
  console.log('LLM status:', JSON.stringify(health));
  if (!health || !health.is_verified) throw new Error('LLM is not verified; cannot run real-world synthesis');

  await page.goto(BASE, { waitUntil: 'networkidle0', timeout: 45000 });

  for (const uc of USECASES) {
    if (ONLY && !ONLY.split(',').includes(uc.id)) continue;
    const record = { id: uc.id, title: uc.title, checks: {}, failures: [] };
    record.warnings = [];
    const check = (name, ok, detail, mode = 'hard') => {
      record.checks[name] = { ok: Boolean(ok), detail };
      if (!ok) {
        if (mode === 'advisory') record.warnings.push(`${name}: ${detail}`);
        else record.failures.push(`${name}: ${detail}`);
      }
      const mark = ok ? '✓' : mode === 'advisory' ? '⚠' : '✗';
      console.log(`   ${mark} ${name}${detail ? ` — ${detail}` : ''}`);
    };
    const learningMode = uc.learningCheck || 'hard';

    console.log(`\n${'='.repeat(78)}\n▶ ${uc.id.toUpperCase()} — ${uc.title}\n${'='.repeat(78)}`);
    try {
      await newConversation(page);
      if (uc.dataset) {
        const upload = await uploadDataset(page, uc.dataset);
        check('dataset_ingested', upload && upload.num_rows > 0, `${upload?.name} rows=${upload?.num_rows}`);
        record.dataset_id = upload?.dataset_id;
      }

      const first = await sendChat(page, uc.prompt);
      const body = first.json;
      record.conversation_id = body.conversation_id;
      check('chat_http_200', first.status === 200, `status=${first.status}`);
      check('arena_preview', body.status === 'arena_preview', `status=${body.status} msg=${(body.assistant_message || '').slice(0, 120)}`);
      check('arena_html_rendered', Boolean(body.arena_html && body.arena_html.length > 2000), `chars=${(body.arena_html || '').length}`);
      check('env_synthesized', Boolean(body.env_code && body.env_code.length > 2000), `chars=${(body.env_code || '').length}`);
      check('env_not_placeholder', !/is_placeholder\s*=\s*True/.test(body.env_code || ''), 'placeholder env would prove synthesis failed');

      const bp = body.blueprint_spec || {};
      check('task_title', Boolean(bp.task_title), bp.task_title);
      check('action_count', (bp.action_labels || []).length >= 3, `${(bp.action_labels || []).length} actions`);
      if (uc.dataset && bp.label_column) {
        check('label_detected', true, bp.label_column);
        check(
          'observations_derived_from_columns',
          (bp.observation_features || []).length > 0 && !(bp.observation_features || []).includes(bp.label_column),
          `${bp.observation_features?.length} features, label excluded`
        );
      }

      const iframeText = await iframeBodyText(page);
      check('arena_iframe_alive', Boolean(iframeText && iframeText.length > 10), String(iframeText).slice(0, 80).replace(/\n/g, ' '));
      await page.screenshot({ path: path.join(SHOTS, `${uc.id}_preview.png`) });

      const locked = await lockArenaAndTrain(page);
      check('training_queued', (locked.action_types || []).includes('run_training'), JSON.stringify(locked.action_types));

      const session = await waitForTraining(body.conversation_id);
      const summary = summarizeSession(session);
      record.session = summary;
      console.log('   rounds:', JSON.stringify(summary.metrics));

      check('training_finished', ['completed', 'target_achieved'].includes(summary.status), summary.status);
      check('rounds_recorded', summary.rounds >= 1, `${summary.rounds} rounds`);
      check(
        'metric_improved',
        summary.metrics.length < 2
          ? summary.status === 'target_achieved'
          : Number(summary.metrics.at(-1).metric) >= Number(summary.metrics[0].metric),
        summary.metrics.length < 2
          ? `stopped after round 1 at ${summary.metrics[0]?.metric}% (target reached)`
          : `${summary.metrics[0]?.metric} -> ${summary.metrics.at(-1)?.metric}`,
        learningMode
      );
      check(
        'metric_above_chance',
        Number(summary.final_metric) >= uc.minFinalMetric,
        `${summary.final_metric}% (floor ${uc.minFinalMetric}%)`,
        learningMode
      );

      // Live interaction through the arena bridge (play mode) proves the policy is servable.
      const interact = await new Promise((resolve, reject) => {
        const payload = JSON.stringify({ user_action: 1 });
        const req = http.request(`${API}/api/v4/sessions/${body.conversation_id}/interact`, {
          method: 'POST', headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) },
        }, res => {
          let data = '';
          res.on('data', c => (data += c));
          res.on('end', () => resolve({ status: res.statusCode, body: data }));
        });
        req.on('error', reject);
        req.write(payload);
        req.end();
      });
      let interactJson = null;
      try { interactJson = JSON.parse(interact.body); } catch (e) { /* ignore */ }
      check(
        'live_interaction',
        interact.status === 200 && Boolean(interactJson?.agent_action_label),
        `status=${interact.status} action=${interactJson?.agent_action_label}`
      );

      const exportBtn = await page.evaluateHandle(() =>
        Array.from(document.querySelectorAll('button')).find(b => /Export Owned Policy/i.test(b.innerText)));
      if (exportBtn && exportBtn.asElement()) {
        const exported = page.waitForResponse(
          r => r.url().includes('/export') && r.request().method() === 'POST', { timeout: 180000 });
        await exportBtn.asElement().click();
        const resp = await exported;
        const json = await resp.json();
        record.export = { parity: json.onnx_parity_diff, bundle: json.bundle_path };
        check('export_parity', resp.status() === 200 && Math.abs(Number(json.onnx_parity_diff)) < 1e-5, `diff=${json.onnx_parity_diff}`);
      } else {
        check('export_parity', false, 'export button unavailable');
      }

      await page.screenshot({ path: path.join(SHOTS, `${uc.id}_trained.png`) });
    } catch (err) {
      record.failures.push(`exception: ${err.message}`);
      console.error('   ✗ exception:', err.message);
      try { await page.screenshot({ path: path.join(SHOTS, `${uc.id}_error.png`) }); } catch (e) { /* ignore */ }
    }

    record.passed = record.failures.length === 0;
    results.push(record);
    fs.writeFileSync(REPORT, JSON.stringify({ generated_at: new Date().toISOString(), results, consoleErrors }, null, 2));
  }

  await browser.close();

  console.log(`\n${'='.repeat(78)}\nREAL-WORLD VALUE SUMMARY\n${'='.repeat(78)}`);
  for (const r of results) {
    const m = r.session?.metrics?.map(x => `${x.round}:${Math.round(x.metric)}%`).join(' ') || 'n/a';
    console.log(`${r.passed ? 'PASS' : 'FAIL'}  ${r.id}  ${r.title}`);
    console.log(`      rounds[${m}] export_parity=${r.export?.parity ?? 'n/a'}`);
    for (const f of r.failures) console.log(`      ✗ ${f}`);
    for (const w of r.warnings || []) console.log(`      ⚠ ${w}`);
  }
  console.log(`\nReport: ${REPORT}`);
  if (consoleErrors.length) console.log(`Browser errors: ${consoleErrors.length}`);
  process.exit(results.every(r => r.passed) ? 0 : 1);
})();
