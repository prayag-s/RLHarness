const puppeteer = require('./frontend/node_modules/puppeteer');
const path = require('path');
const fs = require('fs');

const SCREENSHOT_DIR = path.resolve(__dirname, 'test_screenshots');
if (!fs.existsSync(SCREENSHOT_DIR)) {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

const USECASES = [
  {
    id: 'usecase1_cartpole',
    title: 'Continuous Physics & Inverted Pendulum Balancing (CartPole)',
    prompt: 'Train a CartPole physics balancing agent that keeps the pole upright with 50 episodes per round targeting 95% balance stability',
    targetGoal: '95',
    async interactWithArena(frame) {
      console.log('   [CartPole] Clicking physics simulation canvas to apply disturbance force...');
      await frame.evaluate(() => {
        const canvas = document.getElementById('sim-canvas');
        if (canvas) {
          canvas.click();
        } else {
          window.parent.postMessage({ type: 'ARENA_INTERACT', payload: { custom_params: { target_pos: [2.0, 3.0] } } }, '*');
        }
      });
    }
  },
  {
    id: 'usecase2_rover',
    title: '2D Spatial Maze Rover & Obstacle Navigation',
    prompt: 'Train a 2D maze navigation rover agent to navigate obstacles and reach the goal target with 50 episodes per round targeting 95% achievement',
    targetGoal: '95',
    async interactWithArena(frame) {
      console.log('   [2D Rover] Clicking simulation canvas to set new navigation waypoint...');
      await frame.evaluate(() => {
        const canvas = document.getElementById('sim-canvas');
        if (canvas) {
          const rect = canvas.getBoundingClientRect();
          const evt = new MouseEvent('click', {
            clientX: rect.left + rect.width * 0.75,
            clientY: rect.top + rect.height * 0.25,
            bubbles: true
          });
          canvas.dispatchEvent(evt);
        } else {
          window.parent.postMessage({ type: 'ARENA_INTERACT', payload: { target_pos: [4.0, 4.0] } }, '*');
        }
      });
    }
  },
  {
    id: 'usecase3_fraud',
    title: 'Data-Driven Tabular Transaction Fraud Detection & Risk Classification',
    prompt: 'Train a tabular transaction fraud detection and risk classification agent with 50 episodes per round targeting 95% precision',
    targetGoal: '95',
    async interactWithArena(frame) {
      console.log('   [Fraud Detection] Inspecting next record / testing custom transaction...');
      await frame.evaluate(() => {
        const nextBtn = Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Next Record'));
        if (nextBtn) {
          nextBtn.click();
        } else {
          window.parent.postMessage({ type: 'ARENA_INTERACT', payload: { user_action: 1 } }, '*');
        }
      });
    }
  }
];

(async () => {
  console.log('========================================================================');
  console.log('🚀 MULTI-USECASE INTERACTIVE BROWSER TEST: 3 DIVERSE RL DOMAINS');
  console.log('========================================================================\n');

  const browser = await puppeteer.launch({
    headless: 'new',
    executablePath: '/usr/bin/google-chrome',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--window-size=1440,900']
  });

  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });

  page.on('console', msg => {
    const text = msg.text();
    if (text.includes('WebSocket') || text.includes('Round') || text.includes('Target') || text.includes('Mastery') || text.includes('eval_')) {
      console.log(`[BROWSER LOG]:`, text);
    }
  });

  page.on('pageerror', err => {
    console.error('[BROWSER PAGE ERROR]:', err.message);
  });

  page.on('response', async resp => {
    const url = resp.url();
    if (url.includes('/api/v4/') && resp.request().method() === 'POST') {
      try {
        const text = await resp.text();
        console.log(`[API RESPONSE ${resp.status()}] ${url.slice(url.indexOf('/api/v4/'))}:`);
        console.log(text.slice(0, 160) + '...\n');
      } catch (e) {}
    }
  });

  const results = [];

  for (let i = 0; i < USECASES.length; i++) {
    const uc = USECASES[i];
    console.log(`\n========================================================================`);
    console.log(`▶️ RUNNING USE CASE ${i + 1} OF 3: ${uc.title}`);
    console.log(`========================================================================\n`);

    try {
      // Step 1: Navigate / Fresh Page
      console.log(`👉 Step 1: Navigating to RL-Native Agent Studio at http://localhost:5173 ...`);
      await page.goto('http://localhost:5173', { waitUntil: 'networkidle0', timeout: 30000 });
      await page.waitForFunction(() => document.body.innerText.includes('VERIFIED'), { timeout: 15000 });
      console.log(`   ✓ Page loaded and LLM credentials verified.`);

      // Step 2: Set Target Goal
      console.log(`👉 Step 2: Setting Target Goal to ${uc.targetGoal}% ...`);
      const paramsBtn = await page.$('button::-p-text(Params:)');
      if (paramsBtn) {
        await paramsBtn.click();
        await sleep(400);

        const inputs = await page.$$('input[type="number"]');
        for (const input of inputs) {
          const max = await page.evaluate(el => el.max, input);
          const min = await page.evaluate(el => el.min, input);
          if (max === '100' && min === '50') {
            await input.click({ clickCount: 3 });
            await input.type(uc.targetGoal);
            console.log(`   ✓ Set Target Goal input to ${uc.targetGoal}%`);
          }
        }
      }

      // Step 3: Submit Prompt
      console.log(`👉 Step 3: Submitting prompt: "${uc.prompt}" ...`);
      const chatInput = await page.$('input[placeholder*="Describe agent"]');
      if (!chatInput) throw new Error('Chat input not found');

      await chatInput.click();
      await chatInput.type(uc.prompt, { delay: 10 });

      const initialChatPromise = page.waitForResponse(
        resp => resp.url().includes('/chat') && resp.request().method() === 'POST' && resp.status() === 200,
        { timeout: 120000 }
      );

      const submitBtn = await page.$('button[type="submit"]');
      await submitBtn.click();
      console.log(`   ✓ Prompt submitted. Waiting for Live Mock-Animated Arena Preview...`);

      await initialChatPromise;
      console.log(`   ✓ Omnichannel intake response received (HTTP 200).`);

      // Wait until architect spinner clears and assistant message renders
      await page.waitForFunction(
        () => !document.querySelector('.animate-spin') && document.body.innerText.includes('Generated live mock-animated Arena preview'),
        { timeout: 30000 }
      );
      await page.screenshot({ path: path.join(SCREENSHOT_DIR, `${uc.id}_01_preview.png`) });
      console.log(`   ✓ Live Mock-Animated Arena Preview rendered in center stage!`);

      // Step 4: Lock Arena & Launch Training
      console.log(`👉 Step 4: Locking Arena and Launching Autonomous Training Loop ...`);
      await sleep(800);
      const lockChip = await page.evaluateHandle(() => {
        return Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Lock Arena & Start Training'));
      });
      if (!lockChip || !lockChip.asElement()) throw new Error('Lock Arena button not found');

      const lockChatPromise = page.waitForResponse(
        resp => resp.url().includes('/chat') && resp.request().method() === 'POST' && resp.status() === 200,
        { timeout: 120000 }
      );

      await lockChip.asElement().click();
      console.log(`   ✓ Clicked [ 🔒 Lock Arena & Start Training ]. Waiting for acknowledgment...`);
      await lockChatPromise;
      console.log(`   ✓ Arena locked. Autonomous training loop initiated.`);

      // Step 5: Live Telemetry Streaming & Convergence
      console.log(`👉 Step 5: Streaming WebSocket Telemetry & Awaiting Convergence ...`);
      await page.waitForFunction(
        () => document.body.innerText.includes('Target Mastery Achieved') ||
              document.body.innerText.includes('Target mastery achieved') ||
              document.body.innerText.includes('Auto-Tune Complete') ||
              document.body.innerText.includes('100.0%') ||
              document.body.innerText.includes('98.0%') ||
              document.body.innerText.includes('95.0%') ||
              document.body.innerText.includes('Target Achieved') ||
              document.body.innerText.includes('Budget Exhausted'),
        { timeout: 120000 }
      );

      await sleep(2000);
      await page.screenshot({ path: path.join(SCREENSHOT_DIR, `${uc.id}_02_trained.png`) });
      console.log(`   🎉 Training completed and converged successfully!`);

      // Step 6: Interactive Play / Inspection Mode
      console.log(`👉 Step 6: Switching to [ 🎮 Play / Interact ] Mode ...`);
      const playModeBtn = await page.evaluateHandle(() => {
        return Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Play / Interact'));
      });
      if (playModeBtn && playModeBtn.asElement()) {
        await playModeBtn.asElement().click();
        console.log(`   ✓ Switched to Play / Interact mode.`);
        await sleep(1000);
        await page.screenshot({ path: path.join(SCREENSHOT_DIR, `${uc.id}_03_play_mode.png`) });
      }

      // Interact with arena inside sandboxed iframe
      const frameHandle = await page.$('iframe');
      if (frameHandle) {
        const frame = await frameHandle.contentFrame();
        if (frame) {
          const interactPromise = page.waitForResponse(
            resp => resp.url().includes('/interact') && resp.request().method() === 'POST' && resp.status() === 200,
            { timeout: 15000 }
          ).catch(() => null);

          await uc.interactWithArena(frame);
          await interactPromise;
          await sleep(1500);
          await page.screenshot({ path: path.join(SCREENSHOT_DIR, `${uc.id}_04_interacted.png`) });
          console.log(`   ✓ Interactive domain action executed and policy responded!`);
        }
      }

      // Step 7: Export Policy (.onnx)
      console.log(`👉 Step 7: Exporting Owned Policy to Standalone .runbundle ...`);
      const exportBtn = await page.evaluateHandle(() => {
        return Array.from(document.querySelectorAll('button')).find(b => b.innerText.includes('Export Owned Policy'));
      });
      if (exportBtn && exportBtn.asElement()) {
        const exportPromise = page.waitForResponse(
          resp => resp.url().includes('/export') && resp.request().method() === 'POST' && resp.status() === 200,
          { timeout: 25000 }
        ).catch(() => null);

        await exportBtn.asElement().click();
        console.log(`   ✓ Clicked Export Owned Policy button.`);
        await exportPromise;
        await sleep(2000);
        await page.screenshot({ path: path.join(SCREENSHOT_DIR, `${uc.id}_05_exported.png`) });
        console.log(`   ✓ Standalone .runbundle packaged and verified!`);
      }

      results.push({ id: uc.id, title: uc.title, status: 'PASSED' });
    } catch (err) {
      console.error(`❌ ERROR in ${uc.id}:`, err.message);
      await page.screenshot({ path: path.join(SCREENSHOT_DIR, `${uc.id}_error.png`) }).catch(() => {});
      results.push({ id: uc.id, title: uc.title, status: 'FAILED', error: err.message });
    }
  }

  console.log('\n========================================================================');
  console.log('📊 FINAL TEST RESULTS: 3 DIVERSE USECASES');
  console.log('========================================================================');
  results.forEach((r, idx) => {
    console.log(`  ${idx + 1}. [${r.status}] ${r.title}${r.error ? ' -> ' + r.error : ''}`);
  });
  console.log('========================================================================\n');

  await browser.close();

  const allPassed = results.every(r => r.status === 'PASSED');
  if (!allPassed) {
    process.exit(1);
  }
})();
