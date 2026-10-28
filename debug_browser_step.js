const puppeteer = require('./frontend/node_modules/puppeteer');
const path = require('path');

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function waitForChatResponse(page) {
  return page.waitForResponse(
    resp => resp.url().includes('/api/v4/conversations/') &&
            resp.url().includes('/chat') &&
            resp.request().method() === 'POST' &&
            resp.status() === 200,
    { timeout: 60000 }
  );
}

async function submitChatMessage(page, text) {
  await page.waitForFunction(
    () => {
      const inp = document.querySelector('form input[type="text"]');
      return inp && !inp.disabled;
    },
    { timeout: 30000 }
  );
  const chatInput = await page.$('form input[type="text"]');
  await chatInput.click();
  // Clear any existing text
  await chatInput.click({ clickCount: 3 });
  await page.keyboard.press('Backspace');
  await sleep(50);
  await chatInput.type(text, { delay: 2 });
  await sleep(100);

  const promise = waitForChatResponse(page);
  await page.keyboard.press('Enter');
  const resp = await promise;
  const json = await resp.json();
  console.log(`   ✓ Message sent: "${text.slice(0, 30)}..." -> status: ${json.status || 'ok'}`);
  await sleep(800);
  return json;
}

(async () => {
  const browser = await puppeteer.launch({
    headless: 'new',
    executablePath: '/usr/bin/google-chrome',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--window-size=1440,900']
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 900 });

  await page.goto('http://localhost:5173', { waitUntil: 'networkidle0' });
  console.log('Page loaded');

  // Step 1: initial prompt
  const resp1 = await submitChatMessage(page, "Train an agent to triage incoming patient prior-authorization requests into 'Auto-Approve', 'Request Lab Docs', 'Urgent Physician Review', or 'Deny' using our clinical SOP PDF and claims CSV.");
  console.log('Prompt 1 finished. Arena HTML length:', resp1.arena_html?.length);

  // Step 2: restyle
  const restyle1 = await submitChatMessage(page, "Highlight abnormal vital signs in red and add a HIPAA Boundary Verified badge.");
  console.log('Restyle finished! Arena HTML length:', restyle1.arena_html?.length);
  console.log('Contains HIPAA:', restyle1.arena_html?.toLowerCase().includes('hipaa'));

  await browser.close();
  console.log('DONE!');
})();
