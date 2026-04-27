import puppeteer from 'puppeteer';
import fs from 'fs';
import path from 'path';

const browser = await puppeteer.launch({ headless: 'new' });
const page = await browser.newPage();

// Set viewport
await page.setViewport({ width: 1440, height: 900, deviceScaleFactor: 1 });

// Navigate
await page.goto('http://localhost:5173', { waitUntil: 'networkidle2', timeout: 30000 });

// Wait for fonts and animations
await new Promise(r => setTimeout(r, 2000));

const outDir = './screenshots-preview';
if (!fs.existsSync(outDir)) fs.mkdirSync(outDir);

// Full page screenshot
await page.screenshot({ path: path.join(outDir, '01-hero.png'), fullPage: false, clip: { x: 0, y: 0, width: 1440, height: 900 } });
console.log('Captured hero');

// Scroll and capture sections
const sections = [
  { name: '02-features-top', y: 900 },
  { name: '03-features-mid', y: 2200 },
  { name: '04-features-bottom', y: 3800 },
  { name: '05-films', y: 5000 },
  { name: '06-cta', y: 6200 },
];

for (const sec of sections) {
  await page.evaluate((y) => window.scrollTo(0, y), sec.y);
  await new Promise(r => setTimeout(r, 500));
  await page.screenshot({ path: path.join(outDir, `${sec.name}.png`), fullPage: false, clip: { x: 0, y: 0, width: 1440, height: 900 } });
  console.log(`Captured ${sec.name}`);
}

await browser.close();
console.log('All screenshots saved to', outDir);
