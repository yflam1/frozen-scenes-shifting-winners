/**
 * check-build-artifacts.mjs — browser-free post-build guard for the page.
 *
 * Run after `vite build` (which also obfuscates). Gates the deploy on:
 *
 *  1. dist/index.html exists and dist/.nojekyll made it through (branch-style
 *     hosting insurance).
 *  2. No leaked Vite placeholders anywhere in dist text bundles — the
 *     obfuscator's stringArray/base64 pass can move build-time placeholders
 *     out of reach of Vite's post-build literal replace (see material-generation
 *     commit 88dfd82 for the class of bug).
 *  3. Every local asset referenced by dist/index.html resolves to a file in
 *     dist/, and the five figure PNGs are present.
 *
 * Exits non-zero on any failure so it can gate CI and the Pages deploy.
 */
import { readdir, readFile, stat } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const FRONTEND_ROOT = fileURLToPath(new URL("..", import.meta.url));
const DIST = join(FRONTEND_ROOT, "dist");

const PLACEHOLDERS = [
  "__VITE_WORKER_ASSET__",
  "__VITE_ASSET__",
  "__VITE_PUBLIC_ASSET__",
];

const REQUIRED_FILES = [
  ".nojekyll",
  "figures/content_bars.png",
  "figures/dispersion_heatmap.png",
  "figures/montage_pitch.png",
  "figures/rho_bars.png",
  "figures/template_dag.png",
];

const TEXT_EXT = new Set([".js", ".mjs", ".cjs", ".html", ".css"]);

let failures = 0;
const fail = (msg) => {
  console.error(`  FAIL: ${msg}`);
  failures++;
};
const ok = (msg) => console.log(`  PASS: ${msg}`);

async function exists(p) {
  try {
    await stat(p);
    return true;
  } catch {
    return false;
  }
}

async function collectTextFiles(dir) {
  const out = [];
  let entries;
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch {
    return out;
  }
  for (const e of entries) {
    const p = join(dir, e.name);
    if (e.isDirectory()) out.push(...(await collectTextFiles(p)));
    else if (TEXT_EXT.has(e.name.slice(e.name.lastIndexOf(".")))) out.push(p);
  }
  return out;
}

async function main() {
  console.log("build:check — dist guard");

  if (!(await exists(join(DIST, "index.html")))) {
    fail("dist/index.html missing — build did not run?");
  } else {
    ok("dist/index.html exists");
  }

  for (const f of REQUIRED_FILES) {
    if (await exists(join(DIST, f))) ok(`dist/${f} present`);
    else fail(`dist/${f} missing`);
  }

  const files = await collectTextFiles(DIST);
  for (const f of files) {
    const text = await readFile(f, "utf8");
    for (const ph of PLACEHOLDERS) {
      if (text.includes(ph)) fail(`${ph} leaked in ${f}`);
    }
  }
  ok(`no Vite placeholders in ${files.length} text files`);

  const html = await readFile(join(DIST, "index.html"), "utf8");
  const refs = [
    ...html.matchAll(/(?:src|href)="(?!https?:|#|\/\/)([^"]+)"/g),
  ].map((m) => m[1]);
  for (const ref of refs) {
    const clean = ref.replace(/^\.\//, "").split(/[?#]/)[0];
    if (clean && !(await exists(join(DIST, clean)))) {
      fail(`dangling asset reference: ${ref}`);
    }
  }
  ok(`${refs.length} local references resolve`);

  if (failures > 0) {
    console.error(`build:check FAILED (${failures})`);
    process.exit(1);
  }
  console.log("build:check passed");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
