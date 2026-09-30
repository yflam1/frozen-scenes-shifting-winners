import { defineConfig } from "vite";
import obfuscator from "vite-plugin-javascript-obfuscator";

// Relative base so the built site works under any GitHub Pages sub-path
// (https://<user>.github.io/<repo>/) or a project site root without
// hard-coding the hosting path.
export default defineConfig({
  base: "./",

  build: {
    target: "es2022",
    sourcemap: false,
  },

  plugins: [
    // Build-only obfuscation of the page bundle; `npm run dev` stays readable.
    // Light options: this is a content page, so only the usual string-array
    // and renaming passes — the heavier transforms are off.
    obfuscator({
      apply: "build",
      exclude: [/node_modules/],
      options: {
        compact: true,
        simplify: true,
        identifierNamesGenerator: "hexadecimal",
        renameGlobals: false,
        stringArray: true,
        stringArrayEncoding: ["base64"],
        stringArrayThreshold: 0.75,
        stringArrayRotate: true,
        stringArrayWrappersCount: 2,
        unicodeEscapeSequence: true,
        reservedStrings: [
          "__VITE_WORKER_ASSET__",
          "__VITE_ASSET__",
          "__VITE_PUBLIC_ASSET__",
        ],
        transformObjectKeys: false,
        deadCodeInjection: true,
        deadCodeInjectionThreshold: 0.2,
        controlFlowFlattening: false,
        numbersToExpressions: false,
        splitStrings: false,
        selfDefending: false,
        debugProtection: false,
      },
    }),
  ],
});
