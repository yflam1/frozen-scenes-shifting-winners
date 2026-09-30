// Entry point: import styles, render the page, wire the small bits of
// behavior (BibTeX copy button, scroll-to-top visibility).
import "./style.css";
import { renderPage, renderScrollButton } from "./render";

/** The parts of the MathJax v3 API this page relies on (loaded async in HTML). */
interface MathJaxApi {
  startup?: { promise?: Promise<void> };
  typesetPromise?: (elements?: HTMLElement[]) => Promise<void>;
}
declare global {
  interface Window {
    MathJax?: MathJaxApi;
  }
}

/**
 * Re-typeset after the page is in the DOM.
 *
 * MathJax loads via an async CDN <script> while this bundle is a deferred
 * module. Chrome can execute the warm-cached CDN script before deferred
 * modules run, and MathJax only waits for DOMContentLoaded when it sees
 * readyState "loading" — otherwise it typesets immediately, i.e. before the
 * abstract (with its \( ... \) math) has been rendered, leaving the raw
 * delimiters on screen. Safari and Firefox run the async script during
 * parsing and so always wait, which is why only Chrome shows raw \( \).
 * If MathJax is up already, queue a typeset of the finished DOM; if it is
 * still loading, its own startup pass sees the DOM as we leave it here.
 */
function typesetMath(): void {
  const mathjax = window.MathJax;
  if (!mathjax?.startup?.promise) return;
  mathjax.startup.promise
    .then(() => mathjax.typesetPromise?.())
    .catch((err: unknown) => console.error("MathJax typeset failed:", err));
}

function copyBibTeX(): void {
  const bibtex = document.getElementById("bibtex-code");
  const button = document.querySelector(".copy-bibtex-btn");
  const label = button?.querySelector(".copy-text");
  if (!(bibtex instanceof HTMLElement) || !button || !label) return;

  const flashCopied = (): void => {
    button.classList.add("copied");
    label.textContent = "Cop";
    window.setTimeout(() => {
      button.classList.remove("copied");
      label.textContent = "Copy";
    }, 2000);
  };

  navigator.clipboard
    .writeText(bibtex.textContent ?? "")
    .then(flashCopied)
    .catch((err: unknown) => {
      // Fallback for older browsers / non-secure contexts.
      console.error("Failed to copy: ", err);
      const textArea = document.createElement("textarea");
      textArea.value = bibtex.textContent ?? "";
      document.body.appendChild(textArea);
      textArea.select();
      document.execCommand("copy");
      document.body.removeChild(textArea);
      flashCopied();
    });
}

const scrollButton = renderScrollButton();
scrollButton.addEventListener("click", () => {
  window.scrollTo({ top: 0, behavior: "smooth" });
});
document.body.appendChild(scrollButton);

document.body.appendChild(renderPage(copyBibTeX));
typesetMath();

window.addEventListener("scroll", () => {
  scrollButton.classList.toggle("visible", window.scrollY > 300);
});
