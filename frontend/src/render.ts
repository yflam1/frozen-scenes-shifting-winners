// DOM builders for the page. Content lives in content.ts; this module only
// knows how to turn that data into nodes, mirroring the Academic Project Page
// Template's Bulma markup.
import {
  ABSTRACT_HTML,
  BIBTEX,
  HERO,
  SECTIONS,
  TEMPLATE_CREDIT_HTML,
} from "./content";
import type { Section } from "./content";

type Attrs = Record<string, string>;

function el(tag: string, cls?: string, attrs?: Attrs): HTMLElement {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  }
  return node;
}

function setHtml(node: HTMLElement, markup: string): HTMLElement {
  node.innerHTML = markup;
  return node;
}

function heading(
  parent: HTMLElement,
  cls: string,
  text: string,
  tag = "h2",
): void {
  const h = el(tag, cls);
  h.textContent = text;
  parent.appendChild(h);
}

function heroSection(): HTMLElement {
  const section = el("section", "hero");
  const body = el("div", "hero-body");
  const container = el("div", "container is-max-desktop");
  const columns = el("div", "columns is-centered");
  const column = el("div", "column has-text-centered");

  heading(column, "title is-1 publication-title", HERO.title, "h1");

  const authors = el("div", "is-size-5 publication-authors");
  for (const a of HERO.authors) {
    const block = el("span", "author-block");
    const link = el("a", undefined, { href: a.url, target: "_blank" });
    link.textContent = a.name;
    block.appendChild(link);
    if (a.sup) {
      const sup = document.createElement("sup");
      sup.innerHTML = a.sup;
      block.appendChild(sup);
    }
    block.append(",");
    authors.appendChild(block);
  }
  column.appendChild(authors);

  const affil = el("div", "is-size-5 publication-authors");
  const affilBlock = setHtml(el("span", "author-block"), HERO.affiliation);
  affil.appendChild(affilBlock);
  const note = el("span", "eql-cntrb");
  const small = document.createElement("small");
  small.appendChild(document.createElement("br"));
  small.innerHTML += HERO.note;
  note.appendChild(small);
  affil.appendChild(note);
  column.appendChild(affil);

  const linkColumn = el("div", "column has-text-centered");
  const links = el("div", "publication-links");
  for (const b of HERO.buttons) {
    const block = el("span", "link-block");
    const a = el("a", "external-link button is-normal is-rounded is-dark", {
      href: b.url,
      target: "_blank",
    });
    const icon = el("span", "icon");
    const i = document.createElement("i");
    i.className = b.icon;
    icon.appendChild(i);
    const label = el("span");
    label.textContent = b.label;
    a.append(icon, label);
    block.appendChild(a);
    links.appendChild(block);
  }
  linkColumn.appendChild(links);
  column.appendChild(linkColumn);

  columns.appendChild(column);
  container.appendChild(columns);
  body.appendChild(container);
  section.appendChild(body);
  return section;
}

function abstractSection(): HTMLElement {
  const section = el("section", "section hero is-light", { id: "abstract" });
  const container = el("div", "container is-max-desktop");
  const columns = el("div", "columns is-centered has-text-centered");
  const column = el("div", "column is-four-fifths");
  heading(column, "title is-3", "Abstract");
  const content = setHtml(
    el("div", "content has-text-justified"),
    ABSTRACT_HTML,
  );
  column.appendChild(content);
  columns.appendChild(column);
  container.appendChild(columns);
  section.appendChild(container);
  return section;
}

function paperSection(s: Section): HTMLElement {
  const light = s.light === true;
  const section = el("section", light ? "section hero is-light" : "section");
  const container = el("div", "container is-max-desktop");
  heading(container, "title is-3", s.heading);
  const content = el("div", "content paper-content");
  if (s.figure) {
    const figure = el("figure", "paper-fig", { id: s.figure.id });
    const img = document.createElement("img");
    img.src = s.figure.src;
    img.alt = s.figure.alt;
    img.loading = "lazy";
    img.style.maxWidth = s.figure.maxWidth;
    figure.appendChild(img);
    content.appendChild(figure);
  }
  for (const p of s.paragraphs) {
    setHtml(content.appendChild(el("p")), p);
  }
  container.appendChild(content);
  section.appendChild(container);
  return section;
}

function bibtexSection(onCopy: () => void): HTMLElement {
  const section = el("section", "section", { id: "BibTeX" });
  const container = el("div", "container is-max-desktop content");
  const header = el("div", "bibtex-header");
  heading(header, "title", "BibTeX");
  const button = el("button", "copy-bibtex-btn", {
    title: "Copy BibTeX to clipboard",
  });
  button.addEventListener("click", onCopy);
  const icon = document.createElement("i");
  icon.className = "fas fa-copy";
  const label = el("span", "copy-text");
  label.textContent = "Copy";
  button.append(icon, label);
  header.appendChild(button);
  container.appendChild(header);

  const pre = el("pre", undefined, { id: "bibtex-code" });
  const code = document.createElement("code");
  code.textContent = BIBTEX;
  pre.appendChild(code);
  container.appendChild(pre);
  section.appendChild(container);
  return section;
}

function footer(): HTMLElement {
  const footer = el("footer", "footer");
  const container = el("div", "container");
  const columns = el("div", "columns is-centered");
  const column = el("div", "column is-8");
  const content = el("div", "content");
  content.appendChild(setHtml(el("p"), TEMPLATE_CREDIT_HTML));
  column.appendChild(content);
  columns.appendChild(column);
  container.appendChild(columns);
  footer.appendChild(container);
  return footer;
}

/** Build the whole page under <main id="main-content">. */
export function renderPage(onCopyBibTeX: () => void): HTMLElement {
  const main = el("main", undefined, { id: "main-content" });
  main.appendChild(heroSection());
  main.appendChild(abstractSection());
  for (const s of SECTIONS) main.appendChild(paperSection(s));
  main.appendChild(bibtexSection(onCopyBibTeX));
  main.appendChild(footer());
  return main;
}

/** The template's scroll-to-top floating button. */
export function renderScrollButton(): HTMLButtonElement {
  const button = el("button", "scroll-to-top", {
    title: "Scroll to top",
    "aria-label": "Scroll to top",
  }) as HTMLButtonElement;
  const i = document.createElement("i");
  i.className = "fas fa-chevron-up";
  button.appendChild(i);
  return button;
}
