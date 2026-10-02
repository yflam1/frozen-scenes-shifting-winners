// Page content, separated from rendering so edits happen here, not in DOM code.
// Prose paragraphs are page-editorial text (plain-language summaries of the
// paper's results); the abstract is the paper abstract verbatim.

export interface Figure {
  id: string;
  src: string;
  alt: string;
  maxWidth: string;
}

export interface Section {
  heading: string;
  /** Alternate the background: sections render white / light down the page. */
  light?: boolean;
  figure?: Figure;
  paragraphs: string[];
}

export interface Author {
  name: string;
  url: string;
  sup: string; // superscript mark after the name ("" for none)
}

export interface ButtonLink {
  icon: string; // fontawesome/academicons class
  label: string;
  url: string;
}

export const HERO = {
  title:
    "Frozen Scenes, Shifting Winners: Configuration Fragility in Text-to-3D Evaluation",
  authors: [
    { name: "Anson Y. Lam", url: "https://yflam1.github.io/", sup: "*" },
    {
      name: "Shuqing Li",
      url: "https://shuqing-li.github.io/",
      sup: "*,&dagger;",
    },
    {
      name: "Michael R. Lyu",
      url: "https://www.cse.cuhk.edu.hk/lyu/home",
      sup: "",
    },
  ] as Author[],
  affiliation:
    "Department of Computer Science and Engineering<br>The Chinese University of Hong Kong<br>2026",
  note: "*Equal contribution. &dagger;Corresponding author.",
  buttons: [
    {
      icon: "fab fa-github",
      label: "Code",
      url: "https://github.com/yflam1/frozen-scenes-shifting-winners",
    },
    {
      icon: "ai ai-arxiv",
      label: "arXiv",
      url: "https://arxiv.org/abs/2610.00447",
    },
  ] as ButtonLink[],
};

// Paper abstract, verbatim. Math stays as \( ... \) for MathJax.
export const ABSTRACT_HTML =
  "<p>Can a text-to-3D leaderboard change when every generated scene stays fixed? We audit this question for rendered-image evaluation, where camera settings and caption wording become part of the measurement protocol. Across 300 frozen scenes from six generators, we vary eight render and caption factors for 19 alignment evaluators plus one perceptual-quality control, then test four targeted scene degradations. Peak configuration variance exceeds between-generator variance for 17/19 alignment evaluators, with prompt-bootstrap lower bounds above \\(1\\) for 11/19. Rankings are more stable than scores, yet 18/19 evaluators change their point-estimate winner under some configuration. Pairwise protocol margin envelopes show which comparisons keep their direction across the tested settings. Selected pairs have opposite pointwise intervals, but no reversal survives simultaneous inference over the full search. Thus the observed winner changes are descriptive, not confirmed changes in generator superiority. Sensitivity remains separate: no evaluator, even the prompt-free control, exceeds \\(67\\%\\) tie-adjusted directional discrimination on layout scrambling, which is diagnostic rather than human-validated ground truth. The audit separates score stability, decision uncertainty, and targeted sensitivity, and recommends reporting <em>(generator, score, card ID)</em> with protocol-dependent comparisons and selection-aware uncertainty.</p>";

// One section per figure: image first, then a single plain-language paragraph.
export const SECTIONS = [
  {
    heading: "A fixed-artifact counterfactual audit",
    figure: {
      id: "fig-thesis",
      src: "figures/montage_pitch.png",
      alt: "One frozen room rendered at three camera pitches",
      maxWidth: "82%",
    },
    paragraphs: [
      "One frozen room, rendered at three camera pitches from straight overhead to eye level. The generated scene is identical in all three images, and only the camera moved. The top-down view shows the whole layout while the eye-level view shows mostly one wall, so the same scene gives an evaluator different evidence depending on which view a paper happens to render.",
    ],
  },
  {
    heading: "Configuration variance exceeds generator separation",
    light: true,
    figure: {
      id: "fig-heatmap",
      src: "figures/dispersion_heatmap.png",
      alt: "Heatmap of score dispersion across evaluators and configuration factors",
      maxWidth: "90%",
    },
    paragraphs: [
      "How much each measurement choice moves the score, one row per evaluator. Darker cells mean larger swings, and the caption template is the biggest single factor for 18 of 19 evaluators. For 17 of 19 evaluators, the largest swing is bigger than the spread between the six generator averages, so these measurement choices can matter more than the choice of generator.",
    ],
  },
  {
    heading: "Caption wording moves every score",
    figure: {
      id: "fig-template-dag",
      src: "figures/template_dag.png",
      alt: "Directed acyclic token graph of the caption template grammar",
      maxWidth: "100%",
    },
    paragraphs: [
      "Every caption template in the audit, drawn as one small grammar. Each path from START to the prompt is a complete caption, such as &ldquo;a top-down view of {prompt}&rdquo;, and the audit scores all 337 of them. The wording alone moves the score: the best template beats the default by a median of 18% across evaluators without changing a single pixel of the scene.",
    ],
  },
  {
    heading: "Rankings are more stable than scores",
    light: true,
    figure: {
      id: "fig-rho-bars",
      src: "figures/rho_bars.png",
      alt: "Peak configuration-to-signal ratio per evaluator with prompt-bootstrap intervals",
      maxWidth: "95%",
    },
    paragraphs: [
      "One bar per evaluator: the evaluator's largest configuration swing, measured in units of the gap between generator averages. A bar past the dashed line at 1 means measurement choices moved the scores more than swapping the generator did, and for 11 of 19 evaluators the whole interval stays above 1. Rankings are steadier than raw scores, but 18 of 19 evaluators still change their top-ranked generator under some configuration, and none of these flips survives the full simultaneous check.",
    ],
  },
  {
    heading: "Sensitivity remains separate",
    figure: {
      id: "fig-content",
      src: "figures/content_bars.png",
      alt: "Bar chart of directional discrimination for removal and scrambling probes",
      maxWidth: "82%",
    },
    paragraphs: [
      "A separate question: when objects in a scene are removed or their layout is scrambled, does the score drop? Bars are grouped by evaluator, one bar per probe, and the dashed line marks the 50% chance level. No evaluator, including the prompt-free quality control, detects layout scrambling more than 67% of the time, so a stable ranking does not mean the metric understands the scene.",
    ],
  },
  {
    heading: "An audit before a superiority claim",
    light: true,
    paragraphs: [
      "The practical outcome is an audit before a superiority claim: declare the evaluator protocol, inspect method differences over a stated configuration set, quantify paired uncertainty, and test sensitivity to scene properties independently.",
      "A credible comparison should report the evaluator configuration, uncertainty, stable comparison scope, and independent sensitivity evidence, with <em>(generator, score, card ID)</em> linking each number to an auditable trail. The card ID refers to the versioned Evaluator Configuration Card proposed in the paper, which records the render and caption protocol behind a reported score.",
    ],
  },
] as Section[];

export const BIBTEX =
  "@article{lam2026frozen,\n  title={Frozen Scenes, Shifting Winners: Configuration Fragility in Text-to-3D Evaluation},\n  author={Lam, Anson Y. and Li, Shuqing and Lyu, Michael R.},\n  journal={arXiv preprint arXiv:2610.00447},\n  year={2026}\n}";

export const TEMPLATE_CREDIT_HTML =
  'This page was built using the <a href="https://github.com/eliahuhorwitz/Academic-project-page-template" target="_blank">Academic Project Page Template</a> which was adopted from the <a href="https://nerfies.github.io" target="_blank">Nerfies</a> project page. ' +
  "You are free to borrow the source code of this website, we just ask that you link back to this page in the footer. <br> This website is licensed under a " +
  '<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/" target="_blank">Creative Commons Attribution-ShareAlike 4.0 International License</a>.';
