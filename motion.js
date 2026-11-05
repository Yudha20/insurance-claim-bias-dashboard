// Page motion. Streamlit pages change through a server rerun, which overwrites old
// elements one at a time. So on a nav click the page body leaves at once, is held
// hidden while the rerun happens, and springs back only once fully rendered.
// Never write a less-than sign directly before a letter, digit, slash or "!" here:
// the sanitiser would delete this script. motion.py checks for this.
(function () {
  if (window.__csMotion) return;
  window.__csMotion = true;

  const root = document.documentElement;
  root.classList.add("cs-js");
  const $ = (selector, scope) => (scope || document).querySelector(selector);
  const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

  // A damped spring sampled into a CSS easing, so each element overshoots then settles.
  function springEasing(damping, frequency, duration) {
    const omega = 2 * Math.PI * frequency;
    const decay = damping * omega;
    const damped = omega * Math.sqrt(1 - damping * damping);
    const points = [];
    for (let i = 0; i < 61; i++) {
      const t = (i / 60) * duration;
      const value = 1 - Math.exp(-decay * t) * (Math.cos(damped * t) + (decay / damped) * Math.sin(damped * t));
      points.push(+value.toFixed(4));
    }
    points[60] = 1;
    return "linear(" + points.join(",") + ")";
  }
  const supportsLinear = CSS.supports("animation-timing-function", "linear(0, 1)");
  const SPRING = supportsLinear ? springEasing(0.52, 1.75, 0.95) : "cubic-bezier(.34,1.4,.64,1)";
  root.style.setProperty("--cs-spring", SPRING);

  const page = () => $(".st-key-page_view");
  const marker = () => $("#cs-rendered");
  const navButtons = () => Array.from(document.querySelectorAll('[data-testid="stSidebarUserContent"] [class*="st-key-nav_"]'));
  const slugOf = (el) => {
    const match = /st-key-nav_(\S+)/.exec(el.className);
    return match ? match[1] : null;
  };
  const navIndex = (slug) => navButtons().findIndex((el) => slugOf(el) === slug);
  const currentSlug = () => (marker() ? marker().dataset.view : "Claims");

  // ---------- Glass pill: a real spring, so an interrupted move keeps its momentum ----------
  const pill = { el: null, x: 0, y: 0, w: 0, h: 0, vx: 0, vy: 0, vw: 0, vh: 0, tx: 0, ty: 0, tw: 0, th: 0, frame: 0, last: 0, placed: false };
  let activeSlug = null;

  function paintActive(slug) {
    activeSlug = slug;
    let style = document.getElementById("cs-active-style");
    if (!style) {
      style = document.createElement("style");
      style.id = "cs-active-style";
      document.head.appendChild(style);
    }
    const key = 'html.cs-js [data-testid="stSidebarUserContent"] .st-key-nav_' + CSS.escape(slug);
    style.textContent = key + " button[kind]{color:var(--accent)!important}" + key + " button[kind] p{font-weight:700!important}";
  }

  function targetRect(slug) {
    const host = $('[data-testid="stSidebarUserContent"]');
    const nav = $(".st-key-nav_" + CSS.escape(slug));
    const button = nav && $("button", nav);
    if (!host || !button) return null;
    const outer = host.getBoundingClientRect();
    const inner = button.getBoundingClientRect();
    return { x: inner.left - outer.left, y: inner.top - outer.top, w: inner.width, h: inner.height };
  }

  function drawPill() {
    const vy = Math.abs(pill.vy) + Math.abs(pill.vx);
    const stretch = Math.min(0.34, vy / 2600);
    const pinch = Math.min(0.1, vy / 9000);
    const vertical = Math.abs(pill.vy) >= Math.abs(pill.vx);
    const sx = vertical ? 1 - pinch : 1 + stretch;
    const sy = vertical ? 1 + stretch : 1 - pinch;
    pill.el.style.width = Math.max(0, pill.w) + "px";
    pill.el.style.height = Math.max(0, pill.h) + "px";
    pill.el.style.transform = "translate3d(" + pill.x + "px," + pill.y + "px,0) scale(" + sx.toFixed(4) + "," + sy.toFixed(4) + ")";
  }

  function stepPill(now) {
    const dt = Math.min(0.032, (now - pill.last) / 1000 || 0.016);
    pill.last = now;
    const stiffness = 340;
    const damping = 21;
    let moving = false;
    [["x", "vx", "tx"], ["y", "vy", "ty"], ["w", "vw", "tw"], ["h", "vh", "th"]].forEach(([p, v, t]) => {
      pill[v] += (stiffness * (pill[t] - pill[p]) - damping * pill[v]) * dt;
      pill[p] += pill[v] * dt;
      if (Math.abs(pill[t] - pill[p]) > 0.1 || Math.abs(pill[v]) > 1) moving = true;
    });
    if (!moving) {
      ["x", "y", "w", "h"].forEach((p) => { pill[p] = pill["t" + p]; pill["v" + p] = 0; });
    }
    drawPill();
    pill.frame = moving ? requestAnimationFrame(stepPill) : 0;
  }

  function movePill(slug, snap) {
    const host = $('[data-testid="stSidebarUserContent"]');
    const rect = targetRect(slug);
    if (!host || !rect) return;
    if (!pill.el || !pill.el.isConnected) {
      pill.el = document.createElement("div");
      pill.el.className = "cs-pill";
      pill.el.setAttribute("aria-hidden", "true");
      host.prepend(pill.el);
      pill.placed = false;
    }
    pill.tx = rect.x; pill.ty = rect.y; pill.tw = rect.w; pill.th = rect.h;
    if (snap || !pill.placed || reduced()) {
      pill.x = rect.x; pill.y = rect.y; pill.w = rect.w; pill.h = rect.h;
      pill.vx = pill.vy = pill.vw = pill.vh = 0;
      pill.placed = true;
      if (pill.frame) cancelAnimationFrame(pill.frame);
      pill.frame = 0;
      drawPill();
    } else if (!pill.frame) {
      pill.last = performance.now();
      pill.frame = requestAnimationFrame(stepPill);
    }
  }

  // ---------- Page enter and leave ----------
  const animated = new Set();
  function play(el, frames, options) {
    const animation = el.animate(frames, options);
    animated.add(animation);
    const done = () => animated.delete(animation);
    animation.finished.then(done, done);
    return animation;
  }
  function stopAll() {
    animated.forEach((animation) => animation.cancel());
    animated.clear();
  }

  function enter(elements, direction, distance) {
    if (reduced()) {
      elements.forEach((el) => play(el, [{ opacity: 0 }, { opacity: 1 }], { duration: 140, easing: "ease-out" }));
      return;
    }
    elements.forEach((el, i) => {
      const delay = Math.min(i, 7) * 34;
      play(el, [
        { transform: "translateY(" + direction * distance + "px) scale(.972)" },
        { transform: "none" },
      ], { duration: 950, delay, easing: SPRING, fill: "backwards" });
      play(el, [{ opacity: 0 }, { opacity: 1 }], { duration: 240, delay, easing: "ease-out", fill: "backwards" });
    });
  }

  function leave(direction, then) {
    const body = page();
    if (!body || reduced()) { then(); return; }
    const animation = play(body, [
      { opacity: 1, transform: "none", filter: "blur(0)" },
      { opacity: 0, transform: "translateY(" + direction * -12 + "px) scale(.982)", filter: "blur(3px)" },
    ], { duration: 170, easing: "cubic-bezier(.4,0,1,1)", fill: "forwards" });
    // Hold first, then cancel: the exit's held end state must not outlive the exit.
    animation.finished.then(() => { then(); animation.cancel(); }, () => {});
  }

  const visibleChildren = (el) => (el ? Array.from(el.children).filter((child) => child.offsetHeight > 0) : []);

  // Charts and tables finish drawing after their elements arrive.
  function contentDrawn() {
    const body = page();
    if (!body) return true;
    if ($('[data-testid="stSkeleton"],[data-testid="stSkeletonElement"]', body)) return false;
    const charts = Array.from(body.querySelectorAll('[data-testid="stVegaLiteChart"]'));
    if (charts.some((chart) => !$("svg,canvas", chart))) return false;
    const tables = Array.from(body.querySelectorAll('[data-testid="stDataFrame"]'));
    return !tables.some((table) => !$("canvas", table));
  }

  // ---------- Page change: leave, hold, reveal ----------
  const SKELETON_DELAY = 90;
  const SKELETON_MIN = 260;
  const HARD_LIMIT = 30000;
  const DRAW_LIMIT = 1500;
  let pending = null;
  let poll = 0;

  function skeleton(slug) {
    return $('.cs-sk-page[data-view="' + CSS.escape(slug) + '"]');
  }
  function showSkeleton() {
    if (!pending || pending.shown) return;
    const el = skeleton(pending.slug);
    if (!el) return;
    pending.shown = el;
    pending.shownAt = performance.now();
    el.classList.add("is-on");
    enter(reduced() ? [el] : visibleChildren(el.firstElementChild), pending.direction, 14);
  }
  function hideSkeleton(el) {
    if (!el) return;
    const finish = () => el.classList.remove("is-on");
    if (reduced()) { finish(); return; }
    el.animate([{ opacity: 1 }, { opacity: 0 }], { duration: 200, easing: "ease-out" }).finished.then(finish, finish);
  }

  function isReady() {
    const now = performance.now();
    const item = marker();
    const body = page();
    if ($('[data-testid="stException"]')) return true;
    if (now - pending.startedAt > HARD_LIMIT) return true;
    if (!item || item.dataset.view !== pending.slug || +item.dataset.run <= pending.run) return false;
    if (item.closest('[data-stale="true"]')) return false;
    if (body && $('[data-stale="true"]', body)) return false;
    pending.renderedAt = pending.renderedAt || now;
    return contentDrawn() || now - pending.renderedAt > DRAW_LIMIT;
  }

  function reveal() {
    const { direction, shown } = pending;
    pending = null;
    stopAll();
    root.classList.remove("cs-hold");
    hideSkeleton(shown);
    paintActive(currentSlug());
    movePill(currentSlug());
    enter(visibleChildren(page()), direction, 30);
  }

  function tick() {
    if (!pending) { clearInterval(poll); poll = 0; return; }
    const now = performance.now();
    if (!pending.left) return;
    if (!pending.shown && now - pending.leftAt > SKELETON_DELAY && !isReady()) showSkeleton();
    if (!isReady()) return;
    if (pending.shown && now - pending.shownAt < SKELETON_MIN) return;
    reveal();
  }

  function go(slug) {
    const from = pending ? pending.slug : currentSlug();
    if (slug === from) return;
    const direction = navIndex(slug) >= navIndex(from) ? 1 : -1;
    const item = marker();
    const run = item ? +item.dataset.run : 0;
    paintActive(slug);
    movePill(slug);
    const main = $('[data-testid="stMain"]');
    if (main && main.scrollTop > 0) main.scrollTo({ top: 0, behavior: reduced() ? "auto" : "smooth" });

    if (pending) {
      // Changed mind mid-transition: the page is already leaving or hidden, so only retarget.
      if (pending.shown) { pending.shown.classList.remove("is-on"); pending.shown = null; }
      Object.assign(pending, { slug, direction, run, startedAt: performance.now(), renderedAt: 0, leftAt: performance.now() });
      return;
    }
    stopAll();
    const state = { slug, direction, run, startedAt: performance.now(), left: false, shown: null };
    pending = state;
    leave(direction, () => {
      root.classList.add("cs-hold");
      stopAll();
      state.left = true;
      state.leftAt = performance.now();
    });
    if (!poll) poll = setInterval(tick, 40);
  }

  // Capture phase, so the pill moves on the click itself, before Streamlit handles it.
  document.addEventListener("click", (event) => {
    const button = event.target.closest && event.target.closest('[data-testid="stSidebarUserContent"] button');
    const wrapper = button && button.closest('[class*="st-key-nav_"]');
    const slug = wrapper && slugOf(wrapper);
    if (slug) go(slug);
  }, true);

  // ---------- First paint ----------
  const bootStart = performance.now();
  let renderedAt = 0;
  const bootPoll = setInterval(() => {
    const item = marker();
    const now = performance.now();
    if (!item || item.closest('[data-stale="true"]')) {
      if (now - bootStart < HARD_LIMIT) return;
    }
    renderedAt = renderedAt || now;
    if (!contentDrawn() && now - renderedAt < DRAW_LIMIT) return;
    clearInterval(bootPoll);
    paintActive(currentSlug());
    movePill(currentSlug(), true);
    root.classList.add("cs-ready");
    setTimeout(() => root.classList.add("cs-booted"), 360);
    const frame = $('[data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"]');
    const chrome = frame ? visibleChildren(frame).filter((el) => !el.contains(page() || null)) : [];
    enter(chrome.concat(visibleChildren(page())), 1, 12);
  }, 40);

  // ---------- Keep the pill attached and aligned as Streamlit redraws ----------
  let queued = false;
  function syncPill() {
    queued = false;
    const slug = pending ? pending.slug : currentSlug();
    if (!root.classList.contains("cs-ready")) return;
    if (slug !== activeSlug) paintActive(slug);
    if (!pill.frame) movePill(slug, !pill.el || !pill.el.isConnected);
  }
  const queueSync = () => {
    if (!queued) { queued = true; requestAnimationFrame(syncPill); }
  };
  new MutationObserver(queueSync).observe(document.body, { childList: true, subtree: true });
  addEventListener("resize", () => movePill(activeSlug || currentSlug(), true));
  // The sidebar checkbox resizes the rail instantly, so the pill follows without springing.
  document.addEventListener("change", (event) => {
    if (event.target.id === "claims-navigation") requestAnimationFrame(() => movePill(activeSlug || currentSlug(), true));
  });
})();
