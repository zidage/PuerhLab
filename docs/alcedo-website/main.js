import './style.css';

// ========================================
// Translations
// ========================================
const translations = {
  zh: {
    'meta.title': 'Alcedo Studio — 免费开源的专业级 RAW 图像处理软件',
    'meta.desc': 'Alcedo Studio 是一款开源免费的 RAW 图像处理软件，支持 Windows/CUDA 和 macOS/Metal，提供专业级色彩科学与高性能处理引擎。',
    'nav.features': '功能特性',
    'nav.films': '胶片预设',
    'nav.download': '下载',
    'hero.badge1': '开源免费',
    'hero.badge2': 'Windows / macOS',
    'hero.subtitle': '专业级 RAW 图像处理，不该被价格束缚',
    'hero.desc': '开源免费的 RAW 图像处理软件，支持 Windows/CUDA 与 macOS/Metal 双平台。丰富的调整工具、高性能处理引擎、强大的影像管理——为摄影师与创作者而生。',
    'hero.download': '免费下载',
    'features.label': 'FEATURES',
    'features.title': '为创作而生的<br/><span class="text-gradient">强大工具集</span>',
    'features.desc': '从色彩科学到几何校正，从基础调整到智能管理——每一步都经过精心打磨',
    'features.f1.title': '极速图像管理',
    'features.f1.desc': '响应迅速、资源占用极低的图像管理系统。支持市面上绝大多数 RAW 格式，浏览、筛选、评级一气呵成，让海量照片整理不再头疼。',
    'features.f1.tag1': '多格式 RAW 支持',
    'features.f1.tag2': '低资源占用',
    'features.f1.tag3': '快速预览',
    'features.f2.title': '双引擎色彩科学',
    'features.f2.desc': '业界领先的 ACES 与 OpenDRT 双色彩渲染管线，无论是追求真实还原还是艺术风格化，都能找到最适合场景的色彩倾向与影调表达。',
    'features.f2.tag1': 'ACES 色彩管线',
    'features.f2.tag2': 'OpenDRT 渲染',
    'features.f2.tag3': '场景化影调',
    'features.f3.title': '亿级像素，实时调整',
    'features.f3.desc': '曝光、对比度、白平衡、色调曲线……所有基础调整工具性能均经过深度优化。即便面对 <strong>1.5 亿像素</strong> 的巨无霸 RAW 文件，拖动滑块依然丝般顺滑，所见即所得。',
    'features.f3.tag1': '1.5 亿像素实时处理',
    'features.f3.tag2': 'CUDA / Metal 加速',
    'features.f3.tag3': '零延迟反馈',
    'features.f4.title': '高级色彩控制',
    'features.f4.desc': '色轮、HSL 分离调整、通道混合……将每种颜色的色相、饱和度、明度都置于你的指尖之下。想让人像肤色更暖、天空更通透？只需几秒钟。',
    'features.f4.tag1': '色轮调色',
    'features.f4.tag2': 'HSL 精细分离',
    'features.f4.tag3': '通道混合器',
    'features.f5.title': '几何与构图修正',
    'features.f5.desc': '镜头畸变校正、透视变形修复、自由裁剪，配合丰富的画面比例预设（1:1、4:5、16:9、2.39:1 等），助你轻松修正构图瑕疵，打造完美画面。',
    'features.f5.tag1': '镜头畸变校正',
    'features.f5.tag2': '透视变形修复',
    'features.f5.tag3': '多比例裁剪预设',
    'features.f6.title': '专业级导出',
    'features.f6.desc': '从 HDR 高动态范围到标准 JPEG，支持嵌入 ICC 色彩配置文件与 HDR 导出模式。无论你的作品最终流向网络、印刷还是影视流程，色彩都能精准呈现。',
    'features.f6.tag1': 'HDR 导出',
    'features.f6.tag2': 'ICC 色彩配置嵌入',
    'features.f6.tag3': '多格式输出',
    'features.f7.title': '智能影像筛选',
    'features.f7.desc': '基于拍摄日期、相机型号、镜头型号的多维度高级筛选，帮你从成千上万张照片中秒速定位目标。未来还将支持基于 AI 图像内容的智能识别筛选。',
    'features.f7.tag1': 'EXIF 多维度筛选',
    'features.f7.tag2': '日期 / 机身 / 镜头',
    'features.f7.tag3': 'AI 内容识别（即将到来）',
    'films.label': 'FILM PRESETS',
    'films.title': '经典胶片风格<br/><span class="text-gradient">一键直出</span>',
    'films.desc': '基于真实胶片特性数字复刻的 LUT 预设，从黑白经典到彩色负片，让数字影像拥有胶片般的质感与灵魂',
    'films.f1.tag': '高饱和 · 鲜艳',
    'films.f1.desc': '色彩最鲜艳的彩色负片，红与蓝极具张力，适合风光与街拍。',
    'films.f2.tag': '电影感 · 柔和',
    'films.f2.desc': '富士电影胶片，低对比、低饱和，自带温润的电影叙事感。',
    'films.f3.tag': '人像 · 细腻',
    'films.f3.desc': '人像摄影标杆，肤色还原极其细腻自然，影调柔和优雅。',
    'films.f4.tag': '真实 · 自然',
    'films.f4.desc': '追求真实还原的负片，色彩中性自然，适合记录日常。',
    'films.f5.tag': '日光 · 电影',
    'films.f5.desc': '好莱坞日光电影胶片，宽容度惊人， daylight 下的影调层次丰富。',
    'films.f6.tag': '钨丝灯 · 电影',
    'films.f6.desc': '经典钨丝灯电影胶片，夜景与室内光源下呈现迷人的冷暖对比。',
    'cta.title': '准备好开始创作了吗？',
    'cta.desc': 'Alcedo Studio 完全开源免费，无需订阅，没有功能限制。下载即可使用全部专业功能。',
    'cta.win': 'Windows 版下载',
    'cta.mac': 'macOS 版下载',
    'cta.note': '支持 CUDA (NVIDIA) 与 Metal (Apple Silicon) GPU 加速',
    'footer.tagline': '自由创作，从 Alcedo 开始',
    'footer.copy': ' Alcedo Studio. 开源软件，自由使用。'
  },
  en: {
    'meta.title': 'Alcedo Studio — Free & Open-Source Professional RAW Image Processor',
    'meta.desc': 'Alcedo Studio is a free, open-source RAW image processing software supporting Windows/CUDA and macOS/Metal, with professional color science and a high-performance processing engine.',
    'nav.features': 'Features',
    'nav.films': 'Film Presets',
    'nav.download': 'Download',
    'hero.badge1': 'Open Source & Free',
    'hero.badge2': 'Windows / macOS',
    'hero.subtitle': 'Professional RAW processing should not be held back by price',
    'hero.desc': 'A free, open-source RAW image processor supporting both Windows/CUDA and macOS/Metal. Rich adjustment tools, a high-performance engine, and powerful asset management — built for photographers and creators.',
    'hero.download': 'Free Download',
    'features.label': 'FEATURES',
    'features.title': 'A Powerful Toolkit<br/><span class="text-gradient">Built for Creation</span>',
    'features.desc': 'From color science to geometry correction, from basic adjustments to smart management — every step is carefully crafted',
    'features.f1.title': 'Blazing-Fast Asset Management',
    'features.f1.desc': 'A responsive, low-resource image management system supporting the vast majority of RAW formats on the market. Browse, filter, and rate in one seamless flow.',
    'features.f1.tag1': 'Multi-format RAW support',
    'features.f1.tag2': 'Low resource usage',
    'features.f1.tag3': 'Fast preview',
    'features.f2.title': 'Dual-Engine Color Science',
    'features.f2.desc': 'ACES and OpenDRT dual color rendering pipelines. Whether you seek faithful reproduction or artistic stylization, find the perfect color grade for every scene.',
    'features.f2.tag1': 'ACES pipeline',
    'features.f2.tag2': 'OpenDRT rendering',
    'features.f2.tag3': 'Scene-based tones',
    'features.f3.title': '150MP Real-Time Adjustments',
    'features.f3.desc': 'Exposure, contrast, white balance, tone curves... every basic tool is deeply optimized. Even with <strong>150-megapixel</strong> monster RAW files, sliders remain buttery smooth.',
    'features.f3.tag1': '150MP real-time processing',
    'features.f3.tag2': 'CUDA / Metal acceleration',
    'features.f3.tag3': 'Zero-latency feedback',
    'features.f4.title': 'Advanced Color Control',
    'features.f4.desc': 'Color wheels, HSL separation, channel mixer... every hue, saturation, and luminance value is at your fingertips. Warmer skin tones, clearer skies — just seconds away.',
    'features.f4.tag1': 'Color wheels',
    'features.f4.tag2': 'Fine HSL separation',
    'features.f4.tag3': 'Channel mixer',
    'features.f5.title': 'Geometry & Composition',
    'features.f5.desc': 'Lens distortion correction, perspective repair, free crop, with rich aspect ratio presets (1:1, 4:5, 16:9, 2.39:1, etc.). Fix composition flaws and craft the perfect frame.',
    'features.f5.tag1': 'Lens distortion correction',
    'features.f5.tag2': 'Perspective repair',
    'features.f5.tag3': 'Multi-ratio crop presets',
    'features.f6.title': 'Pro-Grade Export',
    'features.f6.desc': 'From HDR to standard JPEG, with embedded ICC profiles and HDR export modes. Whether your work goes to the web, print, or cinema pipeline, colors stay accurate.',
    'features.f6.tag1': 'HDR export',
    'features.f6.tag2': 'ICC profile embedding',
    'features.f6.tag3': 'Multi-format output',
    'features.f7.title': 'Smart Image Filtering',
    'features.f7.desc': 'Multi-dimensional advanced filtering by capture date, camera model, and lens model. Instantly locate targets among thousands of photos. AI content recognition coming soon.',
    'features.f7.tag1': 'EXIF multi-filter',
    'features.f7.tag2': 'Date / Body / Lens',
    'features.f7.tag3': 'AI recognition (coming soon)',
    'films.label': 'FILM PRESETS',
    'films.title': 'Classic Film Looks<br/><span class="text-gradient">One Click Away</span>',
    'films.desc': 'Digitally recreated LUT presets based on real film characteristics, from monochrome classics to color negatives — giving digital images the texture and soul of film.',
    'films.f1.tag': 'High Saturation · Vivid',
    'films.f1.desc': 'The most vibrant color negative film. Reds and blues are extremely punchy — ideal for landscapes and street photography.',
    'films.f2.tag': 'Cinematic · Soft',
    'films.f2.desc': 'Fujifilm motion picture stock. Low contrast, low saturation, with a warm cinematic narrative feel.',
    'films.f3.tag': 'Portrait · Delicate',
    'films.f3.desc': 'The portrait photography benchmark. Extremely delicate and natural skin tone reproduction with soft, elegant tonality.',
    'films.f4.tag': 'True · Natural',
    'films.f4.desc': 'A negative film that pursues faithful reproduction. Neutral and natural colors, perfect for everyday documentation.',
    'films.f5.tag': 'Daylight · Cinema',
    'films.f5.desc': 'Hollywood daylight motion picture stock. Amazing latitude with rich tonal layers under daylight.',
    'films.f6.tag': 'Tungsten · Cinema',
    'films.f6.desc': 'Classic tungsten motion picture stock. Captures mesmerizing warm-cool contrasts under night and indoor lighting.',
    'cta.title': 'Ready to Create?',
    'cta.desc': 'Alcedo Studio is completely free and open-source. No subscription, no feature limits. Download and access all professional features immediately.',
    'cta.win': 'Download for Windows',
    'cta.mac': 'Download for macOS',
    'cta.note': 'Accelerated by CUDA (NVIDIA) and Metal (Apple Silicon)',
    'footer.tagline': 'Create freely, start with Alcedo',
    'footer.copy': ' Alcedo Studio. Open source, free to use.'
  }
};

// ========================================
// i18n Engine
// ========================================
function detectLanguage() {
  const stored = localStorage.getItem('alcedo-lang');
  if (stored && translations[stored]) return stored;
  const navLang = navigator.language || navigator.userLanguage;
  if (navLang && navLang.toLowerCase().startsWith('zh')) return 'zh';
  return 'en';
}

let currentLang = detectLanguage();

function setLanguage(lang) {
  if (!translations[lang]) return;
  currentLang = lang;
  localStorage.setItem('alcedo-lang', lang);
  document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';

  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.getAttribute('data-i18n');
    const value = translations[lang][key];
    if (value === undefined) return;

    if (el.tagName === 'META') {
      el.setAttribute('content', value);
    } else if (el.tagName === 'TITLE') {
      el.textContent = value;
    } else {
      el.innerHTML = value;
    }
  });

  document.querySelectorAll('.lang-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.lang === lang);
  });
}

// ========================================
// Landing Page Interactions
// ========================================

document.documentElement.classList.add('js-enabled');

document.addEventListener('DOMContentLoaded', () => {
  // Initialize i18n
  setLanguage(currentLang);

  document.querySelectorAll('.lang-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const lang = btn.dataset.lang;
      if (lang && lang !== currentLang) {
        setLanguage(lang);
      }
    });
  });

  // ========================================
  // Navbar scroll effect
  // ========================================
  const navbar = document.getElementById('navbar');
  let lastScroll = 0;

  function updateNavbar() {
    const currentScroll = window.scrollY;
    if (currentScroll > 50) {
      navbar.classList.add('scrolled');
    } else {
      navbar.classList.remove('scrolled');
    }
    lastScroll = currentScroll;
  }

  window.addEventListener('scroll', updateNavbar, { passive: true });
  updateNavbar();

  // ========================================
  // Mobile menu toggle
  // ========================================
  const navToggle = document.getElementById('navToggle');
  const navMenu = document.getElementById('navMenu');

  if (navToggle && navMenu) {
    navToggle.addEventListener('click', () => {
      navToggle.classList.toggle('active');
      navMenu.classList.toggle('open');
    });

    // Close menu when clicking a link
    navMenu.querySelectorAll('.nav-link').forEach(link => {
      link.addEventListener('click', () => {
        navToggle.classList.remove('active');
        navMenu.classList.remove('open');
      });
    });
  }

  // ========================================
  // Scroll reveal animations
  // ========================================
  const revealElements = document.querySelectorAll('.reveal');

  const revealObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        revealObserver.unobserve(entry.target);
      }
    });
  }, {
    threshold: 0.1,
    rootMargin: '0px 0px -40px 0px'
  });

  revealElements.forEach(el => {
    const rect = el.getBoundingClientRect();
    if (rect.top < window.innerHeight && rect.bottom > 0) {
      el.classList.add('visible');
    }
    revealObserver.observe(el);
  });

  // ========================================
  // Smooth scroll for anchor links (fallback)
  // ========================================
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      const href = this.getAttribute('href');
      if (href === '#') return;
      const target = document.querySelector(href);
      if (target) {
        e.preventDefault();
        const offsetTop = target.getBoundingClientRect().top + window.scrollY - 80;
        window.scrollTo({
          top: offsetTop,
          behavior: 'smooth'
        });
      }
    });
  });

  // ========================================
  // Parallax effect for hero background
  // ========================================
  const heroBgImg = document.querySelector('.hero-bg-img');
  if (heroBgImg) {
    window.addEventListener('scroll', () => {
      const scrollY = window.scrollY;
      const heroHeight = document.querySelector('.hero').offsetHeight;
      if (scrollY < heroHeight) {
        const progress = scrollY / heroHeight;
        heroBgImg.style.transform = `scale(${1.05 + progress * 0.1}) translateY(${scrollY * 0.3}px)`;
        heroBgImg.style.opacity = 1 - progress * 0.8;
      }
    }, { passive: true });
  }

  // ========================================
  // Feature image tilt effect (subtle)
  // ========================================
  const featureWrappers = document.querySelectorAll('.feature-img-wrapper');

  featureWrappers.forEach(wrapper => {
    wrapper.addEventListener('mousemove', (e) => {
      const rect = wrapper.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;
      const rotateX = (y - centerY) / centerY * -2;
      const rotateY = (x - centerX) / centerX * 2;

      wrapper.style.transform = `translateY(-6px) scale(1.01) perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg)`;
    });

    wrapper.addEventListener('mouseleave', () => {
      wrapper.style.transform = '';
    });
  });

  // ========================================
  // Stagger delay for film cards
  // ========================================
  const filmCards = document.querySelectorAll('.film-card');
  filmCards.forEach((card, index) => {
    card.style.transitionDelay = `${index * 80}ms`;
  });

  // ========================================
  // Stagger delay for feature items
  // ========================================
  const featureItems = document.querySelectorAll('.feature-item');
  featureItems.forEach((item, index) => {
    item.style.transitionDelay = `${index * 100}ms`;
  });
});
