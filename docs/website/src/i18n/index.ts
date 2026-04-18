import en from './en.json';
import zh from './zh.json';
import { DEFAULT_LANG, getLangPath } from '../site';

const translations: Record<string, typeof en> = { en, zh };

export function t(lang: string) {
  return translations[lang] || translations.en;
}

export function getOtherLang(lang: string) {
  return lang === 'zh' ? 'en' : 'zh';
}

export function getOtherLangLabel(lang: string) {
  return lang === 'zh' ? 'EN' : '中文';
}

export { DEFAULT_LANG, getLangPath };
