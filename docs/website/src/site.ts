export const DEFAULT_LANG = 'en';
export const REPOSITORY_URL = 'https://github.com/zidage/PuerhLab';
export const RELEASE_BASE_URL = `${REPOSITORY_URL}/releases/download/v0.2.2`;

export function getLangPath(baseUrl: string, lang: string) {
  return lang === DEFAULT_LANG ? baseUrl : `${baseUrl}${lang}/`;
}

export function getReleaseAssetUrl(assetName: string) {
  return `${RELEASE_BASE_URL}/${assetName}`;
}
