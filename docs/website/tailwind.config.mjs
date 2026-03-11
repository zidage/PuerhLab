/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,ts}'],
  theme: {
    extend: {
      colors: {
        canvas: '#111111',
        deep: '#141414',
        base: '#1F1F1F',
        panel: '#2B2B2B',
        card: '#242424',
        separator: '#363636',
        hover: '#333333',
        gold: '#FCC704',
        'gold-tint': 'rgba(252, 199, 4, 0.18)',
        wine: '#8A0526',
        'wine-tint': 'rgba(138, 5, 38, 0.32)',
        mist: '#E6E6E6',
        muted: '#888888',
        'stats-muted': '#7B7D7C',
        info: '#5B9BD5',
        warning: '#ED7D31',
        success: '#70AD47',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        data: ['IBM Plex Sans', 'Inter', 'system-ui', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      borderRadius: {
        panel: '8px',
      },
      animation: {
        'fade-in-up': 'fadeInUp 0.6s ease-out forwards',
        'scale-reveal': 'scaleReveal 0.8s ease-out forwards',
        'pulse-gold': 'pulseGold 2s ease-in-out infinite',
        'bounce-slow': 'bounceSlow 2s ease-in-out infinite',
      },
      keyframes: {
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(30px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        scaleReveal: {
          '0%': { opacity: '0', transform: 'scale(1.05)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        pulseGold: {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(252, 199, 4, 0.4)' },
          '50%': { boxShadow: '0 0 0 8px rgba(252, 199, 4, 0)' },
        },
        bounceSlow: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(8px)' },
        },
      },
    },
  },
  plugins: [],
};
