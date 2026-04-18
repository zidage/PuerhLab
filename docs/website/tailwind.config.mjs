/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,ts}'],
  theme: {
    extend: {
      colors: {
        canvas: '#121212',
        deep: '#171717',
        base: '#1B1B1B',
        panel: '#1F1F1F',
        card: '#232323',
        separator: '#2A2A2A',
        hover: '#262626',
        action: '#457B9D',
        support: '#2A9D8F',
        accent: '#E9C46A',
        gold: '#E9C46A',
        'gold-tint': 'rgba(233, 196, 106, 0.18)',
        'action-tint': 'rgba(69, 123, 157, 0.18)',
        'support-tint': 'rgba(42, 157, 143, 0.18)',
        wine: '#2A9D8F',
        'wine-tint': 'rgba(42, 157, 143, 0.28)',
        mist: '#F2EFE8',
        muted: '#A0A6AB',
        'stats-muted': '#7E878C',
        info: '#457B9D',
        warning: '#E9C46A',
        success: '#2A9D8F',
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
