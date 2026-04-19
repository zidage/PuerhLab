/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{astro,html,js,ts}'],
  theme: {
    extend: {
      colors: {
        // Surfaces — matching the editor's near-black canvas
        canvas: '#0B0B0D',
        deep: '#101013',
        base: '#141418',
        panel: '#17171C',
        card: '#1B1B21',
        separator: '#242429',
        hover: '#1F1F25',

        // Primary: Kingfisher powder-blue (logo plumage)
        action: '#6AA4C6',
        'action-2': '#4A7F9D',
        'action-tint': 'rgba(106, 164, 198, 0.14)',
        'action-glow': 'rgba(106, 164, 198, 0.28)',

        // Support: deeper steel-blue
        support: '#4A7F9D',
        'support-tint': 'rgba(74, 127, 157, 0.16)',

        // Accent: Kingfisher warm orange (beak + belly)
        accent: '#F0B44A',
        gold: '#F0B44A',
        'gold-tint': 'rgba(240, 180, 74, 0.16)',
        'gold-soft': '#E8A138',

        // Text
        mist: '#EDEDF0',
        muted: '#9A9CA3',
        'stats-muted': '#6B6D74',

        // Semantic
        info: '#6AA4C6',
        warning: '#F0B44A',
        success: '#7BB3A0',
        wine: '#7BB3A0',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        display: ['Manrope', 'Inter', 'system-ui', 'sans-serif'],
        data: ['Manrope', 'Inter', 'ui-monospace', 'system-ui', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'monospace'],
      },
      borderRadius: {
        panel: '10px',
        pill: '999px',
      },
      letterSpacing: {
        label: '0.16em',
      },
      animation: {
        'fade-in-up': 'fadeInUp 0.6s ease-out forwards',
        'scale-reveal': 'scaleReveal 0.8s ease-out forwards',
        'pulse-action': 'pulseAction 2.4s ease-in-out infinite',
        'bounce-slow': 'bounceSlow 2s ease-in-out infinite',
      },
      keyframes: {
        fadeInUp: {
          '0%': { opacity: '0', transform: 'translateY(24px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        scaleReveal: {
          '0%': { opacity: '0', transform: 'scale(1.04)' },
          '100%': { opacity: '1', transform: 'scale(1)' },
        },
        pulseAction: {
          '0%, 100%': { boxShadow: '0 0 0 0 rgba(106, 164, 198, 0.35)' },
          '50%': { boxShadow: '0 0 0 10px rgba(106, 164, 198, 0)' },
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
