/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        status: {
          green: "#16a34a",
          yellow: "#eab308",
          red: "#dc2626",
        },
      },
    },
  },
  plugins: [],
};
