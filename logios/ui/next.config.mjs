import { defineConfig } from 'next';

export default defineConfig({
  reactStrictMode: true,
  swcMinify: true,
  output: 'standalone', //TODO: check later if that's correct
  images: {
    domains: ['your-image-domain.com'], // Add your image domains here
  },
  env: {
    API_URL: process.env.API_URL, // Add your API URL here
  },
  webpack: (config) => {
    // Custom webpack configurations can go here
    return config;
  },
});