// Configuration for different environments
const config = {
  development: {
    API_URL: "http://127.0.0.1:5001"
  },
  production: {
    API_URL: process.env.REACT_APP_API_URL || "https://your-backend-url.herokuapp.com"
  }
};

const environment = process.env.NODE_ENV || 'development';
export const API_URL = config[environment].API_URL; 