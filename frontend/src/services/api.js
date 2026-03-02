import axios from 'axios';

export const createClient = (baseURL) =>
  axios.create({
    baseURL,
    timeout: 600000,
  });
