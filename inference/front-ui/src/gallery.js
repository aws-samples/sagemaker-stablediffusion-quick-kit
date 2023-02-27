import * as React from 'react';
import { createRoot } from 'react-dom/client';
import { Provider as ReduxProvider } from 'react-redux'

import CssBaseline from '@mui/material/CssBaseline';
import { ThemeProvider } from '@mui/material/styles';

import App from './AppGallery';
import theme from './theme';
import store from './store'



const rootElement = document.getElementById('galler');
const root = createRoot(rootElement);

root.render(
  <ReduxProvider store={store}>
  <ThemeProvider theme={theme}>
    {/* CssBaseline kickstart an elegant, consistent, and simple baseline to build upon. */}
    <CssBaseline />
    <App />
  </ThemeProvider>,
  </ReduxProvider>
);
