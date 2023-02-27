import * as React from 'react';
import Container from '@mui/material/Container';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import ProTip from './ProTip';
import GeneratorUI from './GeneratorUI';
import ThemeSwith from './ThemeSwith';
import Copyright from './Copyright';

import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { useSelector} from 'react-redux'
import SvgIcon from '@mui/material/SvgIcon';


export default function App() {
  const theme= useSelector((state) => state.theme.value)
  
  const currentTheme = createTheme({
    palette: {
      mode: theme,
    },
  });

  return (
    <ThemeProvider theme={currentTheme}>
      <CssBaseline />
      <Container maxWidth="lg">
        <Box sx={{ my: 4 }}>
          <Typography variant="h4" component="h1" gutterBottom>
            AIGC Image Generator
          </Typography>
          Power by {' '}
          {/*<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAAB4CAYAAAC3kr3rAAAACXBIWXMAACxKAAAsSgF3enRNAAALvklEQVR42u1dy7GkOhK9nrDq9TigGDdwQ15gw0Tgg1yQBzigPUvt2ChCNdwo8YLmUYU+mSlRlYuz6ehLCciT/0x+Ho/HDyb+/Oe/coVaMa2wK9yKRwRc+P+/fzeuENhnZdTD7/sN73mTE38hHz78P7NC/8oZxrmwbvSXEHMkEVLgwgPsAcmr32CA+q2Is/Th916dRRKeRUQ8mx7od3SC0ozBAikj0KRYEEjxCrZEawRtFftbM7JA2oSzGOSzpDwXWSAvhkBGXCCgqEKQoGVmQlKcCu+KLuPsOvF3xgYEskgwIwU35Rw68fpd0O415GQkI0i4UVuZGEdN0Sfew5BKxAasx4YJ6SwKi6jBhXSV5WRCJ0i4Ud8QOfZBW4+oLR9IQpnlZzdC1i5BGbUiMx02QUyD5NiTRCTcS2rMNCBk+EheNAJZZyRLjY0BmyC2YYIkadeMOGQCFkhdcJ+yMlnVTb0N/e0EiX4I4QVWc20KExy6MllFRJy63FU2ahBkX9RRQWPJXSFR74pEEK5WrH/sark2hdrVAhNkgVQUhdZx2RWI5dFaHmRHB7la7kQQH7Sjyi3YBN/VEFgRUyMOybBeaEmDoO3B3KucBEiQmaKaRbiPTW58awTx4WADQoU5x1Q7pCByqpRSRYtDMp5Bf3G9KcNiCKQa01LyzCCyWBariAbgz/YIrs4CdE8Q2UAFdJYJUvEkuq0OIyN3omR1jpzmCusWR5A1EIbfdUhu1kwdhwAVzeYK8ccE7DqOVDLURLNiQ31C0YFsxnWHwvsQUHWfCvFHD/gsfesyd8e2aAcdyGYIyURAdAfpRgLFHx44e2WZIPAEmSDz9VhpToB7GClclMTnaYAJMjFBcAawwDM9GVmlDtnnF5EpdUNokQdggsxMEJxgHTxeyAguB8TzuwRhc0SxkI+8poZOxTNBcCvQGkmbTog+v0m0loIg6TEjEARttuXbCWKRCDJhxyGRAqQSrWWuNTPQ6diMouPCBLkPQSR2HBJ59j4xXpkI4o8OMYU9YxcLmSBwY6Ees8iV6utHWrUFOf6YEZIQZ2ltyQRpnyAGMu2ZYaHmTJelQ4w/VOK1S/rMbEtEYYKUCY7Dij8ysnYSURGIjExd6aCUo25nqk6QkEqVL9A3QpAOoxCZ0PPVZ7osWK5kbjJCAbXTbN2+VciCWasYErbknZnZ+dXyNkyCZFx/hBTKguyaTVRU6JOLSBOFoIvhSAlCsCdrW0XqkAmioOOQSKG0JW4f0v31BfIgkGfSHTZZoIgxNLDzCJIgAjoOiRRKXXgeCejqgVS6CRc3oJAFwpVqfYmDJnAPBFBQLAtrFgo4/pgAY0/K7SZgmbBSd8o3To4SgkyQcUiMgAMQbAaOPySg+y0qKNNiolDuk70bQXqoOCTSRbIAz9wDxh8eKYGjKijWiYwgNyNHaRbGQ/jqkc9sAoqLeqAMnUFM929j25REWXLaWT5lLy8WQUCKaZHu2gDhpl25e7XWrF4oj7lVkmAvTTjLX8vIrXxy90EZU+C/6sLsXHEcEhnwd0CENRDNmJXmfMaLnVYgn8vAIoj+00BvTcaeLF340orikMhrLICpYgfwDqtP+u0WwGGUDwZQghT01iikh2cpCAJRM4i0QhNw4kAUPremVvGE+54AyRK9mhazr2ZEfGCUBBlL4pBIrT0AJw6GwvijydmMA1lK3TAFSZCFKq3WIEFEyUOPPCt0w+NUEH80v4rnoLxyrYoDIUjGhJj/g79KkowgiQpiztDaLqOOkBzTJBQ+1V0IAhAfCwiCpM4YG4IHQk0QnVNci9TaBqmA2WWS/Jbfo8+sz40QBNFYGQKiT4VpIL83ufs18tmNSAVMmZGNW+5IjgJZ1RAESd1C3hPky8m/xJTg66rEeEEgWVCd4QXomxMkNdtqIQiSVKAjyo/XIMiUGodEvCyPqCltxtn7OxMkQ6F/JEFMJYIMKXFIpFs2Z55FpryLyPjD3Z0cGMuzMb5JKBoyoaBuQ8Jv95HZJoUch8kEl3RigtAQRCLe/ITdewNUVVeRlq5HzuSphOyO/BCCGOogXdfS2IVbD8EzMwnCNkcE9Z5AU86RAuM/gRwZ2yIVBEEURoWSuM2+I86geWzLFqkwfKTAmA8hxwidccXS3AOw5SjtuxkBzwO1ykYT14Oo3pfefQJcNJzihWk1yXwRIO0mhR+jx4pDoBaiSYCzgMx4A1v6f1kn7PgmtEOh9AtCB6fHNSx9gamEngMQgC+jCaHMTFpgKg9JuZancHRXQBJkKHgBJmJOutsNyGBNk5lKgSBax2zhe8FwP2WCh7G5YTLF2zhs7cyVleiUNqVQuOASmJ2fajOv6zL/TjSiuTVx0oBk9qMg07ifPt2gd7AFsnImOx0GQSSSZs+Jb/rM7s2pEUGQjSQNbCOpeEokPXvsNg8UchRYNVuhqo6ttUusmUJIx7dMjmR3MsekL5Vubjm6SBk++FipYovWUl4YhwiEjNLHkCN3cVwNkrz8hl1CqnOuXJRC6XkqiEMWpJSrbowYvkQxtr602l8VsSJW7HuMMdKCTS8DwlmW2u7ViVVrYcGgLbWSLWuLKeHLqj3WAwK+/wVRIFvbG9CFNG6Nz2I4KHcaqnBmAG9MZ07Z7T/es1DtdtqlId0bYdzS2x2y1rYX1mT7chfpYBTCXqt39wf63jFWR86pY4+Un9RiVG8oFLsVowuAC47a94VtYrcPdI5B06rdvwkWGMbh467qUCA8g6KUH35BDAYThMFggjAYTBAGgwnCYDBBGAwmCIPBBGEwmCAMBhOEwWAwQRgMJgjj6/G/n26FPKBjgjC+jQB6hVlhV7gVj0i48DfTinFFzwRh3J0UQxDoJYEIj0TS/F5fMEEYdyGFDBbCI5HiDL+/NewPwPMZjNaIMSJaihgsP4cDzEwURiMWw1Ukxj/4eWFadGq0z2AABd1zC8TYE8S98cFGfnEMInII4hgjmiD9xcEcE4VBQBANINA2I9V7QZDHPyRZIn5c8stkIBFEvahRzIE8OqR4ZbIcPv9mzMmEHf2/JZKlTBQGlpvVtRTnnF3AJJgzJgrjrmR0cWnecn9w4RiFcUOCxBgCe1XS94kl+pHTw4wPSgroq4v0GZXMrY7CBUfG3QkyYBdvDMcpDOK29h7QxRIlabhHcpzC7hcDPut1lro1EX9rL0OG5G7ep8vlCjskTTTLGYzXTYxXAj5eXMNfej9Z7e5w/TJLsEpsVRixyjml0KcvLM9l/FE2D/JkMVTvzLwdiME4CLLKbHkfL2T36u+78oGp5w1Y4CEVw2T5+oB7LPRSlsIAfYYduX2y3CNMdDFZmBQ53khXWEUf4GfS4a3JGVk4E/Z57tMMKCNjZCzz/jqoSxtwrMlZH5jibNgtA22MhQs2ujB9XSCc8LeaPE2mIRpqcWxdmrYSm+vkkTwLlXimK3IKurU/z8om9dD9ErTUwISpRghDMFNuk9uZrtO7ps5eLBq36x1hzKvFYIxil0kREWLvMQyZ59Up1oN2L9bT7ZoamDP2QftsE2rcVBmvfYfw3Gyld1e2TOQ9iU/bU2r2zzwaw7aCUuXscP3Axj8Vnodt4N2YYkX2PnvlX12/tnm2DRLlzNqYw0x09wFEkLu9thsRfINKSwLdr8lpS2nlRbVOlHfxzZ5A+q9N4vVcIbmzAnsC3OU5w49zvyb/26p7axrtrkRJsUZH7Ml1hfnFNT7lGeHsOXjfe9XfgyDtxygMPMyoFvd1tf6yhtJ61mRqcdseA7R9SBDIUXTW6j4E+Xcjm2Oh+gg40jmg89rHEvv7d8y8sPt1XzdqqCAzLpcc9yPI31Yld5CGQZvlqzc1+lSo2eS4L0HOO0TZBWvHhZqaaOt5KlJ/1qX7PQRhsjApEMGzB4wS90l/egPoN06vcdqYpzqZIBHBW82u1LsQYv72yU1u4/6bMPMXxy88N8MEyZp9+ETSbD1gincnM0GgLc14aBz0DWeW9s2QkofDmCC1s2Z7Ap1130K22B87gQcmATz+D9Zp6+NS+L6QAAAAAElFTkSuQmCC" alt="SVG as an image" width="4%"/> */}
          {' Amazon SageMaker, Lambda, Cloudfront'}
          <ThemeSwith/> <br></br>
         {/* <ProTip /> */}
          <GeneratorUI />
          <Copyright />
        </Box>
      </Container></ThemeProvider>
  );
}
