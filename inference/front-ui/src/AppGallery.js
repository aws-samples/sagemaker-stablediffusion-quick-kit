import * as React from 'react';
import Container from '@mui/material/Container';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import Link from '@mui/material/Link';
import ProTip from './ProTip';
import Copyright from './Copyright';
import ThemeSwith from './ThemeSwith';


import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { useSelector} from 'react-redux'

// function Copyright() {
//   return (
//     <div>
//       <p />
//       <Typography variant="body2" color="text.secondary" align="center">
//       {'License© '}<Link color="inherit" href="https://raw.githubusercontent.com/CompVis/stable-diffusion/main/LICENSE">
//       CreativeML Open RAIL-M
//         </Link>{' '}
            
//         {'Copyright © '}
//         <Link color="inherit" href="https://github.com/aws-samples/sagemaker-stablediffusion-quick-kit">
//           SageMaker Stable Diffusion Quick Kit
//         </Link>{' '}
//         {new Date().getFullYear()}
//         {'.'}
//       </Typography>
//     </div>
//   );
// }

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
            AIGC Image Gallery Book
          </Typography>
          <ThemeSwith/>
         <ProTip />
         {/* <GeneratorUI /> */}
          <Copyright />
        </Box>
      </Container></ThemeProvider>
  );
}
