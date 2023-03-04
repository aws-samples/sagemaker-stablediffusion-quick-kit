


import Typography from '@mui/material/Typography';
import Link from '@mui/material/Link';

export default function Copyright() {

  return (
    <div>
    <p />
    <Typography variant="body2" color="text.secondary" align="center">
    {'License© '}<Link color="inherit" href="https://raw.githubusercontent.com/CompVis/stable-diffusion/main/LICENSE">
    CreativeML Open RAIL-M
      </Link><p>{' '}</p>
          
      {'Copyright © '}
      <Link color="inherit" href="https://github.com/aws-samples/sagemaker-stablediffusion-quick-kit">
        SageMaker Stable Diffusion Quick Kit
      </Link>{' '}
      {new Date().getFullYear()}
      {'.'}
    </Typography>
  </div>
  );
}