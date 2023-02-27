import React from 'react'
import { useDispatch,useSelector } from 'react-redux'
import { changeColor } from './slices/theme'
import Switch from '@mui/material/Switch';
import FormLabel from '@mui/material/FormLabel';
import Grid from '@mui/material/Grid';

export default function ThemeSwith() {
  const theme= useSelector((state) => state.theme.value)

  const dispatch = useDispatch()

  return (
    <Grid>
      <Switch
                onChange={() => dispatch(changeColor())}
                inputProps={{ 'aria-label': 'controlled' }}
     />
      <FormLabel id="demo-radio-buttons-group-label">{theme}</FormLabel>
    </Grid>
  )
}