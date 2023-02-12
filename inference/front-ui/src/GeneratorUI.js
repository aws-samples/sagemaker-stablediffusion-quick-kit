import * as React from 'react';
import { useRef, useState } from 'react';
import { useSelector, useDispatch } from 'react-redux'


import { imageOnChange } from './slices/imageSlice'
import { apiConfigLoad,fetchAPIConfigsAsync,BAES_URI  } from './slices/apiConfigSlice'


//import mui 
import Grid from '@mui/material/Unstable_Grid2'; // Grid version 1
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Input from '@mui/material/Input';
import Slider from '@mui/material/Slider';
import Radio from '@mui/material/Radio';
import RadioGroup from '@mui/material/RadioGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import FormLabel from '@mui/material/FormLabel';
import Divider from '@mui/material/Divider';
import LinearProgress from '@mui/material/LinearProgress';

import Zoom from 'react-medium-image-zoom'
import 'react-medium-image-zoom/dist/styles.css'

import Image from 'mui-image'

import axios from 'axios'



const ImageSize = [
    [768, 512],
    [704, 512],
    [640, 512],
    [576, 512],
    [512, 512],
    [512, 576],
    [512, 640],
    [512, 704],
    [512, 768],
]



  


const PostPrompt = (prompt, width, height, steps, seeds, dispath, loadingHandler) => {
    dispath(imageOnChange([]))
    const config = {
        headers: {
            "X-AIGC-Token": ""
        }
    };

    axios.post(BAES_URI+"/invocations", { "prompt": prompt, "width": width, "height": height, "steps": steps, "seed": parseInt(seeds), count: 4 }, config).then((response) => {
        console.log(response.data);
        //imageHander(response.data.result[0])
        dispath(imageOnChange(response.data.result))
        loadingHandler({ display: 'none' })

    }).catch(function (error) {
        console.log(error.response)

    })

}


const AsyncPostPrompt = (SMEndpoint,prompt, width, height, steps, seeds, dispath, loadingHandler) => {
    dispath(imageOnChange([]))
   

    console.log(SMEndpoint)
    const config = {
        headers: {
            "X-SM-Endpoint": SMEndpoint
        }
    };

    axios.post(BAES_URI+"/async_hander", { "prompt": prompt, "width": width, "height": height, "steps": steps, "seed": parseInt(seeds), "sampler": "" }, config).then((response) => {
        console.log(response.data, response.data.task_id);

        let count = 0
        const timer = setInterval(() => {
            axios.get(BAES_URI+"/task/" + response.data.task_id).then((resp) => {
                if (resp.data.status == "completed") {
                    console.log("task completed")
                    if (resp.data.images.length > 0) {
                        dispath(imageOnChange(resp.data.images))
                    }

                    loadingHandler({ display: 'none' })
                    clearInterval(timer)
                }
            }).catch(function (error) {
                count = count + 1
                if (error.response.data.status != "Pending" || count > 30) {
                    clearInterval(timer)
                }
            })

        }, 1000);
    }).catch(function (error) {
        console.log(error.response)
    })

}


export default function GeneratorUI() {

    const images = useSelector((state) => state.image.value)
    const apiConfigs = useSelector((state) => state.apiConfig.value)

    
  
    const dispatch = useDispatch()
    const refPrompt = useRef();
    
    //load apiconfig
    if (apiConfigs.length==0){
       dispatch(fetchAPIConfigsAsync())
       
    }

   



    const [styleOption, setStyleOption] = useState("anime");
    const [imageSizeIdx, setImageSizeIdx] = useState(0);
    const [steps, setSteps] = useState(20);
    const [seeds, setSeeds] = useState(-1);

    const [progress, setProgress] = React.useState(10);
    const [loading, setLoading] = React.useState({ display: 'none' });


    const onChange = e => {
        console.log(e.target.value);
        setStyleOption(e.target.value);
    };

    const imageSizeOnChange = e => {
        setImageSizeIdx(e.target.value);
    };

    const renderListOfImages = (images) => {
        return images.map(image =>
            <Zoom key="{Math.random().toString(36).slice(2, 7)}">
                <Image width={300} src={image} errorIcon={null}></Image>
            </Zoom>)
    }


    React.useEffect(() => {
        const timer = setInterval(() => {
            setProgress((progress) => (progress >= 100 ? 10 : progress + 10));
        }, 1000);
        return () => {
            clearInterval(timer);
        };
    }, []);

    

    return (

        <Box sx={{ flexGrow: 1 }}>

            <Grid container spacing={2}>

                <Grid xs={8}>

                    <TextField
                        inputRef={refPrompt}
                        id="outlined-multiline-static"
                        label="Prompt"
                        multiline
                        fullWidth
                        rows={3}
                        defaultValue="1girl, brown hair, green eyes, colorful, autumn, cumulonimbus clouds, lighting, blue sky, falling leaves, garden"
                    />

                </Grid>
                <Grid xs={4}>
                    <FormLabel id="demo-radio-buttons-group-label">Steps</FormLabel>
                    <Slider valule={steps} min={20} max={50} valueLabelDisplay="auto" onChange={(e) => setSteps(e.target.value)} />

                    <FormLabel id="demo-radio-buttons-group-label">Image Size</FormLabel>
                    <Slider value={imageSizeIdx} min={0} max={8} valueLabelDisplay="auto" onChange={imageSizeOnChange} valueLabelFormat={(x) => ImageSize[x][0] + "x" + ImageSize[x][1]} />

                </Grid>
                <Grid xs={4}>
                    <TextField
                        id="outlined-multiline-static"
                        label="Negative prompt"
                        multiline
                        fullWidth
                        rows={1}
                        placeholder="Negative prompt"
                    />
                </Grid>
                <Grid xs={4}>
                    <FormLabel id="demo-radio-buttons-group-label">Seeds</FormLabel>
                    <Input
                        id="outlined-multiline-static"
                        fullWidth
                        rows={1}
                        value={seeds}
                        onChange={(e) => setSeeds(e.target.value)}
                    />
                </Grid>
                <Grid xs={4}>
                    <RadioGroup
                        row
                        aria-labelledby="demo-row-radio-buttons-group-label"
                        name="row-radio-buttons-group"
                        onChange={onChange}
                        value={styleOption}

                    >
                        
                        {
                         apiConfigs.map((item,i)=>{
                                return <FormControlLabel key={i} value={item.sm_endpoint} control={<Radio />} label={item.label} />
                            })
                        }

                    </RadioGroup>
                </Grid>


                <Grid xs={12}></Grid>
                <Grid xs={5}></Grid>
                <Grid xs={4}>
                    <Button variant="contained" color="success" onClick={() => {
                        console.log(refPrompt.current.value, styleOption, steps, ImageSize[imageSizeIdx], seeds)
                        setLoading({})
                        setProgress(0)
                        AsyncPostPrompt( styleOption,` ${refPrompt.current.value}`, ImageSize[imageSizeIdx][0], ImageSize[imageSizeIdx][1], steps, seeds, dispatch, setLoading)

                    }}
                    >Generate</Button>

                </Grid>
                <Grid xs={12}>
                    <Divider></Divider>
                    <LinearProgress variant="determinate" value={progress} sx={loading} />
                </Grid>
                <Grid xs={4}></Grid>
                <Grid xs={4}>
                    {renderListOfImages(images)}
                </Grid>

            </Grid>
        </Box>

    );
}
