import * as React from 'react';
import { useRef, useState } from 'react';
import { useSelector, useDispatch } from 'react-redux'


import { imageOnChange } from './slices/imageSlice'
import { apiConfigLoad, fetchAPIConfigsAsync, BAES_URI } from './slices/apiConfigSlice'


//import mui 
import Grid from '@mui/material/Unstable_Grid2'; // Grid version 1
import Box from '@mui/material/Box';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Input from '@mui/material/Input';
import InputBase from '@mui/material/InputBase';
import Slider from '@mui/material/Slider';
import Radio from '@mui/material/Radio';
import RadioGroup from '@mui/material/RadioGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import FormLabel from '@mui/material/FormLabel';
import Divider from '@mui/material/Divider';
import LinearProgress from '@mui/material/LinearProgress';
import ImageList from '@mui/material/ImageList';
import ImageListItem from '@mui/material/ImageListItem';

import Card from '@mui/material/Card';
import CardActions from '@mui/material/CardActions';


import ShareIcon from '@mui/icons-material/Share';
import AutorenewIcon from '@mui/icons-material/Autorenew';
import MenuBookIcon from '@mui/icons-material/MenuBook';
import Tooltip from '@mui/material/Tooltip';

import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select, { SelectChangeEvent } from '@mui/material/Select';


import Zoom from 'react-medium-image-zoom'
import 'react-medium-image-zoom/dist/styles.css'


import Alert from '@mui/material/Alert';
import IconButton from '@mui/material/IconButton';
import Collapse from '@mui/material/Collapse';

import CloseIcon from '@mui/icons-material/Close';


import ClickAwayListener from '@mui/material/ClickAwayListener';

import Image from 'mui-image'

import axios from 'axios'
import { Container } from '@mui/system';



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

    axios.post(BAES_URI + "/invocations", { "prompt": prompt, "width": width, "height": height, "steps": steps, "seed": parseInt(seeds), count: 4 }, config).then((response) => {
        console.log(response.data);
        dispath(imageOnChange(response.data.result))
        loadingHandler({ display: 'none' })

    }).catch(function (error) {
        console.log(error.response)

    })

}

const RandomSeedGenrator = () => {
    return Math.floor(Math.random() * 999999999999) + 1000000;
}


const AsyncPostPrompt = (SMEndpointInfo, prompt, width, height, steps, sampler, seeds, imageCount, dispath, loadingHandler, setSeeds) => {

    dispath(imageOnChange([]))

    console.log(SMEndpointInfo)
    const sm_endpoint = SMEndpointInfo.split(",")[0]
    const hit = SMEndpointInfo.split(",")[1]
    const config = {
        headers: {
            "X-SM-Endpoint": sm_endpoint
        }
    };
    if (hit != "") {
        prompt = hit + "," + prompt
    }
    if (imageCount > 5) {
        imageCount = 1
    }


    if (seeds == -1) {
        seeds = RandomSeedGenrator();
        setSeeds(seeds)
    }
    console.log(steps, sampler, seeds)

    axios.post(BAES_URI + "/async_hander", { "prompt": prompt, "width": width, "height": height, "steps": steps, "sampler": sampler, "seed": parseInt(seeds), "count": imageCount }, config).then((response) => {
        console.log(response.data, response.data.task_id);

        let count = 0
        const timer = setInterval(() => {
            axios.get(BAES_URI + "/task/" + response.data.task_id).then((resp) => {
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
    const [apiLoadCount, setAPILoadCount] = useState(0);
    
    

    const [styleOption, setStyleOption] = useState("");
    const [imageSizeIdx, setImageSizeIdx] = useState(0);
    const [steps, setSteps] = useState(20);
    const [imageCount, setImagesCount] = useState(1);
    const [seeds, setSeeds] = useState(-1);
    const [open, setOpen] = useState(false);
    const [sampler, setSampler] = useState("euler_a");

    const [errorMessage, setErrorMessage] = useState("");


    const [progress, setProgress] = React.useState(10);
    const [loading, setLoading] = React.useState({ display: 'none' });

    const [openTip, setOpenTip] = React.useState(false);

    if (apiConfigs.length == 0 && apiLoadCount<3) {
        dispatch(fetchAPIConfigsAsync())
        const count=apiLoadCount+1
        setAPILoadCount(count)
       
    }
   


    const handleTooltipClose = () => {
        setOpenTip(false);
    };

    const handleTooltipOpen = () => {
        setOpenTip(true);
    };


    const onChange = e => {
        console.log(e.target.value);
        setStyleOption(e.target.value);
    };

    const imageSizeOnChange = e => {
        setImageSizeIdx(e.target.value);
    };

    const samplerOnChange = e => {
        setSampler(e.target.value);
    };

    const renderListOfImages = (images) => {

        //     
        return <Grid container spacing={2}>{
            images.map(image =>
                <Grid xs={4}>
                    <Card sx={{ maxWidth: 300 }}>
                        <Zoom key="{Math.random().toString(36)}">
                            <Image width={300} src={image} errorIcon={null}></Image>
                        </Zoom>
                        <CardActions disableSpacing>
                        <Tooltip
                                title={"click share image to AIGC Event Gallery!"}
                                key="{Math.random().toString(36)}"
                            >
                            <IconButton aria-label="share" onClick={() => { handleShareClick(image, ` ${refPrompt.current.value}`, seeds) }}>
                                <MenuBookIcon />
                            </IconButton>
                        </Tooltip>{' '}
                            <Tooltip
                                title={"click copy image url!"}
                                key="{Math.random().toString(36)}"
                            >
                                <IconButton key="{Math.random().toString(36)}" aria-label="share" onClick={() => {
                                    navigator.clipboard.writeText(image)
                                   // handleTooltipOpen()
                                }}>
                                    <ShareIcon />
                                </IconButton>
                            </Tooltip>
                            
                            
                        </CardActions>
                    </Card>
                </Grid>
            )}
        </Grid>
    }

    const handleShareClick = (imageName, prompt, seed) => {
        console.log(imageName, prompt, seed)
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
                        defaultValue="a photo of an astronaut riding a horse on moon"
                    //defaultValue="1girl, brown hair, green eyes, colorful, autumn, cumulonimbus clouds, lighting, blue sky, falling leaves, garden"
                    />

                </Grid>


                <Grid xs={4}>
                    <FormControl sx={{ minWidth: 120 }}>
                        <FormLabel id="demo-radio-buttons-group-label">Sampler</FormLabel>
                        <Select
                            labelId="demo-simple-select-label"
                            id="demo-simple-select"
                            value={sampler}
                            label="Sampler"
                            onChange={samplerOnChange}
                        >
                            <MenuItem value={"euler_a"}>Euler Ancestral</MenuItem>
                            {/* <MenuItem value={"eular"}>Euler</MenuItem>
                        <MenuItem value={"heun"}>Heun</MenuItem>
                        <MenuItem value={"lms"}>LMS</MenuItem>
                        <MenuItem value={"dpm2"}>KDPM2</MenuItem>
                        <MenuItem value={"dpm2_a"}>KDPM2 Ancestral</MenuItem> */}
                            <MenuItem value={"ddim"}>DDIM</MenuItem>
                        </Select>
                    </FormControl>
                    <br />
                    <FormLabel id="demo-radio-buttons-group-label">Steps</FormLabel>
                    <Slider valule={steps} min={20} max={50} valueLabelDisplay="auto" onChange={(e) => setSteps(e.target.value)} />



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
                    <InputBase
                        id="outlined-multiline-static"
                        sx={{ ml: 4, flex: 1 }}
                        placeholder="Stable Diffusion Seed "
                        value={seeds}
                        onChange={(e) => setSeeds(e.target.value)}
                    />


                    <IconButton aria-label="share" color="primary" >
                        <AutorenewIcon onClick={() => { setSeeds(RandomSeedGenrator()) }} />
                    </IconButton>


                </Grid>
                <Grid xs={4}>
                    <FormLabel id="demo-radio-buttons-group-label">Image Size</FormLabel>
                    <Slider value={imageSizeIdx} min={0} max={8} valueLabelDisplay="auto" onChange={imageSizeOnChange} valueLabelFormat={(x) => ImageSize[x][0] + "x" + ImageSize[x][1]} />
                    <FormLabel id="demo-count-buttons-group-label">Image Count</FormLabel>
                    <Slider valule={steps} min={1} max={4} valueLabelDisplay="auto" onChange={(e) => setImagesCount(e.target.value)} />
                    <FormLabel id="demo-radio-buttons-group-label">Model</FormLabel>
                    <RadioGroup
                        row
                        aria-labelledby="demo-row-radio-buttons-group-label"
                        name="row-radio-buttons-group"
                        onChange={onChange}
                        value={styleOption}
                    >

                        {
                            apiConfigs.map((item, i) => {
                                return <FormControlLabel key={i} value={item.sm_endpoint + "," + item.hit} control={<Radio />} label={item.label} />
                            })
                        }

                    </RadioGroup>
                </Grid>

                <Grid xs={12}>
                    <Collapse in={open}>
                        <Alert severity="error"
                            action={
                                <IconButton
                                    aria-label="close"
                                    color="inherit"
                                    size="small"
                                    onClick={() => {
                                        setOpen(false);
                                    }}
                                >
                                    <CloseIcon fontSize="inherit" />
                                </IconButton>
                            }
                            sx={{ mb: 2 }}
                        >
                            {errorMessage}
                        </Alert>
                    </Collapse>

                </Grid>
                <Grid xs={5}>
                </Grid>
                <Grid xs={4}>
                    <Button variant="contained" color="success" onClick={() => {
                        if (styleOption == "") {
                            setErrorMessage("your need select a model.");
                            setOpen(true);
                            return
                        }
                        console.log(refPrompt.current.value, styleOption, steps, sampler, ImageSize[imageSizeIdx], seeds)

                        setOpen(false)
                        setLoading({})
                        setProgress(0)
                        AsyncPostPrompt(styleOption, ` ${refPrompt.current.value}`, ImageSize[imageSizeIdx][0], ImageSize[imageSizeIdx][1], steps, sampler, seeds, imageCount, dispatch, setLoading, setSeeds)

                    }}
                    >Generate</Button>

                </Grid>
                <Grid xs={12}>
                    <Divider></Divider>
                    <LinearProgress variant="determinate" value={progress} sx={loading} />
                </Grid>
                <Grid xs={12}>
                    {renderListOfImages(images)}
                </Grid>
            </Grid>
        </Box>

    );
}
