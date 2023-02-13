import { configureStore } from '@reduxjs/toolkit'
import themeReducer from "./slices/theme";


import counterReducer from './slices/counterSlice'
import imageReducer from "./slices/imageSlice";
import apiConfigReducer from "./slices/apiConfigSlice";



export default configureStore({
    reducer: {
        counter: counterReducer,
        theme:themeReducer,
        image: imageReducer,
        apiConfig: apiConfigReducer
      },
})