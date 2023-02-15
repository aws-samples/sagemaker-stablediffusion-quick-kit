import { createSlice,createAsyncThunk  } from "@reduxjs/toolkit";


import axios from 'axios'

//you can use full URI
//export const BAES_URI='http://sam-s-LoadB-ZCAQMIGAB6IN-2005437513.us-east-1.elb.amazonaws.com' 
//export const BAES_URI='https://dfjcgkift2mhn.cloudfront.net' 
export const BAES_URI='' 

//init api config from lambda, dynamodb
const fetchAPIConfigsAsync = createAsyncThunk(
  'apiconfig',
  async () => {
    const response = await axios.get(BAES_URI+"/config")
    return response.data
  }
)

     
export const apiConfigSlice = createSlice({
  name: "apiConfig",
  initialState: { value: []},
  reducers: {
    apiConfigLoad: (state,action) => {
      state.value= action.payload
    },
  },
  extraReducers: (builder) => {
    // Add reducers for additional action types here, and handle loading state as needed
    builder.addCase(fetchAPIConfigsAsync.fulfilled, (state, action) => {
      // Add user to the state array
      state.value=action.payload
      
    })
  }
});

export const { apiConfigLoad} = apiConfigSlice.actions;
export  {fetchAPIConfigsAsync} ;

export default apiConfigSlice.reducer;

