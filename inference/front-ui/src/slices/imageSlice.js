import { createSlice } from "@reduxjs/toolkit";


export const imageSlice = createSlice({
  name: "image",
  initialState: { value:[]},
  reducers: {
    imageOnChange: (state,action) => {
      state.value= action.payload
    },
  },
});

export const { imageOnChange } = imageSlice.actions;

export default imageSlice.reducer;
