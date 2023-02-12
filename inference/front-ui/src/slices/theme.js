import { createSlice } from "@reduxjs/toolkit";


export const themeSlice = createSlice({
  name: "theme",
  initialState: { value:"light"},
  reducers: {
    changeColor: (state) => {
      state.value= state.value=="light"? "dark":"light"
      
    },
  },
});

export const { changeColor } = themeSlice.actions;

export default themeSlice.reducer;

