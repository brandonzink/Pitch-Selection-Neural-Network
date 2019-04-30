In order to run the code, use "python Pitch_Selection_Network.py" while in the directory.  
  
If you'd like to edit what pitcher it is running on, edit the name at the very bottom of the code (line 250). Make sure to not change the
format as it is used to call other functions, it needs to be "get_pitcher_model('firstname', 'lastname'). Some examples for reference are
included in the code and commented out so you can switch them in if you'd like to test other pitchers. As long as they have been in the 
league for a few years, it will work.  
  
At the end of the execution, it will print the dataframe of actual pitch and predicted pitch, as well as the accuracy (as defined in the
writeup) and the random accuracy, that is, using the same accuracy scoring but with a random pitch location selected based on the 
specific pitchers pitch distribution. 

We've included some models and pitcher data in the "Data" folder for you so that it will automatically run and not have to download a bunch
of data. If you want to see it run from scratch, delete the .csv (for pitcher data) and .h5/.json (the model) that correspond to the pitcher
name. Be sure to see the "NOTE" if running on a new pitcher. 
  
NOTE: IF YOU ARE RUNNING ON A NEW PITCHER FOR THE FIRST TIME, IT MAY FAIL ONCE. JUST RERUN THE CODE AND IT WILL WORK. 